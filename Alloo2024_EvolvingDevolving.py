# ---------------------------------------------------------------------------------
# Written by Samantha Alloo // Date 23.10.2024
# This script contains code that will calculate the multimodal images of a sample
# using the various perspectives of the speckle-based X-ray imaging Fokker--Planck
# equation: 1) evolving and 2) devolving. Within, there is a single- and multiple-
# exposure algorithm for each perspective.
# ---------------------------------------------------------------------------------
# Importing required modules
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy
from scipy import ndimage, misc
from PIL import Image
import time
from scipy.ndimage import median_filter, gaussian_filter
import fabio
import pyedflib
import h5py
# ---------------------------------------------------------------------------------
# Defining additional functions
def kspace_kykx(image_shape: tuple, pixel_size: float = 1):
    # Multiply by 2pi for correct values, since DFT has 2pi in exponent
    rows = image_shape[0]
    columns = image_shape[1]
    ky = 2*math.pi*scipy.fft.fftfreq(rows, d=pixel_size) # spatial frequencies relating to "rows" in real space
    kx = 2*math.pi*scipy.fft.fftfreq(columns, d=pixel_size) # spatial frequencies relating to "columns" in real space
    return ky, kx
def invLaplacian(image,pixel_size):
    # Need to mirror the image to enforce periodicity
    flip = np.concatenate((image, np.flipud(image)), axis=0)
    flip = np.concatenate((flip, np.fliplr(flip)), axis=1)

    ky, kx = kspace_kykx(flip.shape,pixel_size)
    ky2 = ky**2
    kx2 = kx**2

    kr2 = np.add.outer(ky2, kx2)
    regkr2 = 0.0001
    ftimage = np.fft.fft2(flip)
    regdiv = 1/(kr2+regkr2)
    invlapimageflip = -1*np.fft.ifft2(regdiv*ftimage)

    row = int(image.shape[0])
    column = int(image.shape[1])

    invlap = np.real(invlapimageflip[0:row,0:column])
    return invlap, regkr2
def xderivative(image,pixel_size):
    im_mirror_h = np.concatenate((image, np.fliplr(image)), axis=1) # Doing mirroring (horizontally and vertically) to enforce periodicity
    im_mirror_v = np.concatenate((im_mirror_h, np.flipud(im_mirror_h)), axis=0)

    ky, kx = kspace_kykx(im_mirror_v.shape, pixel_size)

    ky0 = ky * 0
    i_kx = kx * (np.zeros((kx.shape),
                          dtype=np.complex128) + 0 + 1j)  # (i) * derivative along rows of DF, has "0" in the real components, and "d(DF)/dx" in the complex

    i_kx_0ky = np.add.outer(ky0, i_kx)

    fft_im = scipy.fft.fft2(im_mirror_v)
    kernfft_dx = i_kx_0ky * fft_im
    dx_im = np.real(scipy.fft.ifft2(kernfft_dx))

    dx_im_crop = dx_im[:image.shape[0],:image.shape[1]]

    return dx_im_crop
def yderivative(image,pixel_size):
    im_mirror_h = np.concatenate((image, np.fliplr(image)), axis=1) # Doing mirroring (horizontally and vertically) to enforce periodicity
    im_mirror_v = np.concatenate((im_mirror_h, np.flipud(im_mirror_h)), axis=0)

    ky, kx = kspace_kykx(im_mirror_v.shape, pixel_size)

    kx0 = kx * 0
    i_ky = ky * (np.zeros((ky.shape),
                          dtype=np.complex128) + 0 + 1j)  # (i) * derivative along rows of DF, has "0" in the real components, and "d(DF)/dx" in the complex

    i_ky_0kx = np.add.outer(i_ky, kx0)

    fft_im = scipy.fft.fft2(im_mirror_v)
    kernfft_dy = i_ky_0kx * fft_im
    dy_im = np.real(scipy.fft.ifft2(kernfft_dy))

    dy_im_crop = dy_im[:image.shape[0], :image.shape[1]]
    return dy_im_crop
def lowpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, beyond some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    lowpass_2d = np.exp(-r * (kr ** 2))

    # plt.imshow(lowpass_2d)
    # plt.title('Low-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return lowpass_2d
def highpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a high-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    # plt.imshow(highpass_2d)
    # plt.title('High-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return highpass_2d
def midpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    C = np.zeros(columns, dtype=np.complex128)
    C = C + 0 + 1j
    ikx = kx * C  # (i) * spatial frequencies in x direction (along columns) - as complex numbers ( has "0" in the real components, and "kx" in the complex)
    denom = np.add.outer((-1 * ky), ikx)  # array with ikx - ky (DENOMINATOR)

    midpass_2d = np.divide(complex(1., 0.) * highpass_2d, denom, out=np.zeros_like(complex(1., 0.) * highpass_2d),
                           where=denom != 0)  # Setting output equal to zero where denominator equals zero

    # plt.imshow(np.real(midpass_2d))
    # plt.title('Mid-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return midpass_2d
# ---------------------------------------------------------------------------------
# Here are all of the solutions for the different inverse problems
# 1) Transport-of-intensity equation single-exposure speckle-based X-ray imaging phase-retrieval algorithm:
def TIE_Speckle(Is, Ir, pixel_size,gamma, prop, wavelength):
    # ---------------------------------------------------------------
    # Implementing the approach in:
    # K. M. Pavlov, H. Li, D. M. Paganin, et al., “Single-shot x-ray speckle-based imaging of a single-material object,”
    # Phys. Rev. Appl. 13, 054023 (2020).
    # ---------------------------------------------------------------
    # Definitions:
    # Is: One sample-plus-speckle image [ndarray]
    # Ir: One speckle-only image [ndarray]
    # pixel_size: Pixel size of the detector [microns]
    # gamma: Ratio of delta to beta for the object
    # prop: Distance between the sample and detector [microns]
    # wavelength: Wavelength of the monochromatic X-ray beam used during imaging [microns]
    # ---------------------------------------------------------------

    IsIr_mirror = np.concatenate((Is/Ir, np.fliplr(Is/Ir)), axis=1) # Doing mirroring (horizontally and vertically) to enforce periodicity for DFT implementation
    IsIr_mirror = np.concatenate((IsIr_mirror, np.flipud(IsIr_mirror)), axis=0)

    ft_IsIr = np.fft.fft2(IsIr_mirror) # Taking the 2D Fourier Transform
    ky, kx = kspace_kykx(ft_IsIr.shape, pixel_size) # Finding the Fourier-space spatial frequencies
    ky2kx2 = np.add.outer(ky**2,kx**2) # Making the k_x^2 + k_y^2 term with correct dimensions
    ins_ifft = ft_IsIr/(1 +((prop*gamma*wavelength)/(4*math.pi))*ky2kx2) # This is what needs the inverse Fourier transform applied to recover the attenuation term
    Iob = np.real(np.fft.ifft2(ins_ifft)) # Taking the inverse Fourier transform and only look at the real component to give the transmission term
    Iob_crop = Iob[:Is.shape[0], :Is.shape[1]] # Cropping off the mirror done

    Phase_crop = gamma/2 * np.log(Iob_crop) # This is the object's recovered phase, using the projection approximation

    return Iob_crop, Phase_crop

# 2) Single-exposure evolving speckle-based X-ray imaging Fokker--Planck perspective
def Single_Evolving(Is, Ir, pixel_size, gamma, prop, wavelength, savedir):
    # ---------------------------------------------------------------
    # Definitions:
    # Is: One sample-plus-speckle image [ndarray]
    # Ir: One speckle-only image [ndarray]
    # pixel_size: Pixel size of the detector [microns]
    # gamma: Ratio of delta to beta for the object
    # prop: Distance between the sample and detector [microns]
    # wavelength: Wavelength of the monochromatic X-ray beam used during imaging [microns]
    # savedir: Directory to save the output images
    # ---------------------------------------------------------------
    # Step 1: Approximate the sample's transmission term and phase shift using TIE Speckle method
    tranmission, phase = TIE_Speckle(Is, Ir, pixel_size, gamma, prop, wavelength)

    # Modify the speckle-only image by multiplying it with the transmission
    Ir_Tran = Ir * tranmission

    # Calculate the flux difference between modified speckle image and sample-plus-speckle image
    Flux = Ir_Tran - Is

    # Compute the logarithmic transmission to determine the optical flow field
    ln_trans = np.log(tranmission)

    # Step 2: Calculate the optical flow using the derivative of the transmission
    # The flow accounts for phase changes due to sample interaction with the X-rays
    Flow = (prop * gamma * wavelength) / (4 * math.pi) * (
            xderivative(Ir_Tran * xderivative(ln_trans, pixel_size), pixel_size) +
            yderivative(Ir_Tran * yderivative(ln_trans, pixel_size), pixel_size))

    # Subtract the flux from the calculated flow to obtain the final Flow minus Flux term
    FlowMINUSFlux = Flow - Flux

    # Step 3: Invert the Laplacian of the Flow minus Flux term to obtain a smoother solution
    invlapFF, reg = invLaplacian(FlowMINUSFlux, pixel_size)

    # Step 4: Compute the diffusion coefficient D using the inverted Laplacian
    D = invlapFF / (prop ** 2 * Ir_Tran)

    # Separate D into positive and negative values for further analysis
    positive_D = np.clip(D, 0, np.inf)  # Clipping at 0 to get positive values only
    negative_D = np.clip(D, -np.inf, 0)  # Clipping at 0 to get negative values only

    # Step 5: Save the calculated transmission and diffusion images
    os.chdir(savedir)  # Change the current directory to the save directory
    Image.fromarray(tranmission).save('TIE_Transmission.tif')  # Save transmission image
    Image.fromarray(D).save('XDF_SingleEvolve_{}.tif'.format(str(reg)))  # Save diffusion coefficient image
    Image.fromarray(positive_D).save('Pos_SingleEvolve_{}.tif'.format(str(reg)))  # Save positive diffusion image
    Image.fromarray(-1 * negative_D).save(
        'Neg_SingleEvolve_{}.tif'.format(str(reg)))  # Save negative diffusion image (inverted)

    print('Single-exposure evolving SBXI Fokker-Planck inverse problem has been solved!')
    # Return the diffusion coefficient and its positive/negative components along with the transmission term
    return D, positive_D, negative_D, tranmission

# 3) Dingle-exposure devolving speckle-based X-ray imaging Fokker--Planck perspective
def Single_Devolving(Is, Ir, pixel_size, gamma, prop, wavelength, savedir):
    # ---------------------------------------------------------------
    # Implementing the approach in:
    # M. A. Beltran, D. M. Paganin, M. K. Croughan, and K. S. Morgan, “Fast implicit diffusive dark-field retrieval for
    # single-exposure, single-mask x-ray imaging,” Optica 10, 422–429 (2023).
    # ---------------------------------------------------------------
    # Definitions:
    # Is: One sample-plus-speckle image [ndarray]
    # Ir: One speckle-only image [ndarray]
    # pixel_size: Pixel size of the detector [microns]
    # gamma: Ratio of delta to beta for the object
    # prop: Distance between the sample and detector [microns]
    # wavelength: Wavelength of the monochromatic X-ray beam used during imaging [microns]
    # savedir: Directory to save the output images
    # ---------------------------------------------------------------

    # Step 1: Approximate the sample's transmission and phase using TIE Speckle method
    tranmission, phase = TIE_Speckle(Is, Ir, pixel_size, gamma, prop, wavelength)

    # Step 2: Calculate the flux, which represents the difference between the sample-plus-speckle
    # image and the modified speckle-only image (scaled by transmission)
    Flux = Is - Ir * tranmission

    # Compute the logarithmic transmission to derive the optical flow
    ln_trans = np.log(tranmission)

    # Step 3: Calculate the optical flow based on the derivatives of the transmission
    # The flow characterizes how the phase changes across the image
    Flow = (prop * gamma * wavelength) / (4 * math.pi) * (
            xderivative(Is * xderivative(ln_trans, pixel_size), pixel_size) +
            yderivative(Is * yderivative(ln_trans, pixel_size), pixel_size))

    # Add the flux and flow to generate the final term used in the Laplacian inversion
    FluxaddFlow = Flux + Flow

    # Step 4: Invert the Laplacian of the Flux plus Flow term to obtain a smoother solution
    invlapFF, reg = invLaplacian(FluxaddFlow, pixel_size)

    # Step 5: Compute the diffusion coefficient D from the inverted Laplacian
    D = invlapFF / (prop ** 2 * Is)

    # Separate D into positive and negative values for further analysis
    positive_D = np.clip(D, 0, np.inf)  # Positive values of D
    negative_D = np.clip(D, -np.inf, 0)  # Negative values of D

    # Step 6: Save the transmission and diffusion coefficient images
    os.chdir(savedir)  # Change directory to the save location
    Image.fromarray(tranmission).save('TIE_Transmission.tif')  # Save transmission image
    Image.fromarray(D).save('XDF_SingleDevolve_{}.tif'.format(str(reg)))  # Save diffusion image
    Image.fromarray(positive_D).save('Pos_SingleDevolve_{}.tif'.format(str(reg)))  # Save positive diffusion image
    Image.fromarray(-1 * negative_D).save(
        'Neg_SingleDevolve_{}.tif'.format(str(reg)))  # Save negative diffusion image (inverted)
    print('Single-exposure devolving SBXI Fokker-Planck inverse problem has been solved!')
    # Return the diffusion coefficient and its positive/negative components along with the transmission term
    return D, positive_D, negative_D, tranmission

# 4) Multiple-exposure evolving speckle-based X-ray imaging Fokker--Planck perspective
def Multiple_Evolving(num_masks, Is, Ir, gamma, wavelength, prop, pixel_size, savedir):
    # ----------------------------------------------------------------
    # Implementing the approach in:
    # S. J. Alloo, K. S. Morgan, D. M. Paganin, and K. M. Pavlov, “Multimodal intrinsic speckle-tracking (MIST) to extract
    # images of rapidly-varying diffuse x-ray dark-field,” Sci. Reports 13, 5424 (2023).
    # ----------------------------------------------------------------
    # num_masks: Number of sets of Speckle-based X-ray imaging data [float]
    # Is: List of sample-reference speckle fields (X-ray beam + mask + sample + detector) [array]
    # Ir: List of reference speckle fields (X-ray beam + mask + detector) [array]
    # gamma: Ratio of real and imaginary refractive index coefficients of the sample [float]
    # wavelength: Wavelength of X-ray beam [microns]
    # prop: Propagation distance between sample and detector [microns]
    # pixel_size: Pixel size of the detector [microns]
    # savedir: Directory where calculated signals are saved [string]
    # ----------------------------------------------------------------

    # Empty lists to store terms for solving the system of linear equations
    coeff_D = []
    coeff_dx = []
    coeff_dy = []
    lapacaian = []
    RHS = []

    # Create empty arrays for QR decomposition
    coefficient_A = np.empty([int((num_masks)), 4, int(rows), int(columns)])
    coefficient_b = np.empty([int((num_masks)), 1, int(rows), int(columns)])

    # Loop through each mask to calculate and store the required coefficients
    for i in range(num_masks):
        rhs = (1 / prop) * (
                    Ir[i, :, :] - Is[i, :, :])  # Right-hand side term (difference between reference and sample fields)
        lap = Ir[i, :, :]  # Laplacian term
        deff = (-1) * np.divide(ndimage.laplace(Ir[i, :, :]), pixel_size ** 2)  # Effective diffusion coefficient (D)
        dy, dx = np.gradient(Ir[i, :, :], pixel_size)  # Gradient terms for X and Y directions
        dy_r = -2 * dy
        dx_r = -2 * dx

        # Append calculated coefficients to lists
        coeff_D.append(deff)
        coeff_dx.append(dx_r)
        coeff_dy.append(dy_r)
        lapacaian.append(lap)
        RHS.append(rhs)

    # Establish the system of linear equations Ax = b where x = [Laplacian(1/wavenumber*Phi - D), D, dx, dy]
    for n in range(len(coeff_dx)):
        coefficient_A[n, :, :, :] = np.array([lapacaian[n], coeff_D[n], coeff_dx[n], coeff_dy[n]])
        coefficient_b[n, :, :, :] = RHS[n]

    # Apply Tikhonov regularization to stabilize the system
    identity = np.identity(4)
    alpha = np.std(coefficient_A) / 10000  # Optimal Tikhonov regularization parameter (can be tweaked)
    reg = np.multiply(alpha, identity)  # Regularization term applied to coefficient array
    reg_repeat = np.repeat(reg, rows * columns).reshape(4, 4, rows, columns)  # Repeat regularization for all pixels
    zero_repeat = np.zeros((4, 1, rows, columns))  # Regularization on right-hand side vector

    # Add regularization to the system
    coefficient_A_reg = np.vstack([coefficient_A, reg_repeat])
    coefficient_b_reg = np.vstack([coefficient_b, zero_repeat])

    # Perform QR decomposition and solve the system of equations
    reg_Qr, reg_Rr = np.linalg.qr(coefficient_A_reg.transpose([2, 3, 0, 1]))
    reg_x = np.linalg.solve(reg_Rr, np.matmul(np.matrix.transpose(reg_Qr.transpose([2, 3, 1, 0])),
                                              coefficient_b_reg.transpose([2, 3, 0, 1])))

    # Extract calculated terms from solution
    lap_phiDF = reg_x[:, :, 0, 0]  # Laplacian term
    DFqr = reg_x[:, :, 1, 0] / prop  # Dark-field (DF) term
    dxDF = reg_x[:, :, 2, 0] / prop  # Gradient of DF in X direction
    dyDF = reg_x[:, :, 3, 0] / prop  # Gradient of DF in Y direction

    # Set working directory to save results
    os.chdir(savedir)

    # Filtering the solutions to obtain the TRUE dark-field signal
    cutoff = 10  # Cutoff parameter for filtering (optimize for SNR/NIQE)
    # rangeOcut = range(0,200,5) # This for-loop can be used to generate 100 filtered dark-field signals for different cut-off parameter values, just place the next 21 lines of code into the for-loop.
    # for cutoff in rangeOcut:

    # Fourier transform of combined gradient terms and mid-pass filtering
    i_dyDF = dyDF * (np.zeros((DFqr.shape), dtype=np.complex128) + 0 + 1j)  # Complex term (i * d(DF)/dy)
    insideft = dxDF + i_dyDF  # Combined gradient term
    insideftm = np.concatenate((insideft, np.flipud(insideft)), axis=0)  # Enforce periodic boundary conditions
    ft_dx_idy = np.fft.fft2(insideftm)
    MP = midpass_2D(ft_dx_idy, cutoff, pixel_size)  # Mid-pass filter
    MP_deriv = MP * ft_dx_idy  # Mid-pass filtered solution

    # Fourier transform and low-pass filtering of DF term
    DFqrm = np.concatenate((DFqr, np.flipud(DFqr)), axis=0)
    ft_DFqr = np.fft.fft2(DFqrm)
    LP = lowpass_2D(ft_DFqr, cutoff, pixel_size)  # Low-pass filter
    LP_DFqr = LP * ft_DFqr  # Low-pass filtered solution

    # Combine filtered solutions to obtain true dark-field signal -- aggregated dark-field
    combined = LP_DFqr + MP_deriv
    DF_filtered = np.fft.ifft2(combined)  # TRUE dark-field
    DF_filtered = np.real(DF_filtered[0:int(rows), :])

    # Calculate phase-shifts and attenuation term
    ref = Ir[0, :, :]  # Reference field
    sam = Is[0, :, :]  # Sample field
    lapphi = (ref - sam + prop ** 2 * np.divide(ndimage.laplace(DF_filtered * ref), pixel_size ** 2)) * (
                (2 * math.pi) / (wavelength * prop * ref))
    phi, reg = invLaplacian(lapphi, pixel_size)  # Solve inverse Laplacian

    # Save phase and attenuation images
    Image.fromarray(phi).save('Phase_MultipleEvolve_{}.tif'.format(str(reg) + 'gamma' + str(gamma) + 'r' + str(cutoff)))
    Iob = np.exp(2 * phi / gamma)  # Object's attenuation term
    Image.fromarray(Iob).save(
        'Transmission_MultipleEvolve_{}.tif'.format(str(reg) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

    # Corrected the aggregated dark-field image for X-ray attenuation in the sample
    DF_atten = np.real(DF_filtered / Iob)
    Image.fromarray(DF_atten).save(
        'XDF_MultipleEvolve_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

    # Save positive and negative dark-field components
    positive_D = np.clip(DF_atten, 0, np.inf)
    negative_D = np.clip(DF_atten, -np.inf, 0)
    Image.fromarray(positive_D).save(
        'PosXDF_MultipleEvolve_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))
    Image.fromarray(-1 * negative_D).save(
        'NegXDF_MultipleEvolve_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

    print('Multiple-exposure evolving SBXI Fokker-Planck inverse problem has been solved!')

    # Return calculated dark-field, positive and negative dark-field, and attenuation
    return DF_atten, positive_D, negative_D, Iob

# 5) Multiple-exposure devolving speckle-based X-ray imaging Fokker--Planck perspective
def Multiple_Devolving(num_masks, Is, Ir, gamma, wavelength, prop, pixel_size, savedir):
    # ----------------------------------------------------------------
    # num_masks: number of Speckle-based X-ray imaging datasets [float]
    # Is: list of sample-reference speckle fields (X-ray beam + mask + sample + detector) [array]
    # Ir: list of reference speckle fields (X-ray beam + mask + detector) [array]
    # gamma: ratio of real to imaginary refractive index coefficients of the sample
    # wavelength: wavelength of X-ray beam [microns]
    # prop: propagation distance between the sample and detector [microns]
    # pixel_size: pixel size of the detector [microns]
    # savedir: directory for saving calculated signals [string]
    # ----------------------------------------------------------------

    coeff_D = []  # Stores the Laplacian of the sample-reference speckle field
    coeff_dx = []  # Stores x-gradient of the sample-reference speckle field
    coeff_dy = []  # Stores y-gradient of the sample-reference speckle field
    lapacaian = []  # Stores the Laplacian term of the system of equations
    RHS = []  # Stores the right-hand side (RHS) of the system of equations

    # Arrays to store terms for QR decomposition
    coefficient_A = np.empty([int(num_masks), 4, int(rows), int(columns)])
    coefficient_b = np.empty([int(num_masks), 1, int(rows), int(columns)])

    # Loop over each mask to calculate and store the coefficients
    for i in range(num_masks):
        rhs = (1 / prop) * (Is[i, :, :] - Ir[i, :, :])  # Compute RHS for the system of equations
        lap = Is[i, :, :]  # Compute the Laplacian term (placeholder)
        deff = np.divide(ndimage.laplace(Is[i, :, :]), pixel_size ** 2)  # Compute the Laplacian of the speckle field
        dy, dx = np.gradient(Is[i, :, :], pixel_size)  # Compute the x and y gradients of the speckle field
        dy_r = 2 * dy  # Adjust y-gradient term
        dx_r = 2 * dx  # Adjust x-gradient term

        # Append computed terms to the respective lists
        coeff_D.append(deff)
        coeff_dx.append(dx_r)
        coeff_dy.append(dy_r)
        lapacaian.append(lap)
        RHS.append(rhs)

    # Assemble the system of linear equations: Ax = b
    for n in range(len(coeff_dx)):
        coefficient_A[n, :, :, :] = np.array([lapacaian[n], coeff_D[n], coeff_dx[n], coeff_dy[n]])  # Coefficient matrix
        coefficient_b[n, :, :, :] = RHS[n]  # RHS vector

    identity = np.identity(4)  # 4x4 identity matrix for Tikhonov Regularization
    alpha = np.std(coefficient_A) / 10000  # Optimal Tikhonov regularization parameter (tweak if system is unstable)
    reg = np.multiply(alpha, identity)  # Regularization matrix
    reg_repeat = np.repeat(reg, rows * columns).reshape(4, 4, rows,
                                                        columns)  # Repeat regularization for all pixel positions
    zero_repeat = np.zeros((4, 1, rows, columns))  # Zero matrix for regularizing RHS vector

    # Apply Tikhonov regularization to the system
    coefficient_A_reg = np.vstack([coefficient_A, reg_repeat])
    coefficient_b_reg = np.vstack([coefficient_b, zero_repeat])

    # Perform QR decomposition
    reg_Qr, reg_Rr = np.linalg.qr(coefficient_A_reg.transpose([2, 3, 0, 1]))

    # Solve the system using QR decomposition (solve Rx = Q^T b instead of inversion)
    reg_x = np.linalg.solve(reg_Rr, np.matmul(np.matrix.transpose(reg_Qr.transpose([2, 3, 1, 0])),
                                              coefficient_b_reg.transpose([2, 3, 0, 1])))

    # Extract solution components
    lap_phiDF = reg_x[:, :, 0, 0]  # Laplacian term (Laplacian(1/wavenumber*Phi - D))
    DFqr = reg_x[:, :, 1, 0] / prop  # Dark-field (DF) term
    dxDF = reg_x[:, :, 2, 0] / prop  # Derivative of DF along x
    dyDF = reg_x[:, :, 3, 0] / prop  # Derivative of DF along y

    # Change working directory to save directory
    os.chdir(savedir)

    # Apply filtering to determine the true dark-field signal
    cutoff = 10  # Cutoff parameter for filtering (optimize this for best SNR or NIQE)

    # Compute Fourier transform of gradient terms
    i_dyDF = dyDF * (np.zeros((DFqr.shape), dtype=np.complex128) + 0 + 1j)
    insideft = dxDF + i_dyDF
    insideftm = np.concatenate((insideft, np.flipud(insideft)),
                               axis=0)  # Mirror to enforce periodic boundary conditions
    ft_dx_idy = np.fft.fft2(insideftm)

    # Apply mid-pass and low-pass filters
    MP = midpass_2D(ft_dx_idy, cutoff, pixel_size)
    MP_deriv = MP * ft_dx_idy  # Mid-pass filtered derivative solution
    DFqrm = np.concatenate((DFqr, np.flipud(DFqr)), axis=0)
    ft_DFqr = np.fft.fft2(DFqrm)
    LP = lowpass_2D(ft_DFqr, cutoff, pixel_size)
    LP_DFqr = LP * ft_DFqr  # Low-pass filtered DF solution

    # Combine filtered solutions to obtain true dark-field signal -- aggregated dark-field
    combined = LP_DFqr + MP_deriv
    DF_filtered = np.fft.ifft2(combined)
    DF_filtered = np.real(DF_filtered[0:int(rows), :])  # Invert Fourier transform and keep real part

    # Compute phase shifts and attenuation term
    ref = Ir[0, :, :]
    sam = Is[0, :, :]
    lapphi = (ref - sam + prop ** 2 * np.divide(ndimage.laplace(DF_filtered * ref), pixel_size ** 2)) * (
                (2 * math.pi) / (wavelength * prop * ref))

    # Invert Laplacian to retrieve phase
    phi, reg = invLaplacian(lapphi, pixel_size)

    # Save phase and transmission images
    phi_im = Image.fromarray(phi).save(
        'Phase_MultipleDevolve_{}.tif'.format(str(reg) + 'gamma' + str(gamma) + 'r' + str(cutoff)))
    Iob = np.exp(2 * phi / gamma)  # Object's attenuation term
    Iob_im = Image.fromarray(Iob).save('Transmission_MultipleDevolve_{}.tif'.format(str(reg) + 'gamma' + str(cutoff)))

    # Corrected the aggregated dark-field image for X-ray attenuation in the sample
    DF_atten = np.real(DF_filtered / Iob)
    DFattim = Image.fromarray(np.real(DF_atten)).save(
        'XDF_MultipleDevolve_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

    # Separate positive and negative values of dark-field signal and save
    positive_D = np.clip(DF_atten, 0, np.inf)
    negative_D = np.clip(DF_atten, -np.inf, 0)
    save_D = Image.fromarray(positive_D).save(
        'PosXDF_MultipleDevolve_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))
    save_D = Image.fromarray(-1 * negative_D).save(
        'NegXDF_MultipleDevolve_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

    # Final output
    print('Multiple-exposure devolving SBXI Fokker-Planck inverse problem has been solved!')
    return DF_atten, positive_D, negative_D, Iob
# ---------------------------------------------------------------------------------
# Four-rod Data: Collected at the MicroCT beamline at the Australian Synchrotron. First published in Alloo et al.
# 2024 "Separating edges from microstructure in X-ray dark-field imaging:
# Evolving and devolving perspectives via the X-ray Fokker--Planck equation"
data = r'C:\Users\sall0037\Documents\DiscoveryProject_FokkerPlanck\Manuscripts\OpticsExpress_EvolvingDevolving2024\GitHub\Python Scripts to Retrieve Images\FourRod_Data' # This is the directory where the data is
os.chdir(data)

num_masks = 13
gamma = 2335 # Value of perspex at 25 keV
wavelength = 4.9594*10**-5 # [microns]
prop = 0.7*10**6 # [microns]
pixel_size = 6.5 # [microns]
savedir = r'C:\Users\sall0037\Documents\DiscoveryProject_FokkerPlanck\Manuscripts\OpticsExpress_EvolvingDevolving2024\GitHub\Python Scripts to Retrieve Images\FouRod_RetrievedSignals'

ff = np.double(np.asarray(Image.open('FF_1m.tif')))[870:1300,140:2360] # Cropping required to get just illuminated field-of-view
rows, columns = ff.shape

Ir = np.empty([int(num_masks),int(rows),int(columns)])
Is = np.empty([int(num_masks),int(rows),int(columns)])
dc = 0
for k in range(1,int(num_masks+1)):
        i = str(k)
        # -------------------------------------------------------------------------
        # Reading in data: change string for start of filename as required
        ir = np.double(np.asarray(Image.open('Ref{}.tif'.format(str(i)))))[870:1300,140:2360]
        isa = np.double(np.asarray(Image.open('Sam{}.tif'.format(str(i)))))[870:1300,140:2360]

        ir = (ir-dc)/(ff-dc)
        isa = (isa-dc)/(ff-dc)

        Is[int(k-1)] = (isa)  # shape =  [num_masks, rows, columns]
        Ir[int(k-1)] = (ir)  # shape =  [num_masks, rows, columns]

print('Data reading completed')
# -------------------------------------------------------------------
# Running the functions
D_Sev, positive_D_Sev, negative_D_Sev, transmission  = Single_Evolving(Is[0], Ir[0], pixel_size,gamma, prop, wavelength,savedir)
D_Sdev, positive_D_Sdev, negative_D_Sdev, tranmission  = Single_Devolving(Is[0], Ir[0], pixel_size,gamma, prop, wavelength,savedir)
D_Mev ,positive_D_Mev, negative_D_Mev, transmission_Mev = Multiple_Evolving(num_masks, Is, Ir, gamma, wavelength, prop, pixel_size, savedir)
D_Mdev ,positive_D_Mdev, negative_D_Mdev, transmission_Mdev = Multiple_Devolving(num_masks, Is, Ir, gamma, wavelength, prop, pixel_size, savedir)
# ---------------------------------------------------------------------------------
