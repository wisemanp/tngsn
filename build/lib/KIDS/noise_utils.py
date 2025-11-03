import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import os
import logging

def add_noise_and_save(hdul, subhaloId, output_dir, psf_fwhm=0.7, gain=3e13, sigma_bkg=2e-12):
    """
    Add realistic noise to KiDS images and save the noisy version.
    
    Parameters:
    -----------
    hdul : astropy.io.fits.HDUList
        Input FITS file
    subhaloId : int
        Subhalo ID for naming
    output_dir : str
        Directory to save noisy images
    psf_fwhm : float
        PSF FWHM in arcseconds (default 0.7)
    gain : float
        Effective gain in electrons per ADU (default 3e13)
    sigma_bkg : float
        Background noise std deviation in ADU/s (default 2e-12)
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f"NOISE - SubhaloID {subhaloId}: Starting noise addition")
    
    # Check the primary HDU
    if hdul[0].data is None:
        raise ValueError("Primary HDU contains no data")
    
    data_shape = hdul[0].data.shape
    header = hdul[0].header.copy()
    
    logger.info(f"NOISE - SubhaloID {subhaloId}: Original data shape: {data_shape}")
    logger.info(f"NOISE - SubhaloID {subhaloId}: Data type: {hdul[0].data.dtype}")
    logger.info(f"NOISE - SubhaloID {subhaloId}: Data size: {hdul[0].data.nbytes / (1024*1024):.2f} MB")
    
    # Get pixel scale for PSF convolution
    pixelscale = header.get('PIXSCALE', 0.2)  # Default pixel scale
    sigma_psf = psf_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / pixelscale
    logger.info(f"NOISE - SubhaloID {subhaloId}: Pixel scale: {pixelscale} arcsec/pixel")
    logger.info(f"NOISE - SubhaloID {subhaloId}: PSF FWHM: {psf_fwhm} arcsec, sigma: {sigma_psf:.2f} pixels")
    
    if len(data_shape) == 3:
        # 3D array: (bands, height, width)
        n_bands, height, width = data_shape
        logger.info(f"NOISE - SubhaloID {subhaloId}: Processing 3D data - {n_bands} bands, {height}x{width} pixels")
        
        # Process each band separately
        noisy_data = np.zeros_like(hdul[0].data, dtype=np.float64)
        logger.info(f"NOISE - SubhaloID {subhaloId}: Created noisy_data array with shape: {noisy_data.shape}")
        
        for i in range(n_bands):
            band_name = ['g', 'r', 'i', 'z'][i] if i < 4 else f'band{i}'
            logger.info(f"NOISE - SubhaloID {subhaloId}: Processing band {i}/{n_bands} ({band_name})")
            
            # Extract single band data
            band_data = hdul[0].data[i].copy().astype(np.float64)
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} extracted shape: {band_data.shape}")
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} stats - min: {np.min(band_data):.3e}, max: {np.max(band_data):.3e}, mean: {np.mean(band_data):.3e}")
            
            # Check for valid data
            finite_count = np.sum(np.isfinite(band_data))
            nonzero_count = np.count_nonzero(band_data)
            total_pixels = band_data.size
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} - finite: {finite_count}/{total_pixels}, non-zero: {nonzero_count}/{total_pixels}")
            
            # Convolve with 2D Gaussian PSF
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} - applying PSF convolution with sigma={sigma_psf:.2f}")
            convolved_image = gaussian_filter(band_data, sigma=sigma_psf)
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} after PSF - shape: {convolved_image.shape}")
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} after PSF - min: {np.min(convolved_image):.3e}, max: {np.max(convolved_image):.3e}")
            
            # Add shot noise (Poisson)
            # Ensure positive values for Poisson noise
            convolved_positive = np.maximum(convolved_image, 1e-20)  # Small positive floor
            max_scaled_flux = np.max(convolved_positive * gain)
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} - max scaled flux for Poisson: {max_scaled_flux:.3e}")
            
            # For very small fluxes, Poisson noise can be approximated as Gaussian
            # to avoid computational issues with very large arrays
            if max_scaled_flux < 1000:
                # Use Gaussian approximation for low flux
                shot_noise = np.random.normal(0, np.sqrt(convolved_positive * gain), convolved_positive.shape) / gain
                logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} - using Gaussian approximation for shot noise")
            else:
                # Use proper Poisson noise
                shot_noise = np.random.poisson(convolved_positive * gain) / gain - convolved_positive
                logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} - using Poisson shot noise")
            
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} shot noise - shape: {shot_noise.shape}, std: {np.std(shot_noise):.3e}")
            
            # Add background noise (Gaussian)
            background_noise = np.random.normal(0, sigma_bkg, convolved_image.shape)
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} background noise - shape: {background_noise.shape}, std: {np.std(background_noise):.3e}")
            
            # Combine all components
            noisy_band = convolved_image + shot_noise + background_noise
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} combined noise - shape: {noisy_band.shape}")
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} final stats - min: {np.min(noisy_band):.3e}, max: {np.max(noisy_band):.3e}, mean: {np.mean(noisy_band):.3e}")
            
            # Store in output array
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} - storing in noisy_data[{i}]")
            noisy_data[i] = noisy_band
            logger.info(f"NOISE - SubhaloID {subhaloId}: Band {band_name} - stored successfully, noisy_data[{i}] shape: {noisy_data[i].shape}")
        
        logger.info(f"NOISE - SubhaloID {subhaloId}: Completed processing all {n_bands} bands")
        logger.info(f"NOISE - SubhaloID {subhaloId}: Final noisy_data shape: {noisy_data.shape}")
        logger.info(f"NOISE - SubhaloID {subhaloId}: Final noisy_data stats - min: {np.min(noisy_data):.3e}, max: {np.max(noisy_data):.3e}")
        
        # Create new HDU with noisy data
        noisy_hdu = fits.PrimaryHDU(data=noisy_data, header=header)
        noisy_hdul = fits.HDUList([noisy_hdu])
        
    elif len(data_shape) == 2:
        # 2D array: single band
        height, width = data_shape
        logger.info(f"NOISE - SubhaloID {subhaloId}: Processing 2D data - {height}x{width} pixels")
        
        image_data = hdul[0].data.copy().astype(np.float64)
        logger.info(f"NOISE - SubhaloID {subhaloId}: 2D data stats - min: {np.min(image_data):.3e}, max: {np.max(image_data):.3e}")
        
        # Convolve with 2D Gaussian PSF
        convolved_image = gaussian_filter(image_data, sigma=sigma_psf)
        
        # Add shot noise (Poisson)
        convolved_positive = np.maximum(convolved_image, 1e-20)
        
        if np.max(convolved_positive * gain) < 1000:
            shot_noise = np.random.normal(0, np.sqrt(convolved_positive * gain), convolved_positive.shape) / gain
            logger.info(f"NOISE - SubhaloID {subhaloId}: Using Gaussian approximation for shot noise")
        else:
            shot_noise = np.random.poisson(convolved_positive * gain) / gain - convolved_positive
            logger.info(f"NOISE - SubhaloID {subhaloId}: Using Poisson shot noise")
        
        # Add background noise (Gaussian)
        background_noise = np.random.normal(0, sigma_bkg, convolved_image.shape)
        
        # Combine all components
        noisy_data = convolved_image + shot_noise + background_noise
        logger.info(f"NOISE - SubhaloID {subhaloId}: 2D final stats - min: {np.min(noisy_data):.3e}, max: {np.max(noisy_data):.3e}")
        
        # Create new HDU with noisy data
        noisy_hdu = fits.PrimaryHDU(data=noisy_data, header=header)
        noisy_hdul = fits.HDUList([noisy_hdu])
        
    else:
        raise ValueError(f"Unsupported data shape: {data_shape}")
    
    # Save the noisy image
    output_path = os.path.join(output_dir, f'noisy_broadband_{subhaloId}.fits')
    logger.info(f"NOISE - SubhaloID {subhaloId}: Saving to {output_path}")
    
    try:
        noisy_hdul.writeto(output_path, overwrite=True)
        logger.info(f"NOISE - SubhaloID {subhaloId}: Successfully saved noisy image")
        
        # Verify the saved file
        with fits.open(output_path) as verify_hdul:
            saved_shape = verify_hdul[0].data.shape
            logger.info(f"NOISE - SubhaloID {subhaloId}: Verified saved file shape: {saved_shape}")
            
    except Exception as e:
        logger.error(f"NOISE - SubhaloID {subhaloId}: Error saving file: {e}")
        raise
    finally:
        noisy_hdul.close()
    
    logger.info(f"NOISE - SubhaloID {subhaloId}: Noise addition completed successfully")
    
    return output_path