import numpy as np
import sep
from astropy.io import fits
from photutils.aperture import EllipticalAperture, aperture_photometry

def perform_aperture_photometry(hdul, source_params, bands=['g', 'r', 'i', 'z'], 
                               zp_dict={'g': 25.0, 'r': 25.0, 'i': 25.0, 'z': 25.0}):
    """
    Perform aperture photometry using elliptical apertures.
    
    Parameters:
    -----------
    hdul : astropy.io.fits.HDUList
        Input FITS file with multiple bands
    source_params : dict
        Source parameters from detect_sources
    bands : list
        List of band names
    zp_dict : dict
        Zero-point magnitudes for each band
        
    Returns:
    --------
    dict : Magnitudes for each band
    dict : Colors (magnitude differences)
    """
    
    x, y = source_params['x'], source_params['y']
    a, b, theta = source_params['a'], source_params['b'], source_params['theta']
    
    # Define elliptical aperture
    aperture = EllipticalAperture((x, y), a, b, theta)
    
    magnitudes = {}
    
    # Check data structure
    if hdul[0].data is None:
        raise ValueError("Primary HDU contains no data")
    
    data_shape = hdul[0].data.shape
    print(f"DEBUG PHOTOMETRY: Data shape: {data_shape}")
    
    if len(data_shape) == 3:
        # 3D array: (bands, height, width)
        n_bands = data_shape[0]
        
        for i, band in enumerate(bands):
            if i >= n_bands:
                print(f"WARNING: Band {band} (index {i}) not available, skipping")
                continue
                
            # Extract image data for this band
            image_data = hdul[0].data[i].astype(np.float64)
            print(f"DEBUG PHOTOMETRY: Processing band {band} (index {i}), shape: {image_data.shape}")
            
            # Subtract background for the current band
            bkg_band = sep.Background(image_data)
            image_data_sub = image_data - bkg_band
            
            print(f"DEBUG PHOTOMETRY: Band {band} background: {bkg_band.globalback:.3e}, RMS: {bkg_band.globalrms:.3e}")
            
            # Perform aperture photometry
            phot_table = aperture_photometry(image_data_sub, aperture)
            total_flux = phot_table['aperture_sum'][0]
            
            print(f"DEBUG PHOTOMETRY: Band {band} total flux: {total_flux:.3e}")
            
            # Convert flux to magnitude
            if total_flux > 0:
                magnitude = -2.5 * np.log10(total_flux) + zp_dict[band]
                magnitudes[band] = magnitude
                print(f"DEBUG PHOTOMETRY: Band {band} magnitude: {magnitude:.3f}")
            else:
                magnitudes[band] = np.nan
                print(f"WARNING: Band {band} has non-positive flux: {total_flux}")
                
    elif len(data_shape) == 2:
        # 2D array: single band - assume it's the first band in the list
        band = bands[0]
        image_data = hdul[0].data.astype(np.float64)
        
        bkg_band = sep.Background(image_data)
        image_data_sub = image_data - bkg_band
        
        phot_table = aperture_photometry(image_data_sub, aperture)
        total_flux = phot_table['aperture_sum'][0]
        
        if total_flux > 0:
            magnitude = -2.5 * np.log10(total_flux) + zp_dict[band]
            magnitudes[band] = magnitude
        else:
            magnitudes[band] = np.nan
    
    else:
        raise ValueError(f"Unsupported data shape: {data_shape}")
    
    # Calculate colors
    colors = {}
    band_list = list(magnitudes.keys())
    for i in range(len(band_list) - 1):
        if not np.isnan(magnitudes[band_list[i]]) and not np.isnan(magnitudes[band_list[i+1]]):
            color_name = f'{band_list[i]}-{band_list[i+1]}'
            colors[color_name] = magnitudes[band_list[i]] - magnitudes[band_list[i+1]]
            print(f"DEBUG PHOTOMETRY: Color {color_name}: {colors[color_name]:.3f}")
    
    return magnitudes, colors