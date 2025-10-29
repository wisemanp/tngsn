import numpy as np
import sep
from astropy.io import fits

def detect_sources(hdul, bands=['g', 'r', 'i', 'z'], threshold=1.5, coadd_bands=['r', 'i', 'z']):
    """
    Detect sources using SEP on a coadd image.
    
    Parameters:
    -----------
    hdul : astropy.io.fits.HDUList
        Input FITS file with multiple bands
    bands : list
        List of band names corresponding to band indices
    threshold : float
        Detection threshold in sigmas above background
    coadd_bands : list
        Bands to use for coadd detection image
        
    Returns:
    --------
    dict : Source parameters (x, y, a, b, theta) for the brightest source
    """
    
    print(f"DEBUG: FITS file has {len(hdul)} HDUs")
    print(f"DEBUG: Requested bands: {bands}")
    print(f"DEBUG: Coadd bands: {coadd_bands}")
    
    # Check the primary HDU
    if hdul[0].data is None:
        raise ValueError("Primary HDU contains no data")
    
    data_shape = hdul[0].data.shape
    print(f"DEBUG: Primary HDU data shape: {data_shape}")
    
    # Handle different data structures
    if len(data_shape) == 3:
        # 3D array: (bands, height, width)
        n_bands, height, width = data_shape
        print(f"DEBUG: 3D data detected - {n_bands} bands, {height}x{width} pixels")
        
        if n_bands != len(bands):
            print(f"WARNING: Expected {len(bands)} bands but found {n_bands}")
        
        # Create coadd from specified bands
        coadd_data = None
        used_bands = []
        
        for i, band in enumerate(bands):
            if band in coadd_bands and i < n_bands:
                band_data = hdul[0].data[i].astype(np.float64)
                print(f"DEBUG: Using band {i} ({band}) for coadd, shape: {band_data.shape}")
                print(f"DEBUG: Band {band} stats - min: {np.min(band_data):.3e}, max: {np.max(band_data):.3e}")
                
                if coadd_data is None:
                    coadd_data = band_data.copy()
                    print(f"DEBUG: Initialized coadd with band {band}")
                else:
                    coadd_data += band_data
                    print(f"DEBUG: Added band {band} to coadd")
                used_bands.append(band)
        
        print(f"DEBUG: Used bands for coadd: {used_bands}")
        
    elif len(data_shape) == 2:
        # 2D array: single band
        print(f"DEBUG: 2D data detected - single band, {data_shape[0]}x{data_shape[1]} pixels")
        coadd_data = hdul[0].data.astype(np.float64)
        used_bands = ["single_band"]
        
    else:
        raise ValueError(f"Unsupported data shape: {data_shape}")
    
    if coadd_data is None:
        raise ValueError("No valid bands found for coadd")
    
    print(f"DEBUG: Final coadd shape: {coadd_data.shape}")
    print(f"DEBUG: Coadd stats - min: {np.min(coadd_data):.3e}, max: {np.max(coadd_data):.3e}, mean: {np.mean(coadd_data):.3e}")
    
    # Ensure data is in the right format for SEP
    coadd_data = coadd_data.astype(np.float64)
    
    # Check for non-finite values
    finite_mask = np.isfinite(coadd_data)
    if not np.all(finite_mask):
        print(f"WARNING: Found {np.sum(~finite_mask)} non-finite values in coadd data")
        coadd_data = np.where(finite_mask, coadd_data, 0.0)
    
    try:
        # Measure and subtract background
        bkg = sep.Background(coadd_data)
        bkg_subtracted = coadd_data - bkg
        
        print(f"DEBUG: Background level: {bkg.globalback:.3e}, RMS: {bkg.globalrms:.3e}")
        print(f"DEBUG: Background subtracted data - min: {np.min(bkg_subtracted):.3e}, max: {np.max(bkg_subtracted):.3e}")
        
        # Detect objects
        objects = sep.extract(bkg_subtracted, threshold, err=bkg.globalrms)
        
        print(f"DEBUG: Detected {len(objects)} objects")
        
        if len(objects) == 0:
            # Try with lower threshold
            lower_threshold = threshold * 0.5
            print(f"DEBUG: No objects found, trying lower threshold: {lower_threshold}")
            objects = sep.extract(bkg_subtracted, lower_threshold, err=bkg.globalrms)
            print(f"DEBUG: Detected {len(objects)} objects with lower threshold")
            
            if len(objects) == 0:
                raise ValueError("No objects detected even with lower threshold")
        
        # Select the brightest object (highest flux)
        brightest_idx = np.argmax(objects['flux'])
        obj = objects[brightest_idx]
        
        print(f"DEBUG: Selected object {brightest_idx} with flux {obj['flux']:.3e}")
        print(f"DEBUG: Object position: ({obj['x']:.2f}, {obj['y']:.2f})")
        print(f"DEBUG: Object ellipse: a={obj['a']:.2f}, b={obj['b']:.2f}, theta={obj['theta']:.2f}")
        
        source_params = {
            'x': obj['x'],
            'y': obj['y'], 
            'a': obj['a'],
            'b': obj['b'],
            'theta': obj['theta'],
            'flux': obj['flux'],
            'background': bkg
        }
        
        return source_params
        
    except Exception as e:
        print(f"ERROR in source detection: {e}")
        print(f"DEBUG: Coadd data shape: {coadd_data.shape}")
        print(f"DEBUG: Coadd data type: {coadd_data.dtype}")
        print(f"DEBUG: Coadd finite values: {np.sum(np.isfinite(coadd_data))}/{coadd_data.size}")
        raise