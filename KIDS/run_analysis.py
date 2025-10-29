import pandas as pd
import numpy as np
import os
from astropy.io import fits
from tqdm import tqdm
import json
import argparse
import logging
from datetime import datetime

from noise_utils import add_noise_and_save
from source_extraction import detect_sources  
from photometry import perform_aperture_photometry

def setup_logging(params, stage_name=None):
    """Setup logging configuration."""
    
    # Create logs directory
    logs_dir = os.path.join(params['results_dir'], 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if stage_name:
        log_filename = f'{stage_name}_{timestamp}.log'
    else:
        log_filename = f'analysis_{timestamp}.log'
    
    log_path = os.path.join(logs_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger

def setup_paths_and_params(sim='TNG50-1', base_path=None):
    """Setup paths and parameters for the analysis."""
    
    # Determine base path
    if base_path is None:
        # Try environment variable first
        base_path = os.environ.get('TNGSN_PATH')
        if base_path is None:
            # Try to infer from current script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.dirname(script_dir)  # Go up one level from KIDS/
            print(f"Using inferred base path: {base_path}")
        else:
            print(f"Using TNGSN_PATH environment variable: {base_path}")
    else:
        print(f"Using provided base path: {base_path}")
    
    # Setup paths
    kidsdir = os.path.join(base_path, 'data', sim, 'KIDS', 'snapnum_096', 'zx', 'data')
    noisy_dir = os.path.join(base_path, 'data', sim, 'KIDS', 'snapnum_096', 'zx', 'noisy')
    results_dir = os.path.join(base_path, 'data', sim, 'KIDS', 'results')
    
    # Create directories
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load galaxy metadata
    galmeta_path = os.path.join(base_path, 'data', sim, 'galmeta.csv')
    if not os.path.exists(galmeta_path):
        raise FileNotFoundError(f"Galaxy metadata file not found: {galmeta_path}")
    galmeta = pd.read_csv(galmeta_path, index_col=0)
    
    # Parameters
    params = {
        'bands': ['g', 'r', 'i', 'z'],
        'zp_dict': {'g': 25.0, 'r': 25.0, 'i': 25.0, 'z': 25.0},
        'threshold': 1.5,
        'base_path': base_path,
        'kidsdir': kidsdir,
        'noisy_dir': noisy_dir,
        'results_dir': results_dir,
        'sim': sim
    }
    
    return galmeta, params

def log_fits_info(hdul, subhaloId, logger, stage=""):
    """Log information about FITS file."""
    logger.info(f"{stage} - SubhaloID {subhaloId}: Processing FITS file with {len(hdul)} HDUs")
    
    for i, hdu in enumerate(hdul):
        if hdu.data is not None:
            shape = hdu.data.shape
            dtype = hdu.data.dtype
            size_mb = hdu.data.nbytes / (1024 * 1024)
            band = ['g', 'r', 'i', 'z'][i] if i < 4 else f'HDU{i}'
            
            # Calculate basic statistics
            data_min = np.nanmin(hdu.data)
            data_max = np.nanmax(hdu.data)
            data_mean = np.nanmean(hdu.data)
            data_std = np.nanstd(hdu.data)
            
            logger.info(f"{stage} - SubhaloID {subhaloId} - HDU{i} (Band {band}): "
                       f"Shape={shape}, Size={size_mb:.2f}MB, Type={dtype}")
            logger.info(f"{stage} - SubhaloID {subhaloId} - HDU{i} (Band {band}) Stats: "
                       f"Min={data_min:.3e}, Max={data_max:.3e}, Mean={data_mean:.3e}, Std={data_std:.3e}")
            
            # Check if data contains any non-zero values
            non_zero_count = np.count_nonzero(hdu.data)
            total_pixels = hdu.data.size
            logger.info(f"{stage} - SubhaloID {subhaloId} - HDU{i} (Band {band}): "
                       f"Non-zero pixels: {non_zero_count}/{total_pixels} ({100*non_zero_count/total_pixels:.1f}%)")
        else:
            logger.info(f"{stage} - SubhaloID {subhaloId} - HDU{i}: No data")

def stage1_add_noise(galmeta, params, resume=True, test_mode=False, test_count=5):
    """Stage 1: Add noise to all images."""
    
    logger = setup_logging(params, 'stage1_noise')
    
    logger.info("=" * 50)
    logger.info("STAGE 1: Adding noise to images")
    if test_mode:
        logger.info(f"TEST MODE: Processing only {test_count} galaxies")
    logger.info("=" * 50)
    logger.info(f"Processing {len(galmeta)} galaxies from simulation {params['sim']}")
    
    # DEBUG: Show what's in the metadata
    logger.info(f"First 10 galaxy IDs in metadata: {list(galmeta.index[:10])}")
    logger.info(f"Last 10 galaxy IDs in metadata: {list(galmeta.index[-10:])}")
    logger.info(f"Metadata index type: {type(galmeta.index[0])}")
    
    # Track completed galaxies
    completed_file = os.path.join(params['results_dir'], 'stage1_completed.json')
    completed = set()
    
    if resume and os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            completed_ids = json.load(f)
            completed = set(completed_ids)
        logger.info(f"Resuming: {len(completed)} galaxies already completed from previous runs")
        logger.info(f"First 10 completed IDs: {sorted(list(completed))[:10]}")
        logger.info(f"Sample completed ID type: {type(list(completed)[0]) if completed else 'None'}")
    
    # Check overlap
    galmeta_ids = set(int(x) for x in galmeta.index)
    overlap = completed.intersection(galmeta_ids)
    logger.info(f"Overlap between completed and current metadata: {len(overlap)} galaxies")
    
    failed = []
    skipped_count = 0
    new_processed = 0
    
    # In test mode, limit the iterations
    galmeta_subset = galmeta.head(test_count) if test_mode else galmeta
    desc = f"Adding noise (TEST MODE - {test_count} galaxies)" if test_mode else "Adding noise"
    
    for idx, subhaloId in enumerate(tqdm(galmeta_subset.index, desc=desc)):
        subhaloId = int(subhaloId)
        
        # DEBUG: Log every iteration for first few
        if idx < 10:
            logger.info(f"DEBUG: Iteration {idx}, SubhaloID {subhaloId}, in completed: {subhaloId in completed}")
        
        if subhaloId in completed:
            skipped_count += 1
            if skipped_count <= 5:  # Log first few skips
                logger.info(f"SubhaloID {subhaloId}: Already completed, skipping (skip #{skipped_count})")
            elif skipped_count == 6:
                logger.info(f"... (continuing to skip already completed galaxies, total skipped so far: {skipped_count})")
            elif skipped_count % 100 == 0:  # Log every 100 skips
                logger.info(f"... (skipped {skipped_count} galaxies so far)")
            continue
            
        try:
            new_processed += 1
            logger.info(f"SubhaloID {subhaloId}: Starting noise addition (new galaxy {new_processed})")
            
            # Load original image
            fits_path = os.path.join(params['kidsdir'], f'broadband_{subhaloId}.fits')
            if not os.path.exists(fits_path):
                logger.error(f"SubhaloID {subhaloId}: File not found: {fits_path}")
                failed.append(subhaloId)
                continue
                
            logger.info(f"SubhaloID {subhaloId}: Loading FITS file: {fits_path}")
            hdul = fits.open(fits_path)
            
            # Log FITS file information (reduce verbosity)
            logger.info(f"SubhaloID {subhaloId}: FITS file loaded with {len(hdul)} HDUs, shape: {hdul[0].data.shape if hdul[0].data is not None else 'No data'}")
            
            # Add noise and save
            logger.info(f"SubhaloID {subhaloId}: Adding noise (PSF, shot noise, background)")
            noisy_path = add_noise_and_save(hdul, subhaloId, params['noisy_dir'])
            logger.info(f"SubhaloID {subhaloId}: Noisy image saved to: {noisy_path}")
            
            hdul.close()
            
            # Mark as completed
            completed.add(subhaloId)
            logger.info(f"SubhaloID {subhaloId}: Successfully completed noise addition")
            
            # Save progress periodically
            if new_processed % 10 == 0:
                with open(completed_file, 'w') as f:
                    json.dump(list(completed), f)
                logger.info(f"Progress saved: {len(completed)} galaxies completed total")
            
            # In test mode, exit after processing the requested number
            if test_mode and new_processed >= test_count:
                logger.info(f"TEST MODE: Reached target of {test_count} newly processed galaxies, stopping")
                break
                
        except Exception as e:
            logger.error(f"SubhaloID {subhaloId}: Error during noise addition: {e}")
            failed.append(subhaloId)
            continue
    
    # Final summary
    logger.info(f"Stage 1 Summary:")
    logger.info(f"  Total galaxies in metadata: {len(galmeta)}")
    if test_mode:
        logger.info(f"  TEST MODE: Limited to {len(galmeta_subset)} galaxies")
    logger.info(f"  Already completed (skipped): {skipped_count}")
    logger.info(f"  Newly processed: {new_processed}")
    logger.info(f"  Newly successful: {new_processed - len(failed)}")
    logger.info(f"  Newly failed: {len(failed)}")
    logger.info(f"  Total completed now: {len(completed)}")
    
    # Save final completion status
    with open(completed_file, 'w') as f:
        json.dump(list(completed), f)
    
    # Save failed list
    if failed:
        failed_file = os.path.join(params['results_dir'], 'stage1_failed.json')
        with open(failed_file, 'w') as f:
            json.dump(failed, f)
        logger.warning(f"Stage 1: {len(failed)} galaxies failed: {failed}")
    
    logger.info(f"Stage 1 completed: {len(completed)} total successful, {len(failed)} failed")
    return completed, failed

def stage2_source_extraction(galmeta, params, resume=True, test_mode=False, test_count=5):
    """Stage 2: Extract sources from noisy images."""
    
    logger = setup_logging(params, 'stage2_extraction')
    
    logger.info("=" * 50)
    logger.info("STAGE 2: Source extraction")
    if test_mode:
        logger.info(f"TEST MODE: Processing only {test_count} galaxies")
    logger.info("=" * 50)
    
    # Check stage 1 completion
    stage1_file = os.path.join(params['results_dir'], 'stage1_completed.json')
    if not os.path.exists(stage1_file):
        logger.error("Stage 1 not completed. Run stage 1 first.")
        raise FileNotFoundError("Stage 1 not completed. Run stage 1 first.")
    
    with open(stage1_file, 'r') as f:
        stage1_completed = set(json.load(f))
    
    logger.info(f"Stage 1 completed {len(stage1_completed)} galaxies. Processing these for source extraction.")
    logger.info(f"Detection threshold: {params['threshold']} sigma")
    
    # Track completed galaxies
    completed_file = os.path.join(params['results_dir'], 'stage2_completed.json')
    source_params_file = os.path.join(params['results_dir'], 'source_parameters.json')
    
    completed = set()
    source_params_dict = {}
    
    if resume and os.path.exists(completed_file):
        with open(completed_file, 'r') as f:
            completed = set(json.load(f))
        with open(source_params_file, 'r') as f:
            source_params_dict = json.load(f)
        logger.info(f"Resuming: {len(completed)} galaxies already processed")
    
    failed = []
    processed_count = 0
    
    # In test mode, limit the number of galaxies to process
    stage1_list = list(stage1_completed)
    if test_mode:
        stage1_list = stage1_list[:test_count]
        logger.info(f"TEST MODE: Limited to first {len(stage1_list)} galaxies from stage 1")
    
    desc = f"Source extraction (TEST MODE - {len(stage1_list)} galaxies)" if test_mode else "Source extraction"
    
    for subhaloId in tqdm(stage1_list, desc=desc):
        subhaloId = int(subhaloId)
        
        if subhaloId in completed:
            logger.debug(f"SubhaloID {subhaloId}: Already completed, skipping")
            continue
            
        try:
            processed_count += 1
            logger.info(f"SubhaloID {subhaloId}: Starting source extraction (processing {processed_count})")
            
            # Load noisy image
            noisy_path = os.path.join(params['noisy_dir'], f'noisy_broadband_{subhaloId}.fits')
            if not os.path.exists(noisy_path):
                logger.error(f"SubhaloID {subhaloId}: Noisy file not found: {noisy_path}")
                failed.append(subhaloId)
                continue
                
            logger.info(f"SubhaloID {subhaloId}: Loading noisy FITS file: {noisy_path}")
            noisy_hdul = fits.open(noisy_path)
            
            # Log FITS file information
            log_fits_info(noisy_hdul, subhaloId, logger, "STAGE2")
            
            # Source extraction
            logger.info(f"SubhaloID {subhaloId}: Running source detection on coadd image")
            source_params = detect_sources(noisy_hdul, bands=params['bands'], threshold=params['threshold'])
            
            # Log source detection results
            logger.info(f"SubhaloID {subhaloId}: Source detected at position ({source_params['x']:.2f}, {source_params['y']:.2f})")
            logger.info(f"SubhaloID {subhaloId}: Ellipse parameters: a={source_params['a']:.2f}, b={source_params['b']:.2f}, theta={source_params['theta']:.2f}")
            logger.info(f"SubhaloID {subhaloId}: Source flux: {source_params['flux']:.3e}")
            
            noisy_hdul.close()
            
            # Convert numpy types to Python types for JSON serialization
            source_params_serializable = {}
            for key, value in source_params.items():
                if key == 'background':
                    continue  # Skip background object
                if isinstance(value, np.ndarray):
                    source_params_serializable[key] = value.tolist()
                elif isinstance(value, (np.float64, np.float32)):
                    source_params_serializable[key] = float(value)
                elif isinstance(value, (np.int64, np.int32)):
                    source_params_serializable[key] = int(value)
                else:
                    source_params_serializable[key] = value
            
            source_params_dict[str(subhaloId)] = source_params_serializable
            completed.add(subhaloId)
            logger.info(f"SubhaloID {subhaloId}: Successfully completed source extraction")
            
            # Save progress periodically
            if len(completed) % 10 == 0:
                with open(completed_file, 'w') as f:
                    json.dump(list(completed), f)
                with open(source_params_file, 'w') as f:
                    json.dump(source_params_dict, f)
                logger.info(f"Progress saved: {len(completed)} galaxies completed")
            
            # In test mode, stop after processing the target count
            if test_mode and processed_count >= test_count:
                logger.info(f"TEST MODE: Processed {processed_count} galaxies, stopping")
                break
                
        except Exception as e:
            logger.error(f"SubhaloID {subhaloId}: Error during source extraction: {e}")
            failed.append(subhaloId)
            continue
    
    # Save final results
    with open(completed_file, 'w') as f:
        json.dump(list(completed), f)
    with open(source_params_file, 'w') as f:
        json.dump(source_params_dict, f)
    
    # Save failed list
    if failed:
        failed_file = os.path.join(params['results_dir'], 'stage2_failed.json')
        with open(failed_file, 'w') as f:
            json.dump(failed, f)
        logger.warning(f"Stage 2: {len(failed)} galaxies failed: {failed}")
    
    logger.info(f"Stage 2 completed: {len(completed)} successful, {len(failed)} failed")
    return completed, failed, source_params_dict

def stage3_photometry(galmeta, params, source_params_dict, resume=True, test_mode=False, test_count=5):
    """Stage 3: Perform aperture photometry."""
    
    logger = setup_logging(params, 'stage3_photometry')
    
    logger.info("=" * 50)
    logger.info("STAGE 3: Aperture photometry")
    if test_mode:
        logger.info(f"TEST MODE: Processing only {test_count} galaxies")
    logger.info("=" * 50)
    
    # Check stage 2 completion
    stage2_file = os.path.join(params['results_dir'], 'stage2_completed.json')
    if not os.path.exists(stage2_file):
        logger.error("Stage 2 not completed. Run stage 2 first.")
        raise FileNotFoundError("Stage 2 not completed. Run stage 2 first.")
    
    with open(stage2_file, 'r') as f:
        stage2_completed = set(json.load(f))
    
    logger.info(f"Stage 2 completed {len(stage2_completed)} galaxies. Processing these for photometry.")
    logger.info(f"Bands: {params['bands']}")
    logger.info(f"Zero-points: {params['zp_dict']}")
    
    # Track results
    results_file = os.path.join(params['results_dir'], 'photometry_results_partial.csv')
    results = []
    
    if resume and os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        results = existing_results.to_dict('records')
        processed_ids = set(existing_results['subhaloId'].astype(int))
        logger.info(f"Resuming: {len(processed_ids)} galaxies already processed")
    else:
        processed_ids = set()
    
    failed = []
    processed_count = 0
    
    # In test mode, limit the number of galaxies to process
    stage2_list = list(stage2_completed)
    if test_mode:
        stage2_list = stage2_list[:test_count]
        logger.info(f"TEST MODE: Limited to first {len(stage2_list)} galaxies from stage 2")
    
    desc = f"Photometry (TEST MODE - {len(stage2_list)} galaxies)" if test_mode else "Photometry"
    
    for subhaloId in tqdm(stage2_list, desc=desc):
        subhaloId = int(subhaloId)
        
        if subhaloId in processed_ids:
            logger.debug(f"SubhaloID {subhaloId}: Already completed, skipping")
            continue
            
        try:
            processed_count += 1
            logger.info(f"SubhaloID {subhaloId}: Starting aperture photometry (processing {processed_count})")
            
            # Load noisy image
            noisy_path = os.path.join(params['noisy_dir'], f'noisy_broadband_{subhaloId}.fits')
            logger.info(f"SubhaloID {subhaloId}: Loading noisy FITS file: {noisy_path}")
            noisy_hdul = fits.open(noisy_path)
            
            # Log FITS file information
            log_fits_info(noisy_hdul, subhaloId, logger, "STAGE3")
            
            # Get source parameters
            source_params = source_params_dict[str(subhaloId)]
            logger.info(f"SubhaloID {subhaloId}: Using aperture at ({source_params['x']:.2f}, {source_params['y']:.2f}) "
                       f"with a={source_params['a']:.2f}, b={source_params['b']:.2f}")
            
            # Photometry
            logger.info(f"SubhaloID {subhaloId}: Performing aperture photometry for bands {params['bands']}")
            magnitudes, colors = perform_aperture_photometry(
                noisy_hdul, source_params, 
                bands=params['bands'], 
                zp_dict=params['zp_dict']
            )
            
            # Log photometry results
            logger.info(f"SubhaloID {subhaloId}: Magnitudes: {magnitudes}")
            logger.info(f"SubhaloID {subhaloId}: Colors: {colors}")
            
            noisy_hdul.close()
            
            # Store results
            result = {
                'subhaloId': subhaloId,
                'x': source_params['x'],
                'y': source_params['y'],
                'a': source_params['a'], 
                'b': source_params['b'],
                'theta': source_params['theta'],
                **magnitudes,
                **colors
            }
            results.append(result)
            logger.info(f"SubhaloID {subhaloId}: Successfully completed photometry")
            
            # Save progress periodically
            if len(results) % 10 == 0:
                results_df = pd.DataFrame(results)
                results_df.to_csv(results_file, index=False)
                logger.info(f"Progress saved: {len(results)} galaxies completed")
            
            # In test mode, stop after processing the target count
            if test_mode and processed_count >= test_count:
                logger.info(f"TEST MODE: Processed {processed_count} galaxies, stopping")
                break
                
        except Exception as e:
            logger.error(f"SubhaloID {subhaloId}: Error during photometry: {e}")
            failed.append(subhaloId)
            continue
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    # Save failed list
    if failed:
        failed_file = os.path.join(params['results_dir'], 'stage3_failed.json')
        with open(failed_file, 'w') as f:
            json.dump(failed, f)
        logger.warning(f"Stage 3: {len(failed)} galaxies failed: {failed}")
    
    logger.info(f"Stage 3 completed: {len(results)} successful, {len(failed)} failed")
    return results_df, failed

def run_full_analysis(sim='TNG50-1', base_path=None, stages=None, resume=True, test_mode=False, test_count=5):
    """
    Run the full analysis pipeline in stages.
    
    Parameters:
    -----------
    sim : str
        Simulation name
    base_path : str, optional
        Base path to the TNGSN directory
    stages : list, optional
        List of stages to run (1, 2, 3). If None, runs all stages.
    resume : bool
        Whether to resume from previous progress
    test_mode : bool
        If True, only process a small number of galaxies for testing
    test_count : int
        Number of galaxies to process in test mode
    """
    
    if stages is None:
        stages = [1, 2, 3]
    
    # Setup
    galmeta, params = setup_paths_and_params(sim, base_path)
    
    # Setup main logger
    logger = setup_logging(params, 'main')
    logger.info(f"Starting analysis pipeline for simulation {sim}")
    logger.info(f"Stages to run: {stages}")
    logger.info(f"Resume mode: {resume}")
    if test_mode:
        logger.info(f"TEST MODE: Processing only {test_count} galaxies per stage")
    logger.info(f"Total galaxies in metadata: {len(galmeta)}")
    
    # Stage 1: Add noise
    if 1 in stages:
        completed1, failed1 = stage1_add_noise(galmeta, params, resume, test_mode, test_count)
    
    # Stage 2: Source extraction
    if 2 in stages:
        completed2, failed2, source_params_dict = stage2_source_extraction(galmeta, params, resume, test_mode, test_count)
    else:
        # Load existing source parameters if stage 2 not run
        source_params_file = os.path.join(params['results_dir'], 'source_parameters.json')
        if os.path.exists(source_params_file):
            with open(source_params_file, 'r') as f:
                source_params_dict = json.load(f)
            logger.info(f"Loaded existing source parameters for {len(source_params_dict)} galaxies")
        else:
            source_params_dict = {}
            logger.warning("No existing source parameters found")
    
    # Stage 3: Photometry
    if 3 in stages:
        results_df, failed3 = stage3_photometry(galmeta, params, source_params_dict, resume, test_mode, test_count)
        
        # Save final results
        final_output = os.path.join(params['base_path'], f'photometry_results_{sim}.csv')
        results_df.to_csv(final_output, index=False)
        logger.info(f"Final results saved to {final_output}")
        
        return results_df
    
    logger.info("Analysis pipeline completed for requested stages")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KIDS photometry analysis in stages')
    parser.add_argument('--sim', default='TNG50-1', help='Simulation name (default: TNG50-1)')
    parser.add_argument('--base-path', help='Base path to TNGSN directory (optional)')
    parser.add_argument('--stages', nargs='+', type=int, choices=[1, 2, 3], 
                       help='Stages to run (1=noise, 2=extraction, 3=photometry). Default: all stages')
    parser.add_argument('--no-resume', action='store_true', help='Start from scratch (don\'t resume)')
    parser.add_argument('--test', action='store_true', help='Run in test mode (process only a few galaxies)')
    parser.add_argument('--test-count', type=int, default=5, help='Number of galaxies to process in test mode (default: 5)')
    
    args = parser.parse_args()
    
    # Run the analysis
    results = run_full_analysis(
        sim=args.sim, 
        base_path=args.base_path, 
        stages=args.stages,
        resume=not args.no_resume,
        test_mode=args.test,
        test_count=args.test_count
    )
    
    if results is not None:
        print(f"Final analysis completed: {len(results)} galaxies processed successfully")