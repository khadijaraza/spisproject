# final_pipeline.py

import lightkurve as lk
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import warnings
import multiprocessing
import os
import astropy.units as u
from collections import Counter
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.mast import Catalogs

# Ignore harmless warnings from astroquery about messy data in unused columns
warnings.filterwarnings('ignore', message='.*has a unit but is kept as a Column.*')

# =============================================================================
# HELPER FUNCTION TO CONDENSE ERROR MESSAGES
# =============================================================================
def get_error_summary(exception_obj, max_len=150):
    """Creates a short, readable summary of an exception message."""
    summary = str(exception_obj).split('\n')[0]
    if len(summary) > max_len:
        summary = summary[:max_len] + "..."
    return summary

def load_tics_from_file(filename):
    """
    Loads a list of TIC IDs from a text file.

    Args:
        filename (str): The path to the text file.

    Returns:
        list: A list of TIC IDs as strings, or an empty list if the file is not found.
    """
    # First, check if the file exists to avoid errors
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return []
    
    # Use 'with open' to safely open and automatically close the file
    with open(filename, 'r') as f:
        # Use a list comprehension to read each line, strip the extra whitespace/newline
        # characters, and add it to the list.
        tic_list = [line.strip() for line in f]
        
    return tic_list

def load_collections_from_disk(input_folder):
    """
    Loads all FITS files from a directory into a list of 
    LightCurveCollection objects.

    Args:
        input_folder (str): The path to the folder containing the FITS files.

    Returns:
        list: A list where each item is a LightCurveCollection for a distinct star.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Directory '{input_folder}' not found.")
        return []
    
    list_of_collections = []
    # Get a list of all the FITS files in the directory
    fits_files = [f for f in os.listdir(input_folder) if f.endswith('.fits')]
    
    print(f"Loading {len(fits_files)} collections from '{input_folder}'...")
    for filename in tqdm(fits_files, desc=f"Loading from {input_folder}"):
        filepath = os.path.join(input_folder, filename)
        try:
            # `lightkurve.read` can load both single light curves and collections
            lc_or_collection = lk.read(filepath)
            list_of_collections.append(lc_or_collection)
        except Exception as e:
            print(f"\nWarning: Could not load file {filename}. Reason: {e}")
            continue
            
    return list_of_collections

# =============================================================================
# STEP 1: GATHER RAW TIC IDs
# =============================================================================
def get_raw_tic_lists():
    """Queries archives to get initial lists of planet and false positive TICs."""
    print("--- Step 1: Gathering Raw TIC Lists ---")
    try:
        confirmed_planets_catalog = NasaExoplanetArchive.query_criteria(table="pscomppars", select="tic_id")
        confirmed_planets_df = confirmed_planets_catalog.to_pandas()
        planet_tics = [f"TIC {tic_id}" for tic_id in confirmed_planets_df['tic_id'].dropna().astype(str).str.replace('TIC ', '').unique() if tic_id]
        print(f"Found {len(planet_tics)} unique raw TIC IDs for confirmed planets.")
    except Exception as e:
        print(f"Could not query NASA Exoplanet Archive for confirmed planets: {get_error_summary(e)}")
        planet_tics = []

    try:
        toi_catalog = NasaExoplanetArchive.query_criteria(table="toi", select="tid, tfopwg_disp")
        toi_df = toi_catalog.to_pandas()
        fp_df = toi_df[toi_df['tfopwg_disp'] == 'FP']
        fp_tics = [f"TIC {int(tid)}" for tid in fp_df['tid'].dropna().unique()]
        print(f"Found {len(fp_tics)} unique raw TIC IDs classified as False Positives.")
    except Exception as e:
        print(f"Could not query TOI Catalog for false positives: {get_error_summary(e)}")
        fp_tics = []
    
    return planet_tics, fp_tics

# =============================================================================
# STEP 2: VALIDATE TIC IDs
# =============================================================================
def validate_tics_resumable(raw_tic_list, progress_file):
    """Validates a list of TICs and saves progress, making it resumable."""
    print(f"\n--- Step 2: Validating TICs (log file: {progress_file}) ---")
    already_validated = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            already_validated = set(line.strip() for line in f)
    print(f"Loaded {len(already_validated)} previously validated TICs.")

    clean_tic_list = list(already_validated)
    with open(progress_file, 'a') as f_progress:
        tics_to_check = [tic for tic in raw_tic_list if tic not in already_validated]
        if not tics_to_check:
            print("No new TICs to validate.")
            return clean_tic_list

        print(f"Validating {len(tics_to_check)} new TICs...")
        for tic in tqdm(tics_to_check, desc="Validating TICs"):
            try:
                search = lk.search_lightcurve(target=tic, mission="TESS", author = "TESS-SPOC")
                if len(search) > 0:
                    f_progress.write(f"{tic}\n")
                    clean_tic_list.append(tic)
            except Exception:
                continue
    return clean_tic_list

# =============================================================================
# STEP 3: BATCH DOWNLOAD DATA
# =============================================================================
def download_hybrid_resumable(list_of_tics, output_folder, batch_size=250):
    """
    Downloads data using a fast batch method with a robust one-by-one fallback
    for any failed batches. The process is fully resumable.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n--- Starting Hybrid Download for {len(list_of_tics)} TICs into '{output_folder}' ---")

    # First, see what we've already downloaded to make it resumable
    already_downloaded_ids = {f.split('_')[1].split('.')[0] for f in os.listdir(output_folder)}
    pending_tics = [tic for tic in list_of_tics if tic.split(' ')[1] not in already_downloaded_ids]
    
    if not pending_tics:
        print("All TICs have already been downloaded.")
        return

    print(f"Found {len(already_downloaded_ids)} already downloaded TICs. Resuming download for {len(pending_tics)} TICs.")
    
    num_batches = int(np.ceil(len(pending_tics) / batch_size))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Suppress harmless warnings
        
        for i in tqdm(range(num_batches), desc="Downloading Batches"):
            batch_tics = pending_tics[i*batch_size : (i+1)*batch_size]
            
            # --- OPTIMISTIC FAST PATH ---
            try:
                search_result = lk.search_lightcurve(target=batch_tics, mission="TESS", author="TESS-SPOC")
                if len(search_result) > 0:
                    lc_collection_batch = search_result.download_all()
                    # Group by star and save each one's collection
                    for tic_id in lc_collection_batch.unique_targets:
                        tic_collection = lc_collection_batch[lc_collection_batch.TARGETID == tic_id]
                        output_path = os.path.join(output_folder, f"tic_{tic_id}.fits")
                        tic_collection.write(output_path)
            
            # --- ROBUST FALLBACK PATH ---
            except Exception as e:
                print(f"\nBatch {i+1} failed ({e}). Switching to one-by-one download for this batch...")
                for tic in tqdm(batch_tics, desc="--> Downloading one-by-one", leave=False):
                    try:
                        tic_id_num = tic.split(' ')[1]
                        output_path = os.path.join(output_folder, f"tic_{tic_id_num}.fits")
                        if os.path.exists(output_path):
                            continue
                            
                        search = lk.search_lightcurve(target=tic, mission="TESS", author="TESS-SPOC")
                        if len(search) > 0:
                            lc_collection = search.download_all()
                            if lc_collection:
                                lc_collection.write(output_path)
                    except Exception:
                        continue # Skip any individual TICs that still fail

# =============================================================================
# STEP 4: PREPROCESSING FUNCTIONS
# =============================================================================
def find_transit_parameters(lc_collection, min_power=10):
    """Robustly finds the period, duration, and transit time for a star."""
    results = []
    duration_grid = np.linspace(0.04, 0.5, 20)
    for lc_sector in lc_collection:
        try:
            bls = lc_sector.remove_nans().to_periodogram('bls', duration=duration_grid, minimum_period=np.max(duration_grid) + 0.01)
            power = np.max(bls.power.value)
            if power > min_power:
                results.append({
                    'period': bls.period_at_max_power,
                    'duration': bls.duration_at_max_power,
                    'transit_time': bls.transit_time_at_max_power,
                    'power': power
                })
        except Exception:
            continue
    if results:
        return max(results, key=lambda x: x['power'])
    return None

def process_lightcurve(stitched_lc, params):
    """Processes a stitched light curve using pre-found parameters."""
    try:
        period, duration, transit_time = params['period'], params['duration'], params['transit_time']
        window = int(duration.value * 5); window += 1 if window % 2 == 0 else 0; window = max(21, window)
        flat_lc = stitched_lc.flatten(window_length=window)
        folded_lc = flat_lc.fold(period=period, epoch_time=transit_time)
        zoomed_lc = folded_lc[(folded_lc.time.value > -0.5) & (folded_lc.time.value < 0.5)]
        binned_lc = zoomed_lc.bin(time_bin_size=0.01)
        median_flux = np.nanmedian(binned_lc.flux.value)
        if np.isclose(median_flux, 0): return None
        return binned_lc / median_flux
    except Exception:
        return None

def process_target_for_parallel(args):
    """A wrapper function for the parallel processing pool."""
    lc_collection_for_tic, label = args
    target_id = lc_collection_for_tic[0].TARGETID
    try:
        params = find_transit_parameters(lc_collection_for_tic)
        if not params: return None
        stitched_lc = lc_collection_for_tic.stitch()
        processed_lc = process_lightcurve(stitched_lc, params)
        if processed_lc:
            return (processed_lc, label)
        return None
    except Exception:
        return None

# =============================================================================
# MAIN SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    # --- STAGE 1 & 2: Get and Validate TIC Lists ---

    raw_planet_tics, raw_fp_tics = get_raw_tic_lists()
    clean_planet_tics = validate_tics_resumable(raw_planet_tics, 'validated_planet_tics.txt')
    clean_fp_tics = validate_tics_resumable(raw_fp_tics, 'validated_fp_tics.txt')
    
    # --- STAGE 3: Download Data in Batches ---
    download_hybrid_resumable(clean_planet_tics, "./planets")
    download_hybrid_resumable(clean_fp_tics, "./false_positives")
    list_of_planet_collections = load_collections_from_disk("./planets")
    list_of_fp_collections = load_collections_from_disk("./false_positives")
    
    # --- STAGE 4: Parallel Preprocessing ---
    print("\n--- Step 4: Preparing Data for Parallel Processing ---")
    tasks = [(collection, True) for collection in list_of_planet_collections] + \
            [(collection, False) for collection in list_of_fp_collections]
    
    if not tasks:
        print("No data was downloaded to process.")
    else:
        print(f"Prepared {len(tasks)} targets for processing.")
        print("\n--- Starting Parallel Processing on All CPU Cores ---")
        # Use a multiprocessing Pool to run tasks in parallel
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(process_target_for_parallel, tasks), total=len(tasks)))
        
        # --- STAGE 5: Save Final Dataset ---
        full_dataset = [res for res in results if res is not None]
        output_filename = 'final_dataset.pkl'
        try:
            with open(output_filename, 'wb') as f:
                pickle.dump(full_dataset, f)
            print(f"\n✅ Dataset with {len(full_dataset)} items was successfully saved to '{output_filename}'")
        except Exception as e:
            print(f"\n❌ Error saving dataset to file: {e}")