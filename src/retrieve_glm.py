import pandas as pd
import sys
import argparse
import globals
import s3fs
import xarray as xr
import os
import tenacity
from botocore.exceptions import ConnectTimeoutError
import concurrent.futures

# Use the anonymous credentials to access public data
fs = s3fs.S3FileSystem(anon=True)

# Download all files in parallel, and rename them the same name (without the directory structure)
@tenacity.retry(
    retry=tenacity.retry_if_exception_type(ConnectTimeoutError),
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_attempt(5)
)
def download_file(files):
    """
    Downloads GOES-16 netCDF files from an S3 bucket, filters them for events that fall within specified coordinates,
    and saves the filtered data to disk as a Parquet file.

    Args:
        files (list): A list of strings representing the names of files to be downloaded from an S3 bucket.
    """
    
    def process_file(file):
        print(f"Reading file: {file}")
        filename = f"data/goes16/glm_files/{file.split('/')[-1]}"
        try:
            fs.get(file, filename)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_file, files)


def import_data(initial_year, final_year):
    """
    Downloads and saves GOES-16 (Geostationary Operational Environmental Satellite) data files from the Amazon S3 bucket for a specified time period.

    This function targets the 'noaa-goes16' S3 bucket, specifically accessing the GLM-L2-LCFA (Geostationary Lightning Mapper - Level 2 Lightning Detection) dataset. 
    It downloads data for every day between the initial and final years specified, covering all hours of each day.

    The data is organized in a multiband format and follows the structure: <Product>/<Year>/<Day of Year>/<Hour>/<Filename>.

    Args:
        initial_year (int): The initial year of the time period for which data is to be downloaded. The data for this year starts from January 1st.
        final_year (int): The final year of the time period for which data is to be downloaded. The data for this year includes up to December 31st.

    Returns:
        None. The function saves the downloaded files to a specified location (not defined in this docstring).

    Note: 
    - This function requires a pre-established connection to Amazon S3 and appropriate permissions to access the 'noaa-goes16' bucket.
    - It assumes the GOES-16 data is in the correct format and available for the entire range from the initial_year to the final_year.
    - Ensure sufficient storage space and network bandwidth as the dataset might be large, especially for longer time periods.
    - The function handles data for 24 hours each day, using a 0-23 hour format.
    """
    # Get files of GOES-16 data (multiband format) on multiple dates
    # format: <Product>/<Year>/<Day of Year>/<Hour>/<Filename>
    hours = [f'{h:02d}' for h in range(25)]  # Download all 24 hours of data

    start_date = pd.to_datetime(f'{initial_year}-01-01')
    end_date = pd.to_datetime(f'{final_year}-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    files = []
    for date in dates:
        year = str(date.year)
        day_of_year = f'{date.dayofyear:03d}'
        print(f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year}')
        for hour in hours:
            target = f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year}/{hour}'
            files.extend(fs.ls(target))

    download_file(files)

def main(argv):
    parser = argparse.ArgumentParser(description='Downlaod GLM files')
    parser.add_argument('-b', '--start', required=True, help='Start year to get data')
    parser.add_argument('-e', '--end', required=True, help='End year to get data')
    args = parser.parse_args(argv[1:])

    start_year = args.start
    end_year = args.end

    assert (start_year <= end_year)

    import_data(start_year, end_year)


if __name__ == "__main__":
    main(sys.argv)

