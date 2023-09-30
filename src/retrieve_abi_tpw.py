import pandas as pd
import sys
from datetime import datetime
from globals import *
import s3fs
import xarray as xr
import os
import tenacity
from botocore.exceptions import ConnectTimeoutError
from concurrent.futures import ThreadPoolExecutor

# Use the anonymous credentials to access public data
fs = s3fs.S3FileSystem(anon=True)

# Create folders for storing individual files and the parquet file
os.makedirs("data/goes16/individual_files", exist_ok=True)
os.makedirs("data/goes16/parquet_files", exist_ok=True)

files_list = []

# Download a single file and process it
@tenacity.retry(
    retry=tenacity.retry_if_exception_type((ConnectTimeoutError, OSError)),
    wait=tenacity.wait_exponential(),
    stop=tenacity.stop_after_attempt(5)
)
def process_file(file):
    """
    Download a GOES-16 netCDF file from an S3 bucket, filter it for events that fall within specified coordinates,
    and return the filtered data as a Pandas DataFrame.

    Args:
        file (str): The name of the file to be downloaded from an S3 bucket.

    Returns:
        pd.DataFrame: The filtered data as a Pandas DataFrame.
    """
    filename = file.split('/')[-1]
    fs.get(file, f"data/goes16/abi_files/{filename}")
    filepath = os.path.abspath(f"data/goes16/abi_files/{filename}")
    ds = xr.open_dataset(filepath)
    files_list.append(ds)


def download_files(files):
    """
    Downloads multiple GOES-16 netCDF files in parallel, filters them for events that fall within specified coordinates,
    and returns the filtered data as a list of Pandas DataFrames.

    Args:
        files (list): A list of strings representing the names of files to be downloaded from an S3 bucket.

    Returns:
        list: The filtered data as a list of Pandas DataFrames.
    """
    with ThreadPoolExecutor() as executor:
        executor.map(process_file, files)


def import_data(initial_year, final_year):
    """
    Downloads and saves GOES-16 data files from Amazon S3 for a given station code and time period.

    Args:
        initial_year (int): The initial year of the time period to download data for.
        final_year (int): The final year of the time period to download data for.

    Returns:
        None

    This function first reads a CSV file with relevant dates to download data for, then constructs a list of
    file paths for the requested station code and time period using these dates. The files are then downloaded
    using multithreading for parallel processing.

    Note: This function assumes that the relevant data files are stored in the Amazon S3 bucket 'noaa-goes16'.
    """
    # Get files of GOES-16 data (multiband format) on multiple dates
    # format: <Product>/<Year>/<Day of Year>/<Hour>/<Filename>
    hours = [f'{h:02d}' for h in range(25)]  # Download all 24 hours of data

    start_date = pd.to_datetime(f'{initial_year}-01-01')
    end_date = pd.to_datetime(f'{final_year}-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    count = 0
    files = []
    for date in dates:
        year = str(date.year)
        day_of_year = f'{date.dayofyear:03d}'
        print(f'noaa-goes16/ABI-L2-TPWF/{year}/{day_of_year}')
        for hour in hours:
            target = f'noaa-goes16/ABI-L2-TPWF/{year}/{day_of_year}/{hour}'
            files.extend(fs.ls(target))

    download_files(files)

    if len(files_list) > 0:
        pass    
        # concatenate datasets along the time dimension
        # merged_ds = xr.concat(files_list, dim='t')

        # merged_df = merged_ds.to_dataframe()

        # # Save merged dataframe to a Parquet file
        # merged_df.to_parquet("data/goes16/parquet_files/goes16_ABI_merged_file.parquet")
    else:
        print("No data found within the specified Date.")


def main(argv):
    start_year = 2019
    end_year = datetime.now().year

    help_message = "{0} -b <begin> -e <end>".format(argv[0])

    # try:
    #     opts, args = getopt.getopt(argv[1:], "hs:b:e:t:", ["help", "begin=", "end="])
    # except:
    #     print(help_message)
    #     sys.exit(2)

    # for opt, arg in opts:
    #     if opt in ("-h", "--help"):
    #         print(help_message)
    #         sys.exit(2)
    #     elif opt in ("-b", "--begin"):
    #         if not is_posintstring(arg):
    #             sys.exit("Argument start_year must be an integer. Exit.")
    #         start_year = int(arg)
    #     elif opt in ("-e", "--end"):
    #         if not is_posintstring(arg):
    #             sys.exit("Argument end_year must be an integer. Exit.")
    #         end_year = int(arg)

    assert (start_year <= end_year)

    import_data(start_year, end_year)
   
if __name__ == "__main__":
    main(sys.argv)
