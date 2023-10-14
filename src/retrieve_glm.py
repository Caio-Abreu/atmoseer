import pandas as pd
import sys, getopt
from datetime import datetime
from util import is_posintstring
import globals
import s3fs
import xarray as xr
import os
import tenacity
from botocore.exceptions import ConnectTimeoutError
import concurrent.futures

# Use the anonymous credentials to access public data
fs = s3fs.S3FileSystem(anon=True)

# Latitude and Longitude of RJ
def filter_coordinates(ds:xr.Dataset):
  """
    Filter lightning event data in an xarray Dataset based on latitude and longitude boundaries.

    Args:
        ds (xarray.Dataset): Dataset containing lightning event data with variables `event_energy`, `event_lat`, and `event_lon`.

    Returns:
        xarray.Dataset: A new dataset with the same variables as `ds`, but with lightning events outside of the specified latitude and longitude boundaries removed.
  """
  return ds['event_energy'].where(
      (ds['event_lat'] >= -22.9035) & (ds['event_lat'] <= -22.7469) &
      (ds['event_lon'] >= -43.7958) & (ds['event_lon'] <= -43.0962),
      drop=True)

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
    files_process = []
    
    def process_file(file):
        print(f"Reading file: {file}")
        filename = f"data/goes16/glm_files/{file.split('/')[-1]}"
        try:
            fs.get(file, filename)
            ds = xr.open_dataset(filename)
            ds = filter_coordinates(ds)
            if ds.number_of_events.nbytes != 0:
                df = ds.to_dataframe()
                df['event_time_offset'] = df['event_time_offset'].astype('datetime64[us]')
                files_process.append(df)
            ds.close()
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
        finally:
            os.remove(filename)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_file, files)

    if len(files_process) > 0:
        # concatenate datasets along the time dimension
        merged_df = pd.concat(files_process)

        # Save merged dataframe to a Parquet file
        merged_df.to_parquet("data/goes16/goes16_merged_file.parquet")
    else:
        print("No data found within the specified coordinates and Date.")


def import_data(station_code, initial_year, final_year):
    """
    Downloads and saves GOES-16 data files from Amazon S3 for a given station code and time period.

    Args:
        station_code (str): The station code to download data for.
        initial_year (int): The initial year of the time period to download data for.
        final_year (int): The final year of the time period to download data for.

    Returns:
        None

    This function first reads a CSV file with relevant dates to download data for, then constructs a list of
    file paths for the requested station code and time period using these dates. The files are then downloaded
    using a thread pool executor for parallel processing.

    Note: This function assumes that the relevant data files are stored in the Amazon S3 bucket 'noaa-goes16'.
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

    start_goes_16 = 2018

    # help_message = "{0} -s <station_id> -b <begin> -e <end>".format(argv[0])
    
    # try:
    #     opts, args = getopt.getopt(argv[1:], "hs:b:e:t:", ["help", "station=", "begin=", "end="])
    # except:
    #     print(help_message)
    #     sys.exit(2)
    
    # for opt, arg in opts:
    #     if opt in ("-h", "--help"):
    #         print(help_message)
    #         sys.exit(2)
    #     # elif opt in ("-s", "--station"):
    #     #     station_code = arg
    #     #     if not ((station_code == "all") or (station_code in INMET_STATION_CODES_RJ)):
    #     #         print(help_message)
    #     #         sys.exit(2)
    #     elif opt in ("-b", "--begin"):
    #         if not is_posintstring(arg):
    #             sys.exit("Argument start_year must be an integer. Exit.")
    #         start_year = int(arg)
    #     elif opt in ("-e", "--end"):
    #         if not is_posintstring(arg):
    #             sys.exit("Argument end_year must be an integer. Exit.")
    #         end_year = int(arg)

    # assert (station_code is not None) and (station_code != '')
    # assert (start_year <= end_year) and (start_year >= start_goes_16)

    station_code = 'copacabana'
    start_year = 2018
    end_year = 2023

    import_data(station_code, start_year, end_year)


if __name__ == "__main__":
    main(sys.argv)

