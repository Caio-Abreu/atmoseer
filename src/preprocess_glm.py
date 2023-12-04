from netCDF4 import Dataset
import os
import numpy as np
import time
import glob
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

import argparse
import sys

from globals import *
from util import *

station_ids_for_goes16 = {
    "A652": {
        "name": "forte de copacabana",
        "n_lat": -22.717,
        "s_lat": -23.083,
        'w_lon': -43.733,
        'e_lon': -42.933
        },
    "A602": {
        "n_lat": -23.000,
        "s_lat": -23.100,
        'e_lon': -43.400,
        'w_lon': -43.600
        },
    "A621": {
        "n_lat": -22.800,
        "s_lat": -22.900,
        'e_lon': -43.400,
        'w_lon': -43.450
        },
    "A636": {
        "n_lat": -22.900,
        "s_lat": -22.950,
        'e_lon': -43.400,
        'w_lon': -43.450
        },
    "A627": {
        "n_lat": -22.850,
        "s_lat": -22.900,
        'e_lon': -43.100,
        'w_lon': -43.150
        }
    }

# Latitude and Longitude of RJ
def filter_coordinates(ds, station_id):
  """
    Filter lightning event data in an xarray Dataset based on latitude and longitude boundaries.

    Args:
        ds (xarray.Dataset): Dataset containing lightning event data with variables `event_energy`, `event_lat`, and `event_lon`.
        station_id (str): Station string

    Returns:
        xarray.Dataset: A new dataset with the same variables as `ds`, but with lightning events outside of the specified latitude and longitude boundaries removed.
  """
  # Create a mask based on your conditions
  return ds['event_energy'].where(
        (ds['event_lat'] >= station_ids_for_goes16[station_id]['s_lat']) & (ds['event_lat'] <= station_ids_for_goes16[station_id]['n_lat']) &
        (ds['event_lon'] >= station_ids_for_goes16[station_id]['w_lon']) & (ds['event_lon'] <= station_ids_for_goes16[station_id]['e_lon']),
      drop=True)


def read_and_process_files(files, station_id, g16_pre_process_data_file):
    """
    Read and process a batch of NetCDF files containing TPW data.

    Args:
        files (list of str): A list of file paths to NetCDF files.
        station_id (str): The station ID for processing data.

    Returns:
        g16_pre_process_data_file (list): list with file pre processed
    """
    g16_data_file = []
    for i in range(0, len(files)):
        try:
            nc_indx = i
            g16_data = files[nc_indx]
            g16_data_file.append(g16_data)
            # ds = Dataset('your_netcdf_file.nc', 'r')
            ds = xr.open_dataset(g16_data_file[i], cache=False, )
            ds = filter_coordinates(ds, station_id)
            df = ds.to_dataframe()
            df['event_time_offset'] = df['event_time_offset'].astype('datetime64[us]')
            g16_pre_process_data_file['Datetime'].extend(df['event_time_offset'])
            g16_pre_process_data_file['event_energy'].extend(df['event_energy'])
            ds.close()
        except:
            pass

    return g16_pre_process_data_file

def pre_process_tpw_product(path, station_id):
    """
    Preprocess Total Precipitable Water (TPW) data from NetCDF files and save it to a Parquet file.

    Args:
        path (str): The path to the directory containing NetCDF data files.
        station_id (str): The station ID for processing data.

    Returns:
        None

    This function reads TPW data from a batch of NetCDF files, processes it, and saves it to a Parquet file.
    The function navigates to the specified directory, collects TPW files, and processes them in batches of
    1000 files to optimize memory usage. Processed data is stored in a Pandas DataFrame and then appended
    to an existing Parquet file or a new one is created if it doesn't exist. Finally, the function returns
    None after completing the preprocessing and saving.
    """
    # navigate to directory with .nc data files
    os.chdir(str(path))
    nc_files = glob.glob('*GLM-L2-LCFA*')
    nc_files = sorted(nc_files)

    parquet_dir = '/home/cribeiro/atmoseer/data/parquet_files'

    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)

    parquet_path = f'/home/cribeiro/atmoseer/data/parquet_files/glm_{station_id}_preprocessed_file.parquet'

    batch_size = 1000
    total_files = len(nc_files)

    g16_pre_process_data_file = {'Datetime': [], 'event_energy': []}

    print(f"You have {total_files} to be processed")

    for i in range(0, total_files, batch_size):
        batch_files = nc_files[i:i+batch_size]
        read_and_process_files(batch_files, station_id, g16_pre_process_data_file)
        print(f"{len(g16_pre_process_data_file)} Files was pre processed")

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(g16_pre_process_data_file)

    # Set the 'Datetime' column as the DatetimeIndex
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))

    # Remove time-related columns since now this information is in the index.
    df = df.drop(['Datetime'], axis=1)

    # Append to the existing Parquet file or create a new one
    if os.path.exists(parquet_path):
        table = pq.read_table(parquet_path)
        df_existing = table.to_pandas()
        df_combined = pd.concat([df_existing, df])
    else:
        df_combined = df

    # Save the combined DataFrame to a Parquet file
    df_combined.to_parquet(parquet_path, compression='gzip')

    return

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocess ABI products station data.')
    parser.add_argument('-s', '--station_id', required=True, help='ID of the weather station to preprocess data for.')
    args = parser.parse_args(argv[1:])

    directory = '/home/cribeiro/atmoseer/data/goes16/glm_files_new'

    station_id = args.station_id

    print('\n***Preprocessing GLM Files***')
    pre_process_tpw_product(directory, station_id)
    print('Done!')

if __name__ == '__main__':
    main(sys.argv)