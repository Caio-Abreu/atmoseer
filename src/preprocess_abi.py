from netCDF4 import Dataset
import os
import numpy as np
import time
import glob
import pandas as pd
import pyarrow.parquet as pq

import argparse
import sys

from globals import *
from util import *

def read_and_process_files(files, station_id):
    """
    Read and process a batch of NetCDF files containing TPW data.

    Args:
        files (list of str): A list of file paths to NetCDF files.
        station_id (str): The station ID for processing data.

    Returns:
        tuple: A tuple containing lists of dates, X, Y, and TPW values.
            - dates (list of str): Date and time values.
            - TPW (list of float or None): Total Precipitable Water values.
    """
    g16_data_file = []
    g16nc = []
    xscan = []
    yscan = []
    for i in range(0, len(files)):
        nc_indx = i
        g16_data = files[nc_indx]
        g16_data_file.append(g16_data)
        g16 = Dataset(g16_data_file[i], 'r')
        g16nc.append(g16)
        xtemp = g16nc[i].variables['x'][:]
        xscan.append(xtemp)
        ytemp = g16nc[i].variables['y'][:]
        yscan.append(ytemp)

    # GOES-R projection info and retrieving relevant constants
    proj_info = g16nc[0].variables['goes_imager_projection']
    lon_origin = proj_info.longitude_of_projection_origin
    H = proj_info.perspective_point_height+proj_info.semi_major_axis
    r_eq = proj_info.semi_major_axis
    r_pol = proj_info.semi_minor_axis

    e = np.sqrt((r_eq**2-r_pol**2)/(r_eq**2))
    rad = (np.pi)/180
    if station_id == 'A652':
        latitude = -22.98833
        longitude = -43.19055
    elif station_id == 'A602':
        latitude = -23.050278
        longitude = -43.595556
    elif station_id == 'A621':
        latitude = -22.861389
        longitude = -43.411389
    elif station_id == 'A636':
        latitude = -22.940000
        longitude = -43.402778
    elif station_id == 'A627':
        latitude = -22.867500
        longitude = -43.101944
    lat = rad*latitude
    lon = rad*longitude
    lambda_0 = rad*lon_origin
    lat_origin = np.arctan(((r_pol**2)/(r_eq**2))*np.tan(lat))
    r_c = r_pol/(np.sqrt(1-(e**2)*(np.cos(lat_origin))**2))
    s_x = H - r_c*np.cos(lat_origin)*np.cos(lon-lambda_0)
    s_y = -r_c*np.cos(lat_origin)*np.sin(lon-lambda_0)
    s_z = r_c*np.sin(lat_origin)
    s = np.sqrt(s_x**2+s_y**2+s_z**2)

    # Convert into scan angles
    x = np.arcsin(-s_y/s)
    y = np.arctan(s_z/s_x)

    X = []
    Y = []
    TPW = []
    for i in range(0, len(files)):
        Xtemp = np.abs(xscan[i]-x).argmin()
        Ytemp = np.abs(yscan[i]-y).argmin()
        X.append(Xtemp)
        Y.append(Ytemp)
        flag = g16nc[i].variables['DQF_Overall'][:]
        if flag[Y[i], X[i]] <= 1:
            TPWtemp = g16nc[i].variables['TPW'][:]
            TPWtemp = TPWtemp[Y[i], X[i]]
            if isinstance(TPWtemp.T, np.float32):
                TPW.append(TPWtemp)
            else:
                TPW.append(None)
        else:
            TPW.append(None)

    # Record time of measurement
    t = []
    epoch = []
    date = []
    for i in range(0, len(files)):
        ttemp = g16nc[i].variables['t'][:]
        t.append(ttemp)
        epochtemp = 946728000 + int(t[i])
        epoch.append(epochtemp)
        datetemp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(epoch[i]))
        date.append(datetemp)
        # Close the NetCDF file
        g16nc[i].close()

    return date, TPW

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
    nc_files = glob.glob('*OR_ABI-L2-TPWF*')
    nc_files = sorted(nc_files)

    parquet_dir = '/home/cribeiro/atmoseer/data/parquet_files'

    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)

    parquet_path = f'/home/cribeiro/atmoseer/data/parquet_files/tpw_{station_id}_preprocessed_file.parquet'

    batch_size = 1000
    total_files = len(nc_files)

    df_date = []
    df_tpw = []

    print(f"You have {total_files} to be processed")

    for i in range(0, total_files, batch_size):
        batch_files = nc_files[i:i+batch_size]
        dates, TPW = read_and_process_files(batch_files, station_id)
        df_date.extend(dates)
        df_tpw.extend(TPW)
        print(f"{len(df_date)} Files was pre processed")

    # Create a DataFrame from the list of dictionaries
    data = {'Datetime': df_date, 'TPW': df_tpw}
    df = pd.DataFrame(data)

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

    print(f"Saving file in {parquet_path}")

    # Save the combined DataFrame to a Parquet file
    df_combined.to_parquet(parquet_path, compression='gzip')

    return

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocess ABI products station data.')
    parser.add_argument('-s', '--station_id', required=True, help='ID of the weather station to preprocess data for.')
    args = parser.parse_args(argv[1:])

    directory = 'data/goes16/abi_files'
    
    station_id = args.station_id

    print('\n***Preprocessing TPW Files***')
    pre_process_tpw_product(directory, station_id)
    print('Done!')

if __name__ == '__main__':
    main(sys.argv)