from netCDF4 import Dataset
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import argparse
import sys
from globals import *
from util import *
import util as util


def pre_process_tpw_product(path, station_id):
    """
    Compute the precipitable water vapor (PWV) at ``station_id`` at zenith or in direction of
    ``line_of_sight`` using the Total Precipitable Water (TPW) product by NOAA.

    Args:
        path (str): Working directory with GOES-R files in it
        station_id (str): {"A652", "San Pedro Martir", "Other"} The input "Other" leads to a dialog window 
            where the user has to enter the latitude and longitude (in degrees) of the station_id of interest.

    Returns:
        parquet file
    """

    # navigate to directory with .nc data files
    os.chdir(str(path))
    nc_files = glob.glob('*OR_ABI-L2-TPWF*')
    nc_files = sorted(nc_files)
    g16_data_file = []
    g16nc = []
    xscan = []
    yscan = []
    for i in range(0, len(nc_files)):
        nc_indx = i
        print(f"Reading file number: {nc_indx}")
        g16_data = nc_files[nc_indx]
        g16_data_file.append(g16_data)
        g16 = Dataset(g16_data_file[i], 'r')
        g16nc.append(g16)
        xtemp = g16nc[i].variables['x'][:]
        xscan.append(xtemp)
        ytemp = g16nc[i].variables['y'][:]
        yscan.append(ytemp)
        if nc_indx == 20:
            break

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
        site = 'Forte de copacabana'
    elif station_id == 'San Pedro Martir':
        latitude = 30.9058267
        longitude = -115.4254954
        site = 'San Pedro Mártir'
    elif station_id == 'Other':
        site = 'Lat; {} degrees Lon: {} degrees.'.format(latitude, longitude)
        print('First type latitude coordinate, hit Enter, then longitude coordinate and Enter again.')
        print('Latitude valid range: [-81.3282, 81.3283] \n'
              'Longitude valid range: [-156.2995, 6.2995]')
        print('\n' 'Latitude:')
        latitude = input()
        while float(latitude) < -81.3281 or float(latitude) > 81.3283:
            print('Invalid latitude range. Enter a value between -81.3282 and 81.3283')
            print('Latitude:')
            latitude = input()

        print('\n' 'Longitude:')
        longitude = input()
        while float(longitude) < -156.2995 or float(longitude) > 6.2995:
            print('Invalid longitude range. Enter a value between -156.2995 and 6.2995')
            print('Longitude:')
            longitude = input()
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
    for i in range(0, len(nc_files)):
        print(f"Reading TPW from file number: {i}")
        Xtemp = np.abs(xscan[i]-x).argmin()
        Ytemp = np.abs(yscan[i]-y).argmin()
        X.append(Xtemp)
        Y.append(Ytemp)
        TPWtemp = g16nc[i].variables['TPW'][:]
        TPWtemp = TPWtemp[Y[i], X[i]]
        if isinstance(TPWtemp.T, np.float32):
            TPW.append(TPWtemp)
        else:
            TPW.append(None)
        if i == 20:
            break

    # Record time of measurement
    t = []
    epoch = []
    date = []
    plottime = []
    for i in range(0, len(nc_files)):
        print(f"Reading Date from file number: {i}")
        ttemp = g16nc[i].variables['t'][:]
        t.append(ttemp)
        epochtemp = 946728000 + int(t[i])
        epoch.append(epochtemp)
        datetemp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(epoch[i]))
        date.append(datetemp)
        plottimetemp = time.strftime("%H:%M:%S", time.gmtime(epoch[i]))
        plottime.append(plottimetemp)
        if i == 20:
            break

    day = time.strftime("%a, %d %b %Y", time.gmtime(epoch[1]))

    # if csv:
    np.savetxt('TPW_new_{}_{}.csv'.format(site, day), np.column_stack((date, TPW)),
                delimiter=',', fmt='%s', header='Time,PWV', comments='')

    data = {'Datetime': date, 'TPW': TPW}

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Set the 'Datetime' column as the DatetimeIndex
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))

    # Remove time-related columns since now this information is in the index.
    df = df.drop(['Datetime'], axis = 1)

    parquet_dir = '../parquet_files/'

    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)

    # Specify the path where you want to save the Parquet file
    parquet_path = '../parquet_files/tpw_preprocessed_file.parquet'

    # Save the DataFrame to a Parquet file
    df.to_parquet(parquet_path, compression='gzip')

    return date, TPW

def main(argv):
    parser = argparse.ArgumentParser(description='Preprocess ABI products station data.')
    parser.add_argument('-s', '--station_id', required=True, choices=INMET_STATION_CODES_RJ, help='ID of the weather station to preprocess data for.')
    # args = parser.parse_args(argv[1:])
    # print(args)

    directory = 'data/goes16/abi_files'

    station_id = 'A652' # args.station_id

    # print(f'Going to preprocess data sources according to station id ({args.station_id})...')

    print('\n***Preprocessing weather station data***')

    pre_process_tpw_product(directory, station_id)

    print('Done!')

if __name__ == '__main__':
    main(sys.argv)
