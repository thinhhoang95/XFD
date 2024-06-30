import pandas as pd
import numpy as np
from path_prefix import PATH_PREFIX, extract_file_name
import os 
import multiprocessing
from functools import partial

def create_demand_sheet_directory() -> None:
    # Create f'{PATH_PREFIX}/data/osstate/demand' directory if it does not exist
    directory = f'{PATH_PREFIX}/data/osstate/demand'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'{directory} created because it did not exist')

def unix_time_to_datetime(unix_time: int) -> str:
    # convert unix time to python datetime
    dt = pd.to_datetime(unix_time, unit='s')
    # format dt to be 'YYYY-MM-DD-HH-MM'
    dt = dt.strftime('%Y-%m-%d-%H-%M')
    return dt

def extract_takeoffs_and_landings(file_names: list, df_airport: pd.DataFrame) -> None:
    for file_name in file_names: # file_name must include .csv.gz extension
        print('Extracting takeoffs and landings for', file_name)
        file_name_plain = extract_file_name(file_name) # remove the .csv.gz extension
        # Create f'{PATH_PREFIX}/data/osstate/demand/{file_name_plain}' directory if it does not exist
        directory = f'{PATH_PREFIX}/data/osstate/demand/{file_name_plain}'
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'{directory} created because it did not exist')
        takeoff_demand_file = open(f'{PATH_PREFIX}/data/osstate/demand/{file_name_plain}/takeoff.csv', 'w')
        landing_demand_file = open(f'{PATH_PREFIX}/data/osstate/demand/{file_name_plain}/landing.csv', 'w')

        takeoff_demand_file.write('ident,latitude,longitude,time,airport\n')
        landing_demand_file.write('ident,latitude,longitude,time,airport\n')

        df = pd.read_csv(f'{PATH_PREFIX}/data/osstate/extracted/{file_name}', compression='gzip')
        # Drop all rows with NaN values
        df = df.dropna()
        df.head()
        # Filter out all rows with geoaltitude between 200 and 800 feet
        df_groundprox = df[(df['geoaltitude'] > 200) & (df['geoaltitude'] < 800)]
        df_groundprox.head()
        # Add an ident column to the dataframe, which is the concatenation of the icao24 and callsign columns
        df_groundprox['ident'] = (df_groundprox['callsign'].str.strip()+'_'+df_groundprox['icao24'].str.strip())
        df_groundprox.head()
        unique_idents = df_groundprox['ident'].unique()
        # print(f'Number of unique idents: {len(unique_idents)}')

        # A flight is considered to be taking off if final altitude is greater than initial altitude
        n_takeoffs = 0
        n_landings = 0

        for ident in unique_idents:
            df_ident = df_groundprox[df_groundprox['ident'] == ident]
            df_ident = df_ident.sort_values(by='lastposupdate')
            initial_altitude = df_ident['geoaltitude'].iloc[0]
            final_altitude = df_ident['geoaltitude'].iloc[-1]
            if final_altitude > initial_altitude:
                # Flight is taking off, we note the time of takeoff and the position where the flight took off
                time_takeoff = df_ident['lastposupdate'].iloc[0]
                time_landing = -1
                lat_prox = df_ident['lat'].iloc[0]
                lon_prox = df_ident['lon'].iloc[0]
                n_takeoffs += 1
                # Find the airport closest to the takeoff position
                airport = ''
                # 1. Narrow down the airports to those within +-1 degree of the takeoff position
                df_airport_prox = df_airport[(df_airport['latitude'] > lat_prox-1) & (df_airport['latitude'] < lat_prox+1)]
                df_airport_prox = df_airport_prox[(df_airport_prox['longitude'] > lon_prox-1) & (df_airport_prox['longitude'] < lon_prox+1)]
                # 2. If there are no airports within the region, return ""
                if len(df_airport_prox) == 0:
                    airport = ''
                elif len(df_airport_prox) == 1:
                    airport = df_airport_prox['ident'].iloc[0]
                else:
                    # 3. Find the closest airport
                    # Create a distance column in the df_airport_prox dataframe
                    df_airport_prox['distance'] = np.sqrt((df_airport_prox['latitude'] - lat_prox)**2 + (df_airport_prox['longitude'] - lon_prox)**2)
                    airport = df_airport_prox['ident'].loc[df_airport_prox['distance'].idxmin()] 
                # Write the takeoff demand to the takeoff_demand_file
                takeoff_demand_file.write(f'{ident},{lat_prox},{lon_prox},{unix_time_to_datetime(time_takeoff)},{airport}\n')
            else:
                time_landing = df_ident['lastposupdate'].iloc[-1]
                time_takeoff = -1
                lat_prox = df_ident['lat'].iloc[-1]
                lon_prox = df_ident['lon'].iloc[-1]
                n_landings += 1
                # Find the airport closest to the takeoff position
                airport = ''
                # 1. Narrow down the airports to those within +-1 degree of the takeoff position
                df_airport_prox = df_airport[(df_airport['latitude'] > lat_prox-1) & (df_airport['latitude'] < lat_prox+1)]
                df_airport_prox = df_airport_prox[(df_airport_prox['longitude'] > lon_prox-1) & (df_airport_prox['longitude'] < lon_prox+1)]
                # 2. If there are no airports within the region, return ""
                if len(df_airport_prox) == 0:
                    airport = ''
                elif len(df_airport_prox) == 1:
                    airport = df_airport_prox['ident'].iloc[0]
                else:
                    # 3. Find the closest airport
                    # Create a distance column in the df_airport_prox dataframe
                    df_airport_prox['distance'] = np.sqrt((df_airport_prox['latitude'] - lat_prox)**2 + (df_airport_prox['longitude'] - lon_prox)**2)
                    airport = df_airport_prox['ident'].loc[df_airport_prox['distance'].idxmin()] 
                # Write the landing demand to the landing_demand_file
                landing_demand_file.write(f'{ident},{lat_prox},{lon_prox},{unix_time_to_datetime(time_landing)},{airport}\n')

        takeoff_demand_file.close()
        landing_demand_file.close()
        print(f'Number of takeoffs: {n_takeoffs}')
        print(f'Number of landings: {n_landings}\n')
                


if __name__ == '__main__':
    # Create f'{PATH_PREFIX}/data/osstate/demand' directory if it does not exist
    create_demand_sheet_directory()

    # Load the airport data
    df_airports = pd.read_csv(f'{PATH_PREFIX}/data/airport-codes-xl.csv')
    print(f'Number of airports: {len(df_airports)}')

    # Get all the .csv.gz file in the extracted directory
    extracted_files = os.listdir(f'{PATH_PREFIX}/data/osstate/extracted')
    extracted_files = [file for file in extracted_files if file.endswith('.csv.gz')]
    print(f'There are {len(extracted_files)} files in the extracted directory')

    # extract_takeoffs_and_landings(extracted_files, df_airports)
    # raise Exception('This script is not yet complete. Please remove this line to continue')

    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count()

    # Divide the extracted_files into equal chunks
    chunk_size = len(extracted_files) // num_processes
    chunks = [extracted_files[i:i+chunk_size] for i in range(0, len(extracted_files), chunk_size)]

    # Create a partial function with the extract_takeoffs_and_landings function as an argument
    partial_extract_takeoffs_and_landings = partial(extract_takeoffs_and_landings, df_airport=df_airports)
    
    # Create a pool of processes and map the partial function with the file chunk as an argument to each chunk of files
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(partial_extract_takeoffs_and_landings, chunks)
