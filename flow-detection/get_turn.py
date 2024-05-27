import numpy as np
import pandas as pd
from geo.drift_compensation import get_track_drift_rate
from changepy import pelt
from changepy.costs import normal_mean
import matplotlib.pyplot as plt

from typing import TypedDict

import zarr

class TurnAndRise(TypedDict):
    # Direction change
    tp_time: np.ndarray
    tp_lat: np.ndarray
    tp_lon: np.ndarray
    tp_alt: np.ndarray
    landed: bool # 1 if the changepoint is the moment the aircraft landed (alt < 500)
    
    # Altitude change
    dp_time: np.ndarray
    dp_lat: np.ndarray
    dp_lon: np.ndarray
    dp_alt: np.ndarray

    # Flight identification
    ident: str


def get_turning_points(df_ident: pd.DataFrame) -> TurnAndRise:
    """
    Detects turning points in a flight trajectory based on the provided dataframe. Notice that the last moment for not yet landed flight is NOT considered a turning point.

    Args:
        df_ident (pd.DataFrame): The dataframe of a single identified flight.

    Returns:
        dict: A dictionary containing the turning points and altitude change points.

    Raises:
        None

    """
    # Extract the values from the dataframe
    rlastposupdate = df_ident['lastposupdate'].values # - df_ident['lastposupdate'].min()
    hdg = df_ident['heading'].values
    vel = df_ident['velocity'].values / 1000 # m/s -> km/s
    lat = df_ident['lat'].values
    lon = df_ident['lon'].values
    alt = df_ident['geoaltitude'].values
    ident = df_ident['ident'].values[0]

    # Detection of turning points
    # Compute the drift compensation
    track_drift = np.zeros_like(hdg)
    cumul_drift = 0
    hdg_compensated = np.zeros_like(hdg)
    for i in range(1, len(hdg)):
        # We will use the last time's value to compensate the drift for this time
        track_drift[i] = get_track_drift_rate(lat[i-1], lon[i-1], hdg[i-1]) * vel[i-1] * (rlastposupdate[i] - rlastposupdate[i-1])
        cumul_drift += track_drift[i]
        hdg_compensated[i] = (hdg[i] - cumul_drift) % 360

    # Get the changepoints
    changepoints = pelt(normal_mean(hdg_compensated, 1), len(hdg_compensated))

    # Write down the turning points
    tp_lat = []
    tp_lon = []
    tp_time = []
    tp_alt = []

    flight_not_landed_yet = True

    # One final changepoint at the end of the flight or when the aircraft lands
    landed_at = np.where(alt < 500)[0]
    if len(landed_at) > 0:
        changepoints = np.append(changepoints, landed_at[0])

        # Delete all the changepoints after the aircraft landed
        changepoints = changepoints[changepoints <= landed_at[0]] 
        flight_not_landed_yet = False
    else:
        changepoints = np.append(changepoints, len(hdg_compensated)-1)
        flight_not_landed_yet = True

    for i in range(len(changepoints)-1):
        tp_lat.append(lat[changepoints[i]])
        tp_lon.append(lon[changepoints[i]])
        tp_time.append(rlastposupdate[changepoints[i]])
        tp_alt.append(alt[changepoints[i]])
    
    # Merge changepoints that are too close to each other
    i = 0
    while i < len(tp_lat)-1:
        # print(i, tp_time[i], tp_time[i+1])
        if (tp_time[i+1] - tp_time[i]) < 60:
            # print(f'Merging {i} and {i+1}')
            tp_lat[i] = (tp_lat[i] + tp_lat[i+1]) / 2
            tp_lon[i] = (tp_lon[i] + tp_lon[i+1]) / 2
            tp_time[i] = (tp_time[i] + tp_time[i+1]) / 2
            tp_alt[i] = (tp_alt[i] + tp_alt[i+1]) / 2
            tp_lat.pop(i+1)
            tp_lon.pop(i+1)
            tp_time.pop(i+1)
            tp_alt.pop(i+1)
        else:
            i += 1

    result_turn = {
        'tp_time': np.array(tp_time),
        'tp_lat': np.array(tp_lat),
        'tp_lon': np.array(tp_lon),
        'tp_alt': np.array(tp_alt),
        'landed': not flight_not_landed_yet,
        'ident': ident
    }

    result_alt = get_altitude_change_points(rlastposupdate, lat, lon, alt)

    return {
        **result_turn,
        **result_alt
    }

def get_altitude_change_points(rlastposupdate: pd.DataFrame, lat: np.ndarray, lon: np.ndarray, alt:np.ndarray) -> TurnAndRise:
    """
    Get altitude change points based on the provided data.

    Args:
        rlastposupdate (pd.DataFrame): DataFrame containing the time information.
        lat (np.ndarray): Array of latitude values.
        lon (np.ndarray): Array of longitude values.
        alt (np.ndarray): Array of altitude values.

    Returns:
        dict: A dictionary containing the altitude change points with keys 'dp_time', 'dp_lat', 'dp_lon', and 'dp_alt'.
    """

    dp_time = []
    dp_lat = []
    dp_lon = []
    dp_alt = []

    # Get the changepoints
    changepoints = pelt(normal_mean(alt, 1000), len(alt))

    for i in range(len(changepoints)-1):
        dp_time.append(rlastposupdate[changepoints[i]])
        dp_lat.append(lat[changepoints[i]])
        dp_lon.append(lon[changepoints[i]])
        dp_alt.append(alt[changepoints[i]])

    # Merge changepoints that are too close to each other
    i = 0
    while i < len(dp_lat)-1:
        if (dp_time[i+1] - dp_time[i]) < 120:
            dp_lat[i] = (dp_lat[i] + dp_lat[i+1]) / 2
            dp_lon[i] = (dp_lon[i] + dp_lon[i+1]) / 2
            dp_time[i] = (dp_time[i] + dp_time[i+1]) / 2
            dp_alt[i] = (dp_alt[i] + dp_alt[i+1]) / 2
            dp_lat.pop(i+1)
            dp_lon.pop(i+1)
            dp_time.pop(i+1)
            dp_alt.pop(i+1)
        else:
            i += 1

    return {
        'dp_time': np.array(dp_time),
        'dp_lat': np.array(dp_lat),
        'dp_lon': np.array(dp_lon),
        'dp_alt': np.array(dp_alt)
    }

def plot_changepoints(tr: TurnAndRise, df: pd.DataFrame = None, ident:str = None) -> None:
    tp_lat = tr['tp_lat']
    tp_lon = tr['tp_lon']
    tp_alt = tr['tp_alt']
    tp_time = tr['tp_time']
    dp_lat = tr['dp_lat']
    dp_lon = tr['dp_lon']
    dp_alt = tr['dp_alt']
    dp_time = tr['dp_time']

    # Check if the dataframe has the required columns
    if 'ident' not in df.columns:
        df['ident'] = (df['callsign'].str.strip()+'_'+df['icao24'].str.strip())

    # If ident is specified, filter the dataframe for the specific ident
    if ident is not None:
        df_ident = df[df['ident'] == ident]
    else:
        df_ident = df 
    
    # Plot the changepoints
    plt.figure(figsize=(6,6))
    if df_ident is not None:
        plt.plot(df_ident['lon'], df_ident['lat'], 'black') # flight path
    plt.plot(tp_lon, tp_lat, 'go', markersize=3) # turning points
    plt.plot(dp_lon, dp_lat, 'rx', markersize=3) # altitude change points

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flight Path with Turning Points (Landed: {})'.format(tr['landed']))

def write_turnandrise_to_zarr(tr: TurnAndRise,  zarr_path: str) -> None:
    zarr_group = zarr.open(zarr_path, mode='w')

    # Save each item in the dictionary to the ZARR group
    for key, value in tr.items():
        if isinstance(value, np.ndarray):
            # Store numpy arrays directly
            zarr_group.create_dataset(key, data=value)
        elif isinstance(value, dict):
            # Store dictionaries as sub-groups with attributes
            sub_group = zarr_group.create_group(key)
            # print('key:', key, 'value:', value)
            for sub_key, sub_value in value.items():
                # print('sub_key:', sub_key, 'sub_value:', sub_value)
                sub_group.attrs[sub_key] = sub_value
        else:
            # Store other types of data as attributes
            # print('key:', key, 'value:', value)
            zarr_group.attrs[key] = value

def load_turnandrise_from_zarr(zarr_path: str) -> TurnAndRise:
    # Open the ZARR file in read mode
    zarr_group = zarr.open(zarr_path, mode='r')

    # Load the data into a dictionary
    loaded_dict = {}
    for key in zarr_group:
        # print('Key:', key)
        if isinstance(zarr_group[key], zarr.core.Array):
            # If it's an array, load it as a numpy array
            loaded_dict[key] = zarr_group[key][:]
            #print('Key:', key, 'Value:', loaded_dict[key])
        elif isinstance(zarr_group[key], zarr.hierarchy.Group):
            # If it's a group, load its attributes into a dictionary
            loaded_dict[key] = dict(zarr_group[key].attrs)
            #print('Key:', key, 'Value:', loaded_dict[key])
        
    # Load the attrs 
    for key in zarr_group.attrs:
        loaded_dict[key] = zarr_group.attrs[key]

    return loaded_dict