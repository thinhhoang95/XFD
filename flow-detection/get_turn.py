import numpy as np
import pandas as pd
from geo.drift_compensation import get_track_drift_rate
from changepy import pelt
from changepy.costs import normal_mean
import matplotlib.pyplot as plt

from typing import TypedDict
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


def get_turning_points(df_ident: pd.DataFrame) -> TurnAndRise:
    # Extract the values from the dataframe
    rlastposupdate = df_ident['lastposupdate'].values - df_ident['lastposupdate'].min()
    hdg = df_ident['heading'].values
    vel = df_ident['velocity'].values / 1000 # m/s -> km/s
    lat = df_ident['lat'].values
    lon = df_ident['lon'].values
    alt = df_ident['geoaltitude'].values

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
        'landed': not flight_not_landed_yet
    }

    result_alt = get_altitude_change_points(rlastposupdate, lat, lon, alt)

    return {
        **result_turn,
        **result_alt
    }

def get_altitude_change_points(rlastposupdate: pd.DataFrame, lat: np.ndarray, lon: np.ndarray, alt:np.ndarray) -> TurnAndRise:
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
        # print(i, dp_time[i], dp_time[i+1])
        if (dp_time[i+1] - dp_time[i]) < 120:
            # print(f'Merging {i} and {i+1}')
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

def plot_changepoints(tr: TurnAndRise, df_ident: pd.DataFrame = None) -> None:
    tp_lat = tr['tp_lat']
    tp_lon = tr['tp_lon']
    tp_alt = tr['tp_alt']
    tp_time = tr['tp_time']
    dp_lat = tr['dp_lat']
    dp_lon = tr['dp_lon']
    dp_alt = tr['dp_alt']
    dp_time = tr['dp_time']

    # Plot the changepoints
    plt.figure(figsize=(6,6))
    if df_ident is not None:
        plt.plot(df_ident['lon'], df_ident['lat'], 'black') # flight path
    plt.plot(tp_lon, tp_lat, 'go', markersize=3) # turning points
    plt.plot(dp_lon, dp_lat, 'ro', markersize=3) # altitude change points

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flight Path with Turning Points (Landed: {})'.format(tr['landed']))