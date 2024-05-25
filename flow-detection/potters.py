import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_df(df: pd.DataFrame) -> None:
    """Plot the ADS-B DataFrame

    Args:
        df (pd.DataFrame): Dataframe with latitude and longitude columns
    """
    plt.figure(figsize=(6,6))
    plt.subplot(2, 2, 1)
    plt.plot(df['lon'], df['lat'])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flight Path')

    plt.subplot(2, 2, 2)
    plt.plot(df['lastposupdate'], df['geoaltitude'])
    plt.xlabel('Time')
    plt.ylabel('Altitude')
    plt.title('Altitude vs Time')

    plt.subplot(2, 2, 3)
    plt.plot(df['lastposupdate'], df['heading'])
    plt.xlabel('Time')
    plt.ylabel('Heading')
    plt.title('Heading vs Time')

    plt.subplot(2, 2, 4)
    plt.plot(df['lastposupdate'], df['velocity'])
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Time')

    plt.tight_layout()

    plt.show()

