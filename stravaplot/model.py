import csv
import datetime
import gzip
from collections import namedtuple
from pathlib import Path
from typing import IO
from typing import Optional
from typing import Tuple
from typing import Union

import fitparse
import gpxpy
import pandas as pd
from geopy import distance


def load_activities_metadata(data_dir: str, activity_type: Optional[str] = None) -> pd.DataFrame:
    """
    Load metadata from Strava exported activities.csv file.
    :param data_dir: directory with strava exported data (where the
    activities.csv file is located)
    :param activity_type: set to limit the loaded activities to only one
    activity type (e.g., 'Ride'), or set to None to load all activities
    :return: metadata DataFrame
    """

    def type_filter(d):
        return d['Activity Type'].lower() == activity_type.lower() if activity_type else True

    activities_csv = Path(data_dir) / 'activities.csv'
    with activities_csv.open() as f:
        reader = csv.DictReader(f)
        activities_meta = pd.DataFrame([d for d in reader if type_filter(d)])
    activities_meta['Activity Date'] = pd.to_datetime(activities_meta['Activity Date'])
    return activities_meta


def load_gpx(gpx_file: IO) -> pd.DataFrame:
    """
    Load a GPX file and return the activity track as a DataFrame.
    :param gpx_file: GPX file object
    :return: DataFrame object with GPX track points
    """
    gpx = gpxpy.parse(gpx_file)
    data = [
        {'lat': p.latitude, 'lon': p.longitude, 'alt': p.elevation, 'ts': p.time}
        for p in gpx.tracks[0].segments[0].points
    ]
    gpx_df = pd.DataFrame(data=data).set_index('ts')
    return gpx_df


def load_fit(fit_file: IO) -> pd.DataFrame:
    Point = namedtuple('Point', ['ts', 'lat', 'lon', 'alt', 'hr'])
    with fitparse.FitFile(fit_file, data_processor=fitparse.StandardUnitsDataProcessor()) as fit:
        data = []
        for record in fit.get_messages('record'):
            d = record.get_values()
            data.append(Point(
                ts=d.get('timestamp').replace(tzinfo=datetime.timezone.utc),
                lat=d.get('position_lat'),
                lon=d.get('position_long'),
                alt=d.get('enhanced_altitude'),
                hr=d.get('heart_rate')
            ))
        fit_df = pd.DataFrame(data=data, columns=Point._fields).set_index('ts')
    return fit_df


def load_activities(data_dir: str, activities_meta: pd.DataFrame, resample_freq: str) -> pd.DataFrame:
    """
    Load all Strava activities of the given type, resampled to the given frequency.
    :param data_dir: directory with strava exported data
    :param activities_meta: activity metadata DataFrame
    :param resample_freq: frequency for resampling GPX tracks, in pandas-acceptable
    format (e.g., '15S' for 15 seconds)
    :return: activity meta DataFrame with track DataFrame embedded
    """
    # TODO clean all this up; don't modify in place
    tracks = []
    for _, activity in activities_meta.iterrows():
        file_path = Path(data_dir) / activity['Filename']

        if file_path.is_file():
            suffixes = file_path.suffixes
            if '.gpx' in suffixes:
                if suffixes[-1] == '.gz':
                    with gzip.open(file_path) as f:
                        df = load_gpx(f)
                else:
                    with open(file_path) as f:
                        df = load_gpx(f)
            elif '.fit' in suffixes:
                if suffixes[-1] == '.gz':
                    with gzip.open(file_path) as f:
                        df = load_fit(f)
                else:
                    with open(file_path) as f:
                        df = load_fit(f)
            else:
                raise ValueError(f'unknown file type: {file_path}')

            # Strip any points with unknown lat/lon
            df = df[~(df['lat'].isna() | df['lon'].isna())]

            # Resample the timeseries to reduce number of animated frames
            df = df.resample(resample_freq).nearest(limit=1)
            tracks.append(df)

    activities_meta['track'] = tracks
    return activities_meta


def normalize_timestamps(
        activities: pd.DataFrame,
        timezone: Union[str, datetime.tzinfo] = 'America/New_York'
) -> pd.DataFrame:
    """
    Convert timestamps to the given timezone and reformat as 'HH:MM:SS'.
    :param activities: activities DataFrame
    :param timezone: target timezone for all activities' timestamps
    :return: list of activities with normalized timestamps (in-place updates)
    """
    for _, activity in activities.iterrows():
        df = activity['track']
        # Convert activity's timestamp index to local timezone
        df.index = df.index.tz_convert(timezone)
        # Reformat timestamps to strip date
        df.index = df.index.strftime('%H:%M:%S')
    return activities


def filter_activities_by_origin(activities: pd.DataFrame, target: Tuple[float, float], distance_mi: float) -> pd.Series:
    """
    Generator for filtering activities by the lat/lon of the first tracked
    point.
    :param activities: list of activities
    :param target: (lat, lon) tuple for origin filter
    :param distance_mi: origin filter distance in miles
    :return: filtered activities
    """
    for _, activity in activities.iterrows():
        start_pt = activity['track'].iloc[0]
        origin = (start_pt['lat'], start_pt['lon'])
        if distance.distance(origin, target).miles < distance_mi:
            yield activity
