import copy
import csv
from collections import namedtuple

import logging
from pathlib import Path

import fitparse
from geopy import distance
import gpxpy
import pandas as pd


def load_activities_metadata(data_dir, activity_type=None):
    """
    Load metadata from Strava exported activities.csv file.
    :param data_dir: directory with strava exported data (where the
    activities.csv file is located)
    :param activity_type: set to limit the loaded activities to only one
    activity type (e.g., 'Ride'), or set to None to load all activities
    :return: list of metadata dicts
    """

    def type_filter(d):
        return d['Activity Type'].lower() == activity_type.lower() if activity_type else True

    activities_csv = Path(data_dir) / 'activities.csv'
    with activities_csv.open() as f:
        reader = csv.DictReader(f)
        activities_meta = [d for d in reader if type_filter(d)]
    return activities_meta


def load_gpx(gpx_file):
    """
    Load a GPX file and return the activity track as a DataFrame.
    :param gpx_path: pathlib.Path object for GPX file
    :return: DataFrame object with GPX track points
    """
    gpx = gpxpy.parse(gpx_file)
    data = [
        {'lat': p.latitude, 'lon': p.longitude, 'alt': p.elevation, 'ts': p.time}
        for p in gpx.tracks[0].segments[0].points
    ]
    gpx_df = pd.DataFrame(data=data).set_index('ts')
    return gpx_df


def load_fit(fit_file):
    Point = namedtuple('Point', ['ts', 'lat', 'lon', 'alt', 'hr'])
    with fitparse.FitFile(fit_file, data_processor=fitparse.StandardUnitsDataProcessor()) as fit:
        data = []
        for record in fit.get_messages('record'):
            d = record.get_values()
            data.append(Point(
                ts=d.get('timestamp'),
                lat=d.get('position_lat'),
                lon=d.get('position_long'),
                alt=d.get('enhanced_altitude'),
                hr=d.get('heart_rate')
            ))
        fit_df = pd.DataFrame(data=data, columns=Point._fields).set_index('ts')
    return fit_df


def load_activities(data_dir, activities_meta, resample_freq):
    """
    Load all Strava activities of the given type, resampled to the given frequency.
    :param data_dir: directory with strava exported data
    :param activities_meta: list of activity metadata dicts
    :param resample_freq: frequency for resampling GPX tracks, in pandas-acceptable
    format (e.g., '15S' for 15 seconds)
    :return: list of activities (metadata dict + track DataFrame)
    """
    activities = []
    for activity in copy.deepcopy(activities_meta):
        gpx_path = Path(data_dir) / activity['Filename']

        if gpx_path.is_file():
            try:
                gpx_df = load_gpx(gpx_path)
            except:
                logging.exception(gpx_path)
                raise
            # Resample the timeseries to reduce number of animated frames
            gpx_df = gpx_df.resample(resample_freq).nearest(limit=1)
            activity['track'] = gpx_df
            activities.append(activity)

    return activities


def normalize_timestamps(activities, timezone='America/New_York'):
    """
    Convert timestamps to the given timezone and reformat as 'HH:MM:SS'.
    :param activities: list of activities
    :param timezone: target timezone for all activities' timestamps
    :return: list of activities with normalized timestamps (in-place updates)
    """
    for activity in activities:
        df = activity['track']
        # Convert activity's timestamp index to local timezone
        df.index = df.index.tz_convert(timezone)
        # Reformat timestamps to strip date
        df.index = df.index.strftime('%H:%M:%S')
    return activities


def filter_activities_by_origin(activities, target, distance_mi):
    """
    Generator for filtering activities by the lat/lon of the first tracked
    point.
    :param activities: list of activities
    :param target: (lat, lon) tuple for origin filter
    :param distance_mi: origin filter distance in miles
    :return: filtered activities
    """
    for activity in activities:
        start_pt = activity['track'].iloc[0]
        origin = (start_pt['lat'], start_pt['lon'])
        if distance.distance(origin, target).miles < distance_mi:
            yield activity
