import copy
import csv
from functools import partial
from pathlib import Path

from geopy import distance
import gpxpy
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from numpy import ndarray
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
        return d['type'].lower() == activity_type.lower() if activity_type else True

    activities_csv = Path(data_dir) / 'activities.csv'
    with activities_csv.open() as f:
        reader = csv.DictReader(f)
        activities_meta = [d for d in reader if type_filter(d)]
    return activities_meta


def load_gpx(gpx_path):
    """
    Load a GPX file and return the activity track as a DataFrame.
    :param gpx_path: pathlib.Path object for GPX file
    :return: DataFrame object with GPX track points
    """
    with gpx_path.open() as f:
        gpx = gpxpy.parse(f)
        data = [
            {'lat': p.latitude, 'lon': p.longitude, 'alt': p.elevation, 'ts': p.time}
            for p in gpx.tracks[0].segments[0].points
        ]
        gpx_df = pd.DataFrame(data=data).set_index('ts')
    return gpx_df


def load_activities(data_dir, activities_meta, resample_freq):
    """
    Load all Strava activities of the given type, resampled to the given frequency.
    :param data_dir: directory with strava exported data
    :param activities_meta:
    :param resample_freq:
    :return:
    """
    activities = []
    for activity in copy.deepcopy(activities_meta):
        gpx_path = Path(data_dir) / activity['filename']

        if gpx_path.is_file():
            gpx_df = load_gpx(gpx_path)
            # Resample the timeseries to reduce number of animated frames
            gpx_df = gpx_df.resample(resample_freq).nearest(limit=1)
            activity['track'] = gpx_df
            activities.append(activity)

    return activities


def normalize_timestamps(activities, timezone='America/New_York'):
    """

    :param activities:
    :param timezone:
    :return:
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

    :param activities:
    :param target:
    :param distance_mi:
    :return:
    """
    for activity in activities:
        start_pt = activity['track'].iloc[0]
        origin = (start_pt['lat'], start_pt['lon'])
        if distance.distance(origin, target).miles < distance_mi:
            yield activity


def calc_bounds(activity_tracks, lat_pad=0.01, lon_pad=0.01):
    """
    Calculate the latitude/longitude bounds of a collection of activity tracks,
    with some padding.
    :param activity_tracks: collection of activity track DataFrames
    :param lat_pad: amount to pad latitude boundaries
    :param lon_pad: amount to pad longitude boundaries
    :return: (xlim, ylim) tuple appropriate for matplotlib
    """
    min_lat = min([act['lat'].min() for act in activity_tracks])
    min_lon = min([act['lon'].min() for act in activity_tracks])
    max_lat = max([act['lat'].max() for act in activity_tracks])
    max_lon = max([act['lon'].max() for act in activity_tracks])
    xlim = [min_lon - lon_pad, max_lon + lon_pad]
    ylim = [min_lat - lat_pad, max_lat + lat_pad]
    return xlim, ylim


def init_artists(fig, xlim, ylim, n_rides):
    """

    :param fig:
    :param xlim:
    :param ylim:
    :param n_rides:
    :return:
    """
    ax = plt.axes(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    artists = [
        ax.text(
            0.05,
            0.98,
            '',
            verticalalignment='top',
            horizontalalignment='left',
            transform=ax.transAxes,
            color='red',
            fontsize=15,
        ),
        ax.scatter([], [], color='red', s=40, zorder=4, animated=True),
    ]
    for _ in range(n_rides):
        artists.append(
            ax.plot([], [], color='deepskyblue', lw=0.5, alpha=0.7, animated=True)[0]
        )
    return artists


def init_frame(artists):
    """

    :param artists:
    :return:
    """
    time_artist, head_artist, *ride_artists = artists
    time_artist.set_text('')
    head_artist.set_offsets(ndarray(shape=(0, 2)))
    for artist in ride_artists:
        artist.set_data([], [])
    return artists


def gen_time_indices(all_rides, step_interval):
    """

    :param all_rides:
    :param step_interval:
    :return:
    """
    start = min([r.index.min() for r in all_rides])
    end = max([r.index.max() for r in all_rides])
    # Make some fake dates to get pandas to generate a time range, then strip date component
    dt_range = pd.date_range(
        start='2000/01/01T{}'.format(start),
        end='2000/01/01T{}'.format(end),
        freq=step_interval,
    )
    return [dt.strftime('%H:%M:%S') for dt in dt_range]


def get_point(ride, time_idx):
    """

    :param ride:
    :param time_idx:
    :return:
    """
    try:
        return ride.loc[time_idx]
    except KeyError:
        return None


def update_artists(artists, rides, time_idx):
    """

    :param artists:
    :param rides:
    :param time_idx:
    :return:
    """
    time_artist, head_artist, *ride_artists = artists
    time_artist.set_text(time_idx[:5])
    head_lonlat = []
    for artist, ride in zip(ride_artists, rides):
        point = get_point(ride, time_idx)
        if point is not None:
            lon, lat = artist.get_data()
            lon.append(point['lon'])
            lat.append(point['lat'])
            artist.set_data(lon, lat)
            head_lonlat.append((point['lon'], point['lat']))
    if head_lonlat:
        head_artist.set_offsets(head_lonlat)
    else:
        head_artist.set_offsets(ndarray(shape=(0, 2)))  # empty scatter plot
    return artists


def setup_animation(activities, step_interval):
    """

    :param activities:
    :param step_interval:
    :return:
    """
    # Set up objects and functions for matplotlib FuncAnimation process
    xlim, ylim = calc_bounds(activities)
    indices = gen_time_indices(activities, step_interval)
    fig = plt.figure(figsize=(5, 12))
    artists = init_artists(fig, xlim, ylim, len(activities))
    init = partial(init_frame, artists)
    frames = lambda: indices
    update = partial(update_artists, artists, activities)

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=True,
        interval=15,
        repeat=False,
        save_count=len(indices),
    )
    return ani
