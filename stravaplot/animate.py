from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from numpy import ndarray


def calc_bounds(tracks, lat_pad=0.01, lon_pad=0.01):
    """
    Calculate the latitude/longitude bounds of a collection of activity tracks,
    with some padding.
    :param tracks: collection of track DataFrames
    :param lat_pad: amount to pad latitude boundaries
    :param lon_pad: amount to pad longitude boundaries
    :return: (xlim, ylim) tuple appropriate for matplotlib
    """
    min_lat = min([act['lat'].min() for act in tracks])
    min_lon = min([act['lon'].min() for act in tracks])
    max_lat = max([act['lat'].max() for act in tracks])
    max_lon = max([act['lon'].max() for act in tracks])
    xlim = [min_lon - lon_pad, max_lon + lon_pad]
    ylim = [min_lat - lat_pad, max_lat + lat_pad]
    return xlim, ylim


def init_artists(fig, xlim, ylim, n_tracks):
    """
    Initialize matplotlib artists for composing the animation.
    :param fig: matplotlib Figure
    :param xlim: xlim tuple for matplotlib Axes
    :param ylim: ylim tuple for matplotlib Axes
    :param n_tracks: number of tracks to be animated
    :return: list of artists: (timestamp, "leader" dot, track histories)
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
    for _ in range(n_tracks):
        artists.append(
            ax.plot([], [], color='deepskyblue', lw=0.5, alpha=0.7, animated=True)[0]
        )
    return artists


def init_frame(artists):
    """
    Create initial frame for FuncAnimation animator.
    :param artists: list of artists
    :return: list of artists with initial frame data set
    """
    time_artist, head_artist, *track_artists = artists
    time_artist.set_text('')
    head_artist.set_offsets(ndarray(shape=(0, 2)))
    for artist in track_artists:
        artist.set_data([], [])
    return artists


def gen_time_indices(all_tracks, step_interval):
    """
    Generate time indices for animation, spanning from the earliest track's
    start to the last tracks's end, in the specified step interval (e.g.,
    '15S' for 15 second increments).
    :param all_tracks: list of tracks
    :param step_interval: step interval for generated timestamps
    :return: list of timestamps ('HH:MM:SS' format)
    """
    start = min([r.index.min() for r in all_tracks])
    end = max([r.index.max() for r in all_tracks])
    # Make some fake dates to get pandas to generate a time range, then strip date component
    dt_range = pd.date_range(
        start='2000/01/01T{}'.format(start),
        end='2000/01/01T{}'.format(end),
        freq=step_interval,
    )
    return [dt.strftime('%H:%M:%S') for dt in dt_range]


def get_point(track, time_idx):
    """
    Get GPX track point from a track for a given time index.
    :param track: track DataFrame
    :param time_idx: time index
    :return: GPX track point for time index, or None if no point exists at
    that time
    """
    try:
        return track.loc[time_idx]
    except KeyError:
        return None


def update_artists(artists, tracks, time_idx):
    """
    Update all artists for the given time index.
    :param artists: list of artists
    :param tracks: list of tracks
    :param time_idx: time index for artists to draw
    :return: list of artists
    """
    time_artist, head_artist, *track_artists = artists
    time_artist.set_text(time_idx[:5])
    head_lonlat = []
    for artist, track in zip(track_artists, tracks):
        point = get_point(track, time_idx)
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


def setup_animation(tracks, step_interval):
    """
    Set up data structures and a matplotlib FuncAnimation instance for
    animating a list of track DataFrames.
    :param tracks: list of track DataFrames
    :param step_interval: step interval for animation (e.g., '15S')
    :return: FuncAnimation instance
    """
    # Set up objects and functions for matplotlib FuncAnimation process
    xlim, ylim = calc_bounds(tracks)
    indices = gen_time_indices(tracks, step_interval)
    fig = plt.figure(figsize=(5, 12))
    artists = init_artists(fig, xlim, ylim, len(tracks))
    init = partial(init_frame, artists)
    def frames(): return indices
    update = partial(update_artists, artists, tracks)

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
