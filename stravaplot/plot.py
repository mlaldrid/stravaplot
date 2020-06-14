from typing import Collection
from typing import Iterable
from typing import Tuple

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import pandas as pd


# TODO add distance-based padding
def calc_bounds(
        tracks: Collection[pd.DataFrame],
        lat_pad: float = 0.01,
        lon_pad: float = 0.01
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
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
    xlim = (min_lon - lon_pad, max_lon + lon_pad)
    ylim = (min_lat - lat_pad, max_lat + lat_pad)
    return xlim, ylim


def draw_track(track: pd.DataFrame, ax: plt.Axes, **kwargs) -> plt.Axes:
    """ Draw a single track on an Axes. """
    patch_args = {
        'facecolor': 'None',
        'edgecolor': 'black',
        'lw': 1.5
    }
    patch_args.update(kwargs)

    verts = [(pt.lon, pt.lat) for pt in track.itertuples()]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * (len(verts) - 1)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, **patch_args)
    ax.add_patch(patch)
    return ax


def draw_composite(tracks: Iterable[pd.DataFrame], ax: plt.Axes, **kwargs) -> plt.Axes:
    """ Draw multiple tracks on an Axes. """
    patch_args = {
        'lw': 0.5,
        'alpha': 0.7
    }
    patch_args.update(kwargs)

    for track in tracks:
        ax = draw_track(track, ax, **patch_args)
    return ax
