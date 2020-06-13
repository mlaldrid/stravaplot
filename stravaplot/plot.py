from typing import Collection
from typing import Tuple

from pandas import DataFrame


# TODO add distance-based padding
def calc_bounds(
        tracks: Collection[DataFrame],
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
