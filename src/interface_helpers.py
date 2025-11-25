import pandas as pd
import datetime
import numpy as np


def parse_source_coords(source_coordinates):
    size = np.shape(source_coordinates)
    if len(size) == 1:
        if size[0] == 3:
            source_coordinates = np.array(
                [source_coordinates]
            )  # now a nested array- C++ code expects this format
        else:
            print(
                "[fGP] Error: source_coordinates must be a 3-element array, e.g. [x0, y0, z0]."
            )
            exit(-1)
    else:
        if size[0] == 1 and size[1] == 3:
            return source_coordinates
        elif size[0] > 1 and size[1] == 3:
            raise (
                NotImplementedError(
                    "[fGP] Error: Multi-source currently isn't implemented. Only provide coordinates for a single source, e.g. [x0, y0, z0] or [[x0, y0, z0]]."
                )
            )

    return source_coordinates


def ensure_utc(dt):
    """
    Ensures the input datetime is timezone-aware and in UTC.
    Converts to UTC if it's in a different timezone.
    """
    ts = pd.Timestamp(dt)

    if ts.tz is None:
        raise ValueError(
            f"[FastGaussianPuff] Naive datetime detected: {dt}. Please provide a timezone-aware datetime."
        )

    if ts.tz != datetime.timezone.utc:
        ts = ts.tz_convert("UTC")

    return ts
