# DEFINE GRID FUNCTION

import numpy as np


def make_grid(N, min_max_value):
    """
    Create a square grid over [-min_max_value, min_max_value] x [-min_max_value, min_max_value] and circular pupil mask

    """

    # empty grid from min to mask with N points in between
    x = np.linspace(-min_max_value, min_max_value, N)

    # makes X and Y coordinates for each index
    X, Y = np.meshgrid(x, x)

    # grid spacing
    dx = abs(2*min_max_value/(N-1))
    dy = dx

    # creates circular pupil mask, defines whether point is in the aperture
    mask = X**2 + Y**2 <= min_max_value**2

    # return all values for main program to use
    return X, Y, dx, dy, mask
    








