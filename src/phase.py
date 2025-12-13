# MAP OUT DISTURBANCES
import numpy as np
import pyvista as pv


def tilt_phase(X, Y, mask, a):
    """
    Create a tilt phase disturbance over the grid defined by X and Y with amplitude a

    Parameters:
    X : 2D array
        X coordinates of the grid
    Y : 2D array
        Y coordinates of the grid
    mask : 2D boolean array
        Mask defining the aperture
    a : float
        Amplitude of the tilt disturbance

    Returns:
    phase : 2D array
        Phase disturbance due to tilt
    """

    # calculate tilt phase disturbance
    phi_true = a * X

    # apply mask to phase disturbance
    phase = np.where(mask, phi_true, np.nan)

    return phase


# def random_phase