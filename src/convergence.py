import numpy as np
import pyvista as pv

from grid import make_grid
from phase import tilt_phase, defocus_phase, astigmatism_phase
from phase_reconstruct import poisson_reconstruction
from metrics import rms_error

def slopes_from_phase(phase, dx, dy):
    # d/dx along axis 1, d/dy along axis 0
    sx = (phase[1:-1, 2:] - phase[1:-1, :-2]) / (2*dx)
    sy = (phase[2:, 1:-1] - phase[:-2, 1:-1]) / (2*dy)

    # expand to full grid with NaNs on outside points
    sx_full = np.full_like(phase, np.nan, dtype=float)
    sy_full = np.full_like(phase, np.nan, dtype=float)
    sx_full[1:-1, 1:-1] = sx
    sy_full[1:-1, 1:-1] = sy

    return sx_full, sy_full

def run_case(N, min_max_value, tilt_case, defoc_case, astig_case, weight_t, weight_defoc, weight_astig, rim = 1, remove_tilt = True):
    # initialized grid
    X, Y, dx, dy, mask = make_grid(N, min_max_value)

    # define each phase
    phase_tilt = tilt_phase(X, Y, mask, a=5.0)
    phase_defoc= defocus_phase(X, Y, mask, b=5.0)
    phase_astig = astigmatism_phase(X, Y, mask, c=5.0, theta=90)

    # initialize total phase
    phase = np.zeros_like(X, dtype=float)

    # combine selected cases and weights of each
    if tilt_case:
        phase += weight_t*phase_tilt
    if defoc_case:
        phase += weight_defoc*phase_defoc
    if astig_case:
        phase += weight_astig*phase_astig
    
    
    sx_full, sy_full = slopes_from_phase(phase, dx, dy)

    # mask outside pupil
    sx_full[~mask] = np.nan
    sy_full[~mask] = np.nan

    # reconstruct phase from slopes within mask
    phase_rec = poisson_reconstruction(sx_full, sy_full, dx, dy, mask)

    rms = rms_error(phase, phase_rec, mask, remove_tilt , rim = rim)
    h = dx # for specific case of dx = dy!!!

    return h, rms
