
# import necessary modules
from grid import make_grid
from phase import tilt_phase
import numpy as np
import pyvista as pv

# initialized grid
X, Y, dx, dy, mask = make_grid(N=128)

phase = tilt_phase(X, Y, mask, a=5.0)


# finite central difference of change in phase
sx = (phase[2:, 1:-1] - phase[:-2, 1:-1]) / (2 * dx)
sy = (phase[1:-1, 2:] - phase[1:-1, :-2]) / (2 * dy)

h = np.nanmean(sx)
i = np.nanmean(sy)
j = np.nanstd(sx)
k = np.nanstd(sy)

print(h, i, j, k)