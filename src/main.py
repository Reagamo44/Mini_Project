
# import necessary modules
from grid import make_grid
from phase import tilt_phase, defocus_phase

import numpy as np
import pyvista as pv

# initialized grid
X, Y, dx, dy, mask = make_grid(N=128)

phase = defocus_phase(X, Y, mask, b=5.0)
# phase = tilt_phase(X, Y, mask, a=5.0)


# finite central difference of change in phase
# d(phase)/dx  → axis 1
sx = (phase[1:-1, 2:] - phase[1:-1, :-2]) / (2 * dx)

# d(phase)/dy  → axis 0
sy = (phase[2:, 1:-1] - phase[:-2, 1:-1]) / (2 * dy)

h = np.nanmean(sx)
i = np.nanmean(sy)
j = np.nanstd(sx)
k = np.nanstd(sy)

print(f"sx mean: {h}, sy mean: {i}, sx std: {j}, sy std: {k}")

# expand back to full grid size with nans at edges
sx_full = np.full_like(X, np.nan, dtype=float)
sy_full = np.full_like(X, np.nan, dtype=float)


sx_full[1:-1, 1:-1] = sx
sy_full[1:-1, 1:-1] = sy

# mask outside pupil
sx_full[~mask] = np.nan
sy_full[~mask] = np.nan

# build grid + attach data
grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
grid.point_data["sx"] = sx_full.ravel(order="C")
grid.point_data["sy"] = sy_full.ravel(order="C")

# plot (choose "sx" or "sy")
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="sx", nan_opacity=0.0)
plotter.view_xy()
plotter.show(screenshot = "defocus_slope_x.png")

grid.point_data["phase"] = phase.ravel(order="C")
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="phase", nan_opacity=0.0)
plotter.view_xy()
plotter.show(screenshot = "defocus_phase.png")