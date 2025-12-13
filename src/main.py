
# import necessary modules
from grid import make_grid
from phase import tilt_phase
import numpy as np
import pyvista as pv

# initialized grid
X, Y, dx, dy, mask = make_grid(N=128)

phase = tilt_phase(X, Y, mask, a=5.0)

grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
grid.point_data['phase'] = phase.ravel()

plotter = pv.Plotter()
plotter.add_mesh(grid, scalars='phase')
plotter.view_xy()
plotter.show(screenshot = "tilt_phase.png")