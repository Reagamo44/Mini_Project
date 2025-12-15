
# import necessary modules
from grid import make_grid
from phase import tilt_phase, defocus_phase, astigmatism_phase
from phase_reconstruct import poisson_reconstruction
from metrics import rms_error
from convergence import slopes_from_phase, run_case

import numpy as np
import pyvista as pv


h, rms = run_case(200, 1.0, True, False, False, 1, 0, 0)

print(f"h = {h}      rms = {rms}")







'''
# build grid + attach data
# mostly for plotting purposes
grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
grid.point_data["sx"] = sx_full.ravel(order="C")
grid.point_data["sy"] = sy_full.ravel(order="C")


# the grouped plot text is tempoary, this was the easiest way to quickly use and surpress

grid.point_data["phase"] = phase.ravel(order="C")
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="phase", nan_opacity=0.0)
plotter.view_xy()
plotter.show()

grid.point_data["phase_rec"] = phase_rec.ravel(order="C")
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="phase_rec", nan_opacity=0.0)
plotter.view_xy()
plotter.show(screenshot="phase_rec_Poisson_Neumann_fixed.png")

grid.point_data["phase_error"] = (phase - phase_rec).ravel(order="C")
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="phase_error", nan_opacity=0.0)
plotter.view_xy()
plotter.show(screenshot="phase_error_Poisson_Neumann_fixed.png")
'''