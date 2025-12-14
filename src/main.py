
# import necessary modules
from grid import make_grid
from phase import tilt_phase, defocus_phase, astigmatism_phase
from phase_reconstruct import poisson_reconstruction

import numpy as np
import pyvista as pv

# initialized grid
X, Y, dx, dy, mask = make_grid(N=128)

# phase = tilt_phase(X, Y, mask, a=5.0)
# phase = defocus_phase(X, Y, mask, b=5.0)
phase = astigmatism_phase(X, Y, mask, c=5.0, theta=90)

# finite central difference of change in phase
# d(phase)/dx  → axis 1
sx = (phase[1:-1, 2:] - phase[1:-1, :-2]) / (2 * dx)

# d(phase)/dy  → axis 0
sy = (phase[2:, 1:-1] - phase[:-2, 1:-1]) / (2 * dy)

# print statistics of slopes, temporary debugging info
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

# reconstruct phase from slopes within mask
phase_rec = poisson_reconstruction(sx_full, sy_full, dx, dy, mask)

# scale reconstructed phase to true phase amplitude for better visual comparison
m = mask & np.isfinite(phase_rec) & np.isfinite(phase)
t = phase[m].ravel()
r = phase_rec[m].ravel()
scale_factor = np.dot(t, r) / np.dot(r, r)
print(f"Scale factor applied to reconstructed phase: {scale_factor}")
print(dx**2)
phase_rec *= scale_factor



# calculate RMS error between true and reconstructed phase
# the true test of performance
valid_4_rms = mask & np.isfinite(phase_rec) & np.isfinite(phase)

phase0 = phase[valid_4_rms] - np.nanmean(phase[valid_4_rms])
phase_rec0 = phase_rec[valid_4_rms] - np.nanmean(phase_rec[valid_4_rms])

rms_error = np.sqrt(np.mean((phase0 - phase_rec0)**2))
print(f"RMS error between true and reconstructed phase: {rms_error}")

# build grid + attach data
# mostly for plotting purposes
grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
grid.point_data["sx"] = sx_full.ravel(order="C")
grid.point_data["sy"] = sy_full.ravel(order="C")


# the grouped plot text is tempoary, this was the easiest way to quickly use and surpress
'''
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