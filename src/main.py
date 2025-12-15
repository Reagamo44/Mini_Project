
# import necessary modules
from grid import make_grid
from phase import tilt_phase, defocus_phase, astigmatism_phase
from phase_reconstruct import poisson_reconstruction
from metrics import rms_error
from convergence import slopes_from_phase, run_case


import numpy as np
import pyvista as pv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# define parameters
min_max_value = 1
remove_tilt = True

# Select cases and weights for each
tilt_case = True
weight_t = 1

defoc_case = True
weight_defoc = 1

astig_case = False
weight_astig = 1

# Chose sample sizes and initialize outputs
Ns = [40, 80, 160, 320, 640, 1280]
rims = [0, 1, 2, 3, 4] # to test Neumann as center is approached


hs =[]
rmss_by_rim = {rim: [] for rim in rims}

for N in Ns:
    for rim in rims:
        # run each case
        h, rms = run_case(N, min_max_value, tilt_case, defoc_case, astig_case, weight_t, weight_defoc, weight_astig, remove_tilt, rim)

        rmss_by_rim[rim].append(rms)
    
    # add value to storage array
    hs.append(h)

    #print(f"N = {N:4d}  |  h = {h:.5f}  |  RMS = {rms:.4e}")
    print(f"N={N:4d} | h={h:.5f} | " +
          " ".join([f"rim{rim}={rmss_by_rim[rim][-1]:.2e}" for rim in rims]))
    

hs = np.array(hs)
for rim in rims:
    rmss_by_rim[rim] = np.array(rmss_by_rim[rim])
    plt.loglog(hs, rmss_by_rim[rim], '-o', label = f"rim = {rim}")

ref = rmss_by_rim[0][-1] * (hs / hs[-1])**2
plt.loglog(hs, ref, '--', label=r"$O(h^2)$ reference")

plt.xlabel("h")
plt.ylabel("RMS error")
plt.grid(True, which="both")
plt.legend()
plt.savefig("rms_convergence_rim_sweep.png", dpi=300, bbox_inches="tight")
plt.close()






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