# USE NUMERICAL METHOD TO RECONSTRUCT PHASES TO BETTER ACCURACY

import numpy as np
import scipy.sparse as sps

def poisson_reconstruction(sx_full, sy_full, dx, dy, mask):
    """
    Reconstruct phase from slope measurements using Poisson equation approach

    Parameters:
    sx : 2D array
        Slope measurements in x direction
    sy : 2D array
        Slope measurements in y direction
    mask : 2D boolean array
        Mask defining the aperture

    Returns:
    phase_reconstructed : 2D array
        Reconstructed phase
    """

    # interior = mask.copy() # pupil area

    # prevent indexing error at edges
    #interior[0, :] = interior[-1, :] = False 
    # interior[:, 0] = interior[:, -1] = False


    interior = mask.copy() # pupil area

    # interior is surrounding four points are in the mask (separates boundaries)
    interior &= mask & np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0) & np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1)

    # sets NaN to zero for prevent errors
    sx = np.nan_to_num(sx_full, nan=0.0)
    sy = np.nan_to_num(sy_full, nan=0.0)

    # central differencing
    ds_dx = (sx[2:, 1:-1] - sx[:-2, 1:-1]) / (2 * dx)
    ds_dy = (sy[1:-1, 2:] - sy[1:-1, :-2]) / (2 * dy)

    # fill array with interior values
    rhs = np.zeros_like(sx)
    rhs[1:-1, 1:-1] = ds_dx + ds_dy

    # sets all boundary values outside of pupil to -1 (mostly for later checks and distinguishing boundaries of pupil)
    idx = -np.ones(mask.shape, dtype=int)
    pts = np.argwhere(interior)

    # assign indices to interior points, indexing from class does not work because looking at circular pupil
    for k, (i, j) in enumerate(pts):
        idx[i, j] = k

    # building Ax = b system
    n = len(pts)
    A = sps.lil_matrix((n, n)) # sparse matrix for efficiency (regular matrix too large and mostly zeros)
    b = np.zeros(n) # still unknown

    # condense constants for readability
    Cy = 1/(dy**2)
    Cx = 1/(dx**2) 
    C0 = -2*(Cx + Cy)


    # defining A matrix and b vector
    for row, (i, j) in enumerate(pts):
        A[row, row] = C0

        A[row, idx[i + 1, j]] = Cx
        A[row, idx[i - 1, j]] = Cx
        A[row, idx[i, j + 1]] = Cy
        A[row, idx[i, j - 1]] = Cy

        b[row] = rhs[i, j]

        

    phi_vec = sps.linalg.spsolve(A.tocsr(), b)

    phase = np.full_like(sx_full, np.nan, dtype=float)
    phase[interior] = phi_vec

    phase[interior] -= np.nanmean(phase[interior])
    
    return phase










