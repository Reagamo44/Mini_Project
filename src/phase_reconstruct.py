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

    solve_mask = mask.copy() # pupil area

    # prevent indexing error at edges
    solve_mask[0, :] = solve_mask[-1, :] = False
    solve_mask[:, 0] = solve_mask[:, -1] = False

    pts = np.argwhere(solve_mask)
    n = len(pts)

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

    # assign indices to interior points, indexing from class does not work because looking at circular pupil
    for k, (i, j) in enumerate(pts):
        idx[i, j] = k

    # building Ax = b system
    A = sps.lil_matrix((n, n)) # sparse matrix for efficiency (regular matrix too large and mostly zeros)
    b = np.zeros(n) # still unknown

    # condense constants for readability
    Cy = 1/(dy**2)
    Cx = 1/(dx**2) 


    # defining A matrix and b vector
    for row, (i, j) in enumerate(pts):
        # start with standard diagonal value
        C0 = -2*(Cx + Cy)

        # if neighbor is in pupil, add to matrix, else add to diagonal
        if solve_mask[i + 1, j]:
            A[row, idx[i + 1, j]] = Cx
        else:
            # Neumann boundary condition, adding weight to diagonal (clean assymilation to tilts and warps at edge)
            C0 += Cx

        if solve_mask[i - 1, j]:
            A[row, idx[i - 1, j]] = Cx
        else:
            C0 += Cx

        if solve_mask[i, j + 1]:
            A[row, idx[i, j + 1]] = Cy
        else:
            C0 += Cy

        if solve_mask[i, j - 1]:
            A[row, idx[i, j - 1]] = Cy
        else:
            C0 += Cy
        
        # add diagonal entry
        A[row, row] += C0

        # right hand side entry
        b[row] = rhs[i, j]

    # fix reference point to zero phase (to prevent singularity and make solution unique)
    A[0,:] = 0
    A[0,0] = 1
    b[0]=0

    # solve sparse linear system
    phi_vec = sps.linalg.spsolve(A.tocsr(), b)

    # map solution back to 2D grid
    phase = np.full_like(sx_full, np.nan, dtype=float)
    phase[solve_mask] = phi_vec

    # remove global mean over pupil to fix arbitrary offset
    phase[solve_mask] -= np.nanmean(phase[solve_mask])

    return phase










