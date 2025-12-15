# ERRORS AND MASK EDITING

import numpy as np
'''
Compute RMS error between true and reconstructed phase inside the pupil.

    IMPORTANT:
    If remove_tilt=True, this metric removes piston and tip/tilt (best-fit plane)
    before computing RMS. This is required because slope-based reconstruction
    cannot uniquely recover these modes.

    Parameters
    ----------
    true_phase : 2D array
        Analytic or reference phase
    rec_phase : 2D array
        Reconstructed phase from Poisson solver
    mask : 2D boolean array
        Pupil mask
    remove_tilt : bool
        If True, remove best-fit plane (piston + tip/tilt)
    rim : int
        Number of pixels to exclude near pupil boundary
        (to avoid boundary discretization effects)

    Returns
    -------
    rms : float
        RMS phase error (unitless)

'''

def rms_error(true_phase, rec_phase, mask, remove_tilt=True, rim = 0):
    
    # copy mask to avoid modifying main
    m_copy = mask.copy()

    # erode made inward to avoid boundaries
    for _ in range(rim):
        m_copy = (m_copy & np.roll(m_copy, 1, 0) & np.roll(m_copy, -1, 0) & 
                  np.roll(m_copy, 1, 1) & np.roll(m_copy, -1, 1))
        
    # remove outermost boundaries
    m_copy[0, :] = m_copy[-1, :] = False
    m_copy[:, 0] = m_copy[:, -1] = False

    # valid if in the cnew eroded masks, to be finite in both arrays
    valid = m_copy & np.isfinite(true_phase) & np.isfinite(rec_phase)
    
    # work on copies to preserve inputs
    tp = true_phase.copy()
    rp = rec_phase.copy()

    # Remove piston and tip/tilt by fitting a best fit plane
    # takes form z = a*x + b*y + c
    if remove_tilt: 

        yy, xx = np.indices(tp.shape) # pixel coordinate grids

        # matrix for least-squares plane fit
        A  = np.column_stack([xx[valid], yy[valid], np.ones(np.count_nonzero(valid))])
        
        # fit plane to true phase
        atp, btp, ctp = np.linalg.lstsq(A, tp[valid], rcond=None)[0]
        arp, brp, crp = np.linalg.lstsq(A, rp[valid], rcond=None)[0]


        # subtract best-fit plane from both phases
        tp[valid] -= (atp * xx[valid] + btp * yy[valid] + ctp)
        rp[valid] -= (arp * xx[valid] + brp * yy[valid] + crp)

    # remove piston from both phases
    else:
        tp[valid] -= np.mean(tp[valid])
        rp[valid] -= np.mean(rp[valid])

    # compute rms error
    return float(np.sqrt(np.mean((tp[valid] - rp[valid])**2)))