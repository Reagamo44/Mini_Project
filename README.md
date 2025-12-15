## Phase Reconstruction from Slope Measurements

Numerical reconstruction of a 2D phase field from discrete slope measurements, with error analysis and convergence verification. This project is motivated by wavefront sensing and adaptive optics applications.

---

## Project Overview

Given sampled phase slopes in the x- and y-directions over a masked aperture, this code reconstructs the underlying phase field and evaluates reconstruction accuracy as a function of grid resolution.

Key capabilities include:
- Analytic test phase generation (tilt, defocus, astigmatism)
- Finite-difference slope computation
- Poisson-based phase reconstruction
- RMS error evaluation with tilt removal
- Boundary-effect mitigation via rim exclusion
- Grid refinement and convergence verification

---

## Repository Structure

data/
├── outputs/ # Numerical outputs
├── test_plots/ # Saved figures
│ ├── tilt_phase/
│ ├── defocus_phase/
│ ├── astigmatism_phase/
│ ├── Phase_Reconstruction/
│ ├── rim_ratios.png
│ ├── rms_convergence_first_run.png
│ └── rms_convergence_rim_sweep.png

docs/
└── engineering_log.md # Development notes and design decisions

scripts/
└── plot_results.py # Plotting utilities

src/
├── convergence.py # Grid refinement + convergence analysis
├── grid.py # Grid and circular mask generation
├── main.py # Entry point for running simulations
├── metrics.py # RMS error, tilt removal, rim handling
├── noise.py # Optional noise models
├── phase.py # Analytic test phases
└── phase_reconstruct.py # Poisson phase reconstruction

tests/ # (Optional) test cases


---

## Method Summary

- **Slopes:** Second-order central finite differences  
- **Reconstruction:** Discrete Poisson solver on masked interior domain  
- **Error Metric:** RMS phase error after best-fit tilt removal  
- **Rim Exclusion:** Mask erosion to suppress boundary artifacts  
- **Verification:** Log–log RMS convergence plots with reference O(h)

Phase is reconstructed up to piston and tilt; tilt is removed prior to error evaluation.

---

## Key Results

- RMS reconstruction error decreases with grid refinement
- Observed convergence follows expected numerical order
- Boundary effects are significantly reduced with rim exclusion
- Poisson reconstruction provides stable global solutions

Representative results are saved in `data/test_plots/`.

---

## How to Run

From the repository root:
```bash
python src/main.py