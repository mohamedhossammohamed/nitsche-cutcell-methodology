# Bounded Nitsche Stabilization for Cut Finite Element Methods

Numerical study of stabilization-parameter formulas for symmetric Nitsche-type
unfitted (cut) finite element methods under degenerate cut geometries
("sliver cuts"), where a vanishing volume fraction of a background element is
occupied by the physical domain while the induced boundary measure stays
bounded below.

The classical volume-scaled penalty $\lambda_T = C\,\rho(T)\,h_T^{-1}$ with
$\rho(T) = |T \cap \partial\Omega| / |T \cap \Omega|$ diverges as
$\rho(T) \to \infty$, inflating the stiffness-matrix condition number and
destroying discretization accuracy through round-off. This repository
implements a preregistered benchmark protocol comparing bounded candidate
formulas $\Phi(\rho, h_T, k, \kappa, a_T)$ against that baseline, against
cell aggregation, and against ghost-penalty stabilization, under explicit
falsification criteria.

## Layout

- `nitschecut/geometry.py` — level-set descriptions (circle, ellipse,
  superellipse) with analytic gradients, Hessians, and curvature.
- `nitschecut/quadrature.py` — Gauss rules on simplices and intervals.
- `nitschecut/cutting.py` — exact clipping of background elements against
  level-set regions; machine-precision area/perimeter via Green's theorem;
  adaptive curved-boundary quadrature.
- `nitschecut/mesh.py` — structured triangular/tetrahedral background meshes
  with P1/P2 degrees of freedom.
- `nitschecut/formulas.py` — candidate stabilization formula registry.
- `nitschecut/nitsche.py` — unfitted symmetric Nitsche assembler.
- `nitschecut/solvers.py`, `nitschecut/spectra.py` — solver trio and
  eigenvalue/conditioning measures.
- `nitschecut/errors.py` — energy and L2 error evaluation on cut geometry.
- `nitschecut/bench.py` — benchmark-tier drivers writing Parquet result logs.
- `nitschecut/run.py` — single fixed experiment entry point.
- `tests/` — verification suite (analytic cut-area identities, polynomial
  exactness of quadrature, scikit-fem oracle agreement, benign-cut
  convergence rates).

## Reproducibility

Every run records mesh parameters, cut configuration, formula, solver,
random seeds, and package versions into Parquet tables under `results/`;
all manuscript figures regenerate from those tables alone.
