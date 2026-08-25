# PREREGISTRATION — Bounded Nitsche Stabilization under Sliver Cuts

Frozen before collection of any sweep data. Any change after first data is a new
pre-registration and must be logged as such in README.md run history. The SHA-256
of this file at freeze time is stored in `PREREGISTRATION.hash.txt` and quoted in
every report.

## 1. Research question (fixed)

Does there exist a closed-form stabilization parameter
$\Phi(\rho, h_T, k, \kappa_{\partial\Omega}, a_T)$ for symmetric Nitsche-type
unfitted FEM such that (a) $\lambda_T$ stays bounded as $|T\cap\Omega|\to 0$
at fixed $|T\cap\partial\Omega|$; (b) discrete coercivity
$\gamma_h \ge \gamma_0 > 0$ independent of cut fraction $\varepsilon$ and $h$;
(c) optimal energy-norm rate $O(h^k)$ is preserved; (d)
$\kappa(A_h)=O(h^{-2})$ independent of $\varepsilon$ — and does it compete
with cell aggregation (AgFEM-style) and ghost-penalty stabilization?

## 2. Candidate formulas (frozen; no additions)

All formulas share $\lambda_T = C_k\, h_T^{-1}\,\Psi$, $C_k = 4(k+1)^2$,
$h_T$ = element diameter, $\rho=\rho(T)=|T\cap\partial\Omega|/|T\cap\Omega|$,
$a_T$ = aspect ratio of the bounding box of $T\cap\Omega$,
$\kappa$ = mean $|\kappa_{\partial\Omega}|$ on $T\cap\partial\Omega$.

| id | name | $\Psi$ | free params |
|----|------|--------|-------------|
| F1 | baseline (control) | $\rho$ | — |
| F2a–F2d | hard clip | $\min(\rho,\rho_{cap})$ | $\rho_{cap}\in\{10,50,100,500\}$ |
| F3 | aspect-ratio clip | $\min(\rho,\;10\,a_T)$ | — |
| F4a–F4d | harmonic blend | $\rho\rho_{cap}/(\rho+\rho_{cap})$ | $\rho_{cap}\in\{10,50,100,500\}$ |
| F5 | logarithmic damping | $1+\ln(1+\rho)$ | — |
| F6a–F6b | curvature-augmented | $\rho\,(1+\beta h_T\kappa)$ | $\beta\in\{0.5,1.0\}$ |
| F7 | aggregation hybrid | F4c ($\rho_{cap}{=}100$) when $\varepsilon_T\ge\varepsilon_c$; else aggregate | $\varepsilon_c=10^{-3}$ |
| F8 | fitted harmonic | F4 family with $(C_k,\rho_{cap})$ fitted | fit on T2-train only |

Aggregation (for F7 and the H2 comparator): any active element with
$\varepsilon_T=|T\cap\Omega|/|T|<\varepsilon_c$ is aggregated into the
neighboring element with the largest $|T'\cap\Omega|$ sharing an edge;
its degrees of freedom are constrained piecewise-linearly onto the aggregate
(AGFE-style vertex aggregation), and Nitsche terms are assembled on the
aggregate geometry.

Ghost penalty (for H3):
$s_h(u,v)=\sigma_{GP}\sum_{F\in\mathcal{F}_{ghost}} h_F^{2k+1-d}
\int_F [\partial_n u][\partial_n v]\,dS$,
$\mathcal{F}_{ghost}$ = interior faces of elements edge-adjacent to cut
elements; $\sigma_{GP}\in\{0.1,0.25,0.5,1,2\}$, chosen once as the smallest
value passing the benign-cut ($\varepsilon=0.5$) validation of §4, then frozen.

F8 fitting: grid-plus-refine least squares over $(C_k,\rho_{cap})$ minimizing
the median of $\log_{10}\kappa(A_h)$ over T2-train subject to the T1
$\varepsilon=0.5$ rate gate; refit never after T2-test/T3/T4 evaluation.

## 3. Benchmark tiers, parameter ranges, seeds (frozen)

Manufactured solution everywhere $u=\sin(\pi x)\cos(\pi y)$ (T4:
$\sin\pi x\sin\pi y\sin\pi z$); $f=-\Delta u$; Nitsche RHS consistent.

- **T1 controlled sliver**: unit square, structured triangles (each square
  cell split lower-left→upper-right), circle radius $r=0.25$. Target cell
  $T_0$ = cell with lower corner $(3h,4h)$; disk centre
  $(3.5h,\;4h+s-r)$ where cap height $s$ solves
  $A_{cap}(s;r)=r^2\arccos\!\frac{r-s}{r}-(r-s)\sqrt{2rs-s^2}=\varepsilon h^2$
  by bisection (60 iterations, relative tol $10^{-15}$).
  $\varepsilon\in\{5{\cdot}10^{-1},10^{-1},10^{-2},10^{-3},10^{-4},10^{-5},10^{-6}\}$;
  $n\in\{8,16,32,64,128\}$ ($h=1/n$); $k\in\{1,2\}$.
- **T2 random ensemble**: $N=200$ realisations per mesh $n\in\{16,32,64\}$;
  100 disks ($r\sim U[0.30,0.40]$) + 100 superellipses exponent 6
  (axes $\sim U[0.28,0.38]$), centres $\sim U[0.35,0.65]^2$,
  RNG `numpy.random.default_rng(20260825 + repl)`, repl-indexed.
  Split: train = repl $<140$, test = repl $\ge140$ (frozen by index).
- **T3 curvature sweep**: ellipses, aspect $a/b\in\{1,2,5,10\}$ with
  $b$ chosen so $\pi ab=\pi\cdot0.35^2$; centre $(0.5,0.5)$;
  $n\in\{16,32,64\}$; $k\in\{1,2\}$.
- **T4 3D generalization**: sphere $R=0.3$, centre $(0.5,0.5,0.5)^3$,
  structured tetrahedral cube mesh, $n\in\{6,9,12\}$, $k=1$,
  formulas $\{$F1, F4c, F5, F7$\}$; near-tangential sliver induced by the
  T1 bisection construction applied to a face-adjacent cell column.
- **T5 sign-changing coefficient**: T1 geometry at
  $\varepsilon\in\{0.5,10^{-2},10^{-4}\}$; piecewise
  $\kappa_+=1$ (inside disk), $\kappa_-\in\{-0.9,-1.1\}$ (outside);
  $n\in\{16,32,64\}$, $k=1$.

## 4. Measurement protocol (frozen)

Per configuration: assemble $A_h$ (symmetrized; asymmetry norm logged),
solve with all three solvers to rtol $10^{-10}$ (max 5000 iters):
S1 sparse LU (`scipy.sparse.linalg.splu`), S2 CG + Jacobi, S3 CG + AMG
(pyamg smoothed aggregation). Spectra via ARPACK `eigsh` shift-invert
($\sigma=0$, tol $10^{-10}$, maxiter 500): $\gamma_h=\lambda_{\min}$,
$\lambda_{\max}$, $\kappa=\lambda_{\max}/\lambda_{\min}$;
`ArpackNoConvergence` rows are flagged explicitly with NaN, never silently
dropped. Mass-normalized $\lambda_{\min}(A,M)$ logged alongside (deviation D2).
`svds` cross-check on a seeded random 10% subsample of T1 (rng seed 12345).
Errors: $L^2(\Omega)$ and energy norms on exact-cut quadrature of order
$\max(2k+4,10)$. Convergence slope: OLS on $\log$-$\log$ over the five mesh
points with $R^2$; rates require $R^2\ge0.98$ and $|p-k|\le0.1k$.
Wall-clock assembly/solve times logged. Every row records: tier, formula id,
$\varepsilon$, $n$, $k$, solver id, seeds, package versions, timestamp.

## 5. Falsification gates (numeric; see spec/falsification-gates.md for detail)

- **H1** falsified iff any bounded-$\Psi$ candidate (F2–F6) fails the
  $O(h^k)$ energy-rate gate on T1 at any tested $\varepsilon$, or records
  $\gamma_h<10^{-8}$ at $n=128$ for any $\varepsilon\ge10^{-6}$.
- **H2** falsified iff some closed-form candidate matches F7-aggregation
  conditioning within a factor of 5 at every $\varepsilon\le10^{-2}$ down to
  $10^{-6}$ (paired per-$(n,\varepsilon)$ comparison).
- **H3** falsified iff the best $\Phi$+ghost-penalty combination fails to
  beat $\max(\text{F-alone}, \text{GP-alone})$ conditioning by more than 5%
  beyond measurement noise (median over $\varepsilon$, all meshes).

## 6. Statistics (frozen)

- T2 ensemble: median and IQR of $\kappa$ (never mean).
- Pairwise convergence-order comparisons among the 13 closed-form variants
  (F1–F6): two-sided paired tests with Bonferroni $\alpha'=0.05/78$.
- H2/H3 family-level tests: Wilcoxon signed-rank with Bonferroni across the
  three hypothesis families, $\alpha'=0.05/3$.
- Fitted constants (F8): 95% bootstrap CI, 1000 resamples, seed 20260826.
- Solver-artifact check: per-cell agreement across S1/S2/S3; disagreement
  beyond $10^{-6}$ relative flags the row.

## 7. Deviations from the originating methodology document (logged before data)

- **D1** — solver trio: PETSc is unavailable on this machine (no conda);
  "CG+IC" is replaced by CG+Jacobi and CG+AMG(pyamg). Direct-LU leg retained.
- **D2** — additional mass-normalized eigenvalue logging for interpretability
  of the raw-$\lambda_{\min}$ gate (gate itself applied as written).
- **D3** — FEniCSx dual-implementation cross-check deferred (environment);
  scikit-fem serves as oracle-validated backend instead (unit tests compare
  against high-level scikit-fem assembly on uncut domains).
- **D4** — non-circle/non-ellipse level sets use adaptive-subdivision cutting
  whose geometric error is *measured* per cell (self-consistency estimate
  logged) rather than guaranteed at $10^{-12}$; the $10^{-12}$ guarantee
  applies to the circle family (incl. ellipses via affine reduction), which
  covers T1, T3, T5 and the disk half of T2.

## 8. Terminal-state contract

The loop terminates — positive or negative — only when every tier has run to
completion with logged Parquet output; H1–H3 are evaluated against these
gates on actual data with exact statistics; a literature check positions any
surviving formula against prior art; coercivity claims carry either a
human-checkable derivation or the label "empirically supported, not proven";
and every numeric claim traces to a specific logged row. A negative result is
a publishable terminal state and will be written up as such.
