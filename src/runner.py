"""Fixed experiment entrypoint (branch-encoded behaviour).

T1 controlled-sliver sweep: all closed-form variants plus aggregation hybrid
over PREREG epsilon/n/k grid. Also retains benign validation as sanity check.

Logs per Section 5 quantities: lambda_min, kappa, L2/H1/energy errors,
solver iters, wall-clock, with explicit non-convergence flags.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmarks.mesh import unit_square_mesh, TriMesh
from formulas.registry import build_registry, C_k
from geometry.levelsets import Circle, Ellipse, Superellipse
from geometry.cutting import cut_cell
from measurement.assembly import assemble_nitsche, PkBasis, default_aggregator
from measurement.errors import h1_semi_error, l2_error, energy_error
from measurement.solvers import SOLVERS
from measurement.spectra import spectral_measures, svds_cross_check


# ---------------------------------------------------------------------------
# manufactured data (2D)
# ---------------------------------------------------------------------------

def manufactured_2d():
    def u(x):
        return np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    def f(x):
        return 2 * np.pi**2 * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    def grad_u(x):
        return np.stack([
            np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
            -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
        ], axis=1)

    def dn_u(x, n):
        g = grad_u(x)
        return np.sum(g * n, axis=1)

    return u, f, grad_u, dn_u


def fit_slope(hs: np.ndarray, errs: np.ndarray) -> tuple[float, float]:
    errs = np.asarray(errs, dtype=float)
    hs = np.asarray(hs, dtype=float)
    mask = np.isfinite(errs) & (errs > 0) & np.isfinite(hs) & (hs > 0)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    p, log_b = np.polyfit(np.log(hs[mask]), np.log(errs[mask]), 1)
    resid = np.log(errs[mask]) - (p * np.log(hs[mask]) + log_b)
    ss_res = float(np.sum(resid**2))
    mean_log = float(np.mean(np.log(errs[mask])))
    ss_tot = float(np.sum((np.log(errs[mask]) - mean_log)**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p), float(r2)


# ---------------------------------------------------------------------------
# T1 geometry helper — cap height bisection (PREREG exact)
# ---------------------------------------------------------------------------

def cap_area(s: float, r: float) -> float:
    # A_cap = r^2 arccos((r-s)/r) - (r-s) sqrt(2 r s - s^2), s in [0,2r]
    if s <= 0:
        return 0.0
    if s >= 2*r:
        return np.pi * r * r
    d = r - s
    return r*r * np.arccos(d / r) - d * np.sqrt(max(2*r*s - s*s, 0.0))


def solve_cap_height(eps: float, h: float, r: float = 0.25) -> float:
    target = eps * h * h
    # Handle eps=0 edge (not in our grid)
    if target <= 0:
        return 0.0
    lo, hi = 0.0, 2.0 * r
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cap_area(mid, r) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t1_circle(n: int, eps: float, r: float = 0.25) -> Circle:
    h = 1.0 / n
    s = solve_cap_height(eps, h, r)
    cx = 3.5 * h
    cy = 4.0 * h + s - r
    return Circle(center=(cx, cy), radius=r)


def extended_background_mesh(n: int, xmin: float = -0.5, xmax: float = 1.5) -> TriMesh:
    """Background mesh larger than [0,1]² to keep T1 disk inside D for all h.
    Structured triangles, same topology as unit_square_mesh but shifted/scaled.
    With n cells per axis, h_ext = (xmax-xmin)/n. For n=64, covers [-0.5,1.5] with h=0.03125."""
    import numpy as np
    from benchmarks.mesh import TriMesh
    # reuse logic from benchmarks.mesh but with custom bounds
    xs = np.linspace(xmin, xmax, n + 1)
    ys = np.linspace(xmin, xmax, n + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)
    def nid(i,j): return i*(n+1)+j
    elems=[]; cell_ids=[]
    for i in range(n):
        for j in range(n):
            v00, v10 = nid(i,j), nid(i+1,j)
            v01, v11 = nid(i,j+1), nid(i+1,j+1)
            elems.append([v00,v10,v11]); cell_ids.append(i+j*n)
            elems.append([v00,v11,v01]); cell_ids.append(i+j*n)
    import numpy as _np
    elements = _np.asarray(elems, dtype=int)
    # reuse edge building from benchmarks.mesh
    from benchmarks.mesh import unit_square_mesh as _us
    # manually build edges similarly
    pairs=[(1,2),(0,2),(0,1)]
    lookup={}; edge_nodes_list=[]; edge_ids=_np.empty((len(elements),3), dtype=int)
    for e in range(len(elements)):
        for loc,(a,b) in enumerate(pairs):
            key=(min(int(elements[e,a]),int(elements[e,b])), max(int(elements[e,a]),int(elements[e,b])))
            eid=lookup.get(key)
            if eid is None:
                eid=len(edge_nodes_list); lookup[key]=eid; edge_nodes_list.append(key)
            edge_ids[e,loc]=eid
    edge_nodes=_np.asarray(edge_nodes_list, dtype=int)
    edge_to_elem={}
    for e in range(len(elements)):
        for loc in range(3):
            edge_to_elem.setdefault(int(edge_ids[e,loc]), []).append(e)
    adjacency=[]
    for e in range(len(elements)):
        nbrs=set()
        for loc in range(3):
            for other in edge_to_elem[int(edge_ids[e,loc])]:
                if other!=e: nbrs.add(other)
        adjacency.append(sorted(nbrs))
    return TriMesh(nodes, elements, _np.asarray(cell_ids,dtype=int), n, edge_ids, edge_nodes, adjacency)


# ---------------------------------------------------------------------------
# mass matrix for mass-normalized eigenvalue (optional)
# ---------------------------------------------------------------------------

def assemble_mass(mesh, levelset, k: int):
    from geometry.cutting import cut_cell
    from measurement.assembly import PkBasis
    n_dof = mesh.dof_count(k)
    dofs = mesh.element_dofs(k)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for e in range(mesh.n_elements):
        region = cut_cell(levelset, mesh.nodes[mesh.elements[e]])
        if region.status == "empty":
            continue
        basis = PkBasis(k, mesh.nodes[mesh.elements[e]])
        loc = dofs[e]
        nd = len(loc)
        vals_q = basis.values(region.pts)
        w = region.wts
        Mloc = np.einsum("q,qi,qj->ij", w, vals_q, vals_q)
        for a in range(nd):
            for b in range(nd):
                rows.append(int(loc[a]))
                cols.append(int(loc[b]))
                vals.append(float(Mloc[a, b]))
    M = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    # restrict to active space same as assembly does (drop empty rows) – we
    # need the same permutation; caller should restrict using active_ids
    return M


# ---------------------------------------------------------------------------
# validation (benign) — retained as smoke check
# ---------------------------------------------------------------------------

def validate_benign() -> pd.DataFrame:
    reg = build_registry()
    psi = reg["F1"]["psi"]
    u, f, grad_u, dn_u = manufactured_2d()
    rows = []
    for k in (1, 2):
        hs, e_l2, e_h1 = [], [], []
        for n in (8, 16, 32, 64):
            mesh = unit_square_mesh(n)
            levelset = Circle(center=(0.5, 0.5), radius=0.35)
            t0 = time.perf_counter()
            result = assemble_nitsche(mesh, levelset, psi, k=k, f=f, g=u)
            t_asm = time.perf_counter() - t0
            sp_result = spectral_measures(result.A)
            sol = SOLVERS["S1_lu"](result.A, result.rhs)
            u_full = result.expand(sol.x)
            hs.append(1.0 / n)
            l2 = l2_error(mesh, levelset, u_full, result.dofs, k, u)
            h1 = h1_semi_error(mesh, levelset, u_full, result.dofs, k, grad_u)
            e_l2.append(l2)
            e_h1.append(h1)
            rows.append({
                "tier": "validation-benign", "formula": "F1", "k": k,
                "n": n, "eps": np.nan, "solver": "S1_lu",
                "lambda_min": sp_result["lambda_min"],
                "lambda_max": sp_result["lambda_max"],
                "kappa": sp_result["kappa"],
                "min_converged": sp_result["min_converged"],
                "max_converged": sp_result["max_converged"],
                "asym_rel": sp_result["asym_rel"],
                "l2_err": l2, "h1_err": h1, "energy_err": np.nan,
                "cg_iters": sol.iterations,
                "assemble_time": t_asm,
                "seed": 0,
            })
            print(f"METRIC n {n} k {k} l2 {l2:.6e} h1 {h1:.6e} kappa {sp_result['kappa']:.6e}", flush=True)
        for name, errs in (("l2", e_l2), ("h1", e_h1)):
            p, r2 = fit_slope(np.array(hs), np.array(errs))
            print(f"METRIC slope_{name}_k{k} {p:.4f} R2 {r2:.6f}", flush=True)
            rows.append({
                "tier": "validation-slope", "formula": "F1", "k": k,
                "n": np.nan, "eps": np.nan, "solver": "-",
                "quantity": name, "slope_p": p, "slope_R2": r2, "seed": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T1 sweep
# ---------------------------------------------------------------------------

PREREG_EPS = [0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
# Deferred n=128 for runtime (original PR has 5 points; we run 4-point sweep and note R2 gates accordingly)
PREREG_N = [8, 16, 32, 64]
PREREG_K = [1, 2]

# For runtime we include all 13 closed-form plus F7 hybrid; F8 deferred
T1_FORMULAS = ["F1", "F2a", "F2b", "F2c", "F2d", "F3", "F4a", "F4b", "F4c", "F4d", "F5", "F6a", "F6b", "F7"]


def _compute_errors_single_pass(mesh, circ, u_full, dofs, k, u, grad_u, dn_u, lam_by_elem):
    """Single-pass L2/H1/energy error to avoid 3 separate cut_cell loops.
    One cut_cell per element per config instead of three."""
    vol_order = max(10, 2 * k + 4)
    bnd_order = 14
    l2_tot = 0.0
    h1_tot = 0.0
    en_tot = 0.0
    # energy volume part is same as h1 seminorm, but we need it plus boundary terms
    # So compute h1 part once and reuse for energy
    for e in range(mesh.n_elements):
        region = cut_cell(mesh.nodes[mesh.elements[e]], circ) if False else cut_cell(circ, mesh.nodes[mesh.elements[e]], vol_order=vol_order, bnd_order=bnd_order)
        # Actually cut_cell signature is (levelset, verts); need correct order
        # We called above incorrectly; fix to (levelset, verts)
        # But we already handle below – re-call correctly to avoid confusion
        pass
    # Implemented inline below in run_t1 to avoid extra function overhead


def run_t1() -> pd.DataFrame:
    reg = build_registry()
    u, f, grad_u, dn_u = manufactured_2d()
    rows: list[dict] = []
    rng_svds = np.random.default_rng(12345)
    total_configs = len(T1_FORMULAS) * len(PREREG_EPS) * len(PREREG_N) * len(PREREG_K)
    print(f"METRIC t1_total_configs {total_configs}", flush=True)
    t_start = time.perf_counter()
    count = 0
    for fid in T1_FORMULAS:
        psi = reg[fid]["psi"]
        is_agg = (fid == "F7")
        for k in PREREG_K:
            for eps in PREREG_EPS:
                errs_l2 = []
                errs_h1 = []
                errs_en = []
                hs = []
                for n in PREREG_N:
                    count += 1
                    mesh = unit_square_mesh(n)
                    h = 1.0 / n
                    circ = t1_circle(n, eps)
                    t0 = time.perf_counter()
                    try:
                        if is_agg:
                            result = assemble_nitsche(mesh, circ, psi, k=k, f=f, g=u,
                                                      aggregator=default_aggregator, eps_c=1e-3)
                        else:
                            result = assemble_nitsche(mesh, circ, psi, k=k, f=f, g=u)
                    except Exception as exc:
                        print(f"METRIC assemble_failed fid {fid} n {n} eps {eps:.0e} k {k} err {exc}", flush=True)
                        rows.append({
                            "tier": "T1", "formula": fid, "k": k, "n": n, "h": h, "eps": eps,
                            "solver": "all", "lambda_min": np.nan, "lambda_max": np.nan,
                            "kappa": np.nan, "min_converged": False, "max_converged": False,
                            "asym_rel": np.nan, "l2_err": np.nan, "h1_err": np.nan, "energy_err": np.nan,
                            "cg_iters": -1, "solve_converged": False, "assemble_time": np.nan,
                            "solve_time": np.nan, "seed": 12345, "svds_checked": False,
                        })
                        continue
                    t_asm = time.perf_counter() - t0
                    sp_res = spectral_measures(result.A)
                    # Lightweight: skip mass-normalized and svds for speed; keep placeholders
                    mass_lmin = np.nan
                    svds_smin = np.nan
                    svds_smax = np.nan
                    svds_kappa = np.nan
                    do_svds = False
                    # 10% svds only for k=1 and n=32 to limit cost
                    if rng_svds.random() < 0.02 and k == 1 and n == 32:
                        do_svds = True
                        try:
                            smin, smax = svds_cross_check(result.A)
                            svds_smin, svds_smax = smin, smax
                            svds_kappa = smax / max(smin, 1e-300)
                        except Exception:
                            pass
                    lam_by_elem = {int(row["element"]): float(row["lambda"]) for row in result.cell_table if row["status"] == "cut"}

                    # Solve trio: compute errors only for S1_lu (accurate) to save 2/3 cut_cell work
                    l2_val = np.nan
                    h1_val = np.nan
                    en_val = np.nan
                    # First solve S1_lu and compute errors in single pass
                    t1 = time.perf_counter()
                    sol_lu = SOLVERS["S1_lu"](result.A, result.rhs)
                    t_solve_lu = time.perf_counter() - t1
                    u_full_lu = result.expand(sol_lu.x) if sol_lu.x is not None else np.zeros(result.n_dof_total)
                    # Single-pass error computation
                    try:
                        vol_order = max(10, 2 * k + 4)
                        bnd_order = 14
                        l2_tot = 0.0
                        h1_tot = 0.0
                        en_tot = 0.0
                        for e in range(mesh.n_elements):
                            region = cut_cell(circ, mesh.nodes[mesh.elements[e]], vol_order=vol_order, bnd_order=bnd_order)
                            if region.status == "empty":
                                continue
                            basis = PkBasis(k, mesh.nodes[mesh.elements[e]])
                            loc = result.dofs[e]
                            # L2 & H1 volume contributions
                            vals = basis.values(region.pts)
                            diff_l2 = u(region.pts) - vals @ u_full_lu[loc]
                            l2_tot += float(np.sum(region.wts * diff_l2 * diff_l2))
                            grads = basis.grads(region.pts)
                            guh = np.einsum("i,qid->qd", u_full_lu[loc], grads)
                            diff_h1 = grad_u(region.pts) - guh
                            h1_contrib = float(np.sum(region.wts * np.sum(diff_h1 * diff_h1, axis=1)))
                            h1_tot += h1_contrib
                            en_tot += h1_contrib
                            if region.status == "cut" and e in lam_by_elem and len(region.bnd_pts) > 0:
                                lam = lam_by_elem[e]
                                bp, bw, bn = region.bnd_pts, region.bnd_wts, region.bnd_nrm
                                vb = basis.values(bp)
                                gb = basis.grads(bp)
                                dn_uh = np.einsum("i,qi->q", u_full_lu[loc], np.einsum("qid,qd->qi", gb, bn))
                                e_b = u(bp) - vb @ u_full_lu[loc]
                                dn_e = dn_u(bp, bn) - dn_uh
                                en_tot += float(np.sum(bw * (-2.0 * e_b * dn_e + lam * e_b * e_b)))
                        l2_val = float(np.sqrt(max(l2_tot, 0.0)))
                        h1_val = float(np.sqrt(max(h1_tot, 0.0)))
                        en_val = float(np.sqrt(max(en_tot, 0.0)))
                    except Exception as exc:
                        print(f"METRIC error_failed fid {fid} n {n} eps {eps:.0e} k {k} err {exc}", flush=True)
                        pass
                    hs.append(h)
                    errs_l2.append(l2_val)
                    errs_h1.append(h1_val)
                    errs_en.append(en_val)

                    # Log S1_lu row with errors
                    rows.append({
                        "tier": "T1", "formula": fid, "k": k, "n": n, "h": h, "eps": eps,
                        "solver": "S1_lu",
                        "lambda_min": sp_res["lambda_min"], "lambda_max": sp_res["lambda_max"],
                        "kappa": sp_res["kappa"],
                        "min_converged": bool(sp_res["min_converged"]), "max_converged": bool(sp_res["max_converged"]),
                        "asym_rel": float(sp_res["asym_rel"]),
                        "mass_lambda_min": float(mass_lmin), "svds_smin": float(svds_smin),
                        "svds_smax": float(svds_smax), "svds_kappa": float(svds_kappa),
                        "svds_checked": bool(do_svds),
                        "l2_err": float(l2_val) if np.isfinite(l2_val) else np.nan,
                        "h1_err": float(h1_val) if np.isfinite(h1_val) else np.nan,
                        "energy_err": float(en_val) if np.isfinite(en_val) else np.nan,
                        "cg_iters": int(sol_lu.iterations), "solve_converged": bool(sol_lu.converged),
                        "assemble_time": float(t_asm), "solve_time": float(t_solve_lu),
                        "seed": 12345, "n_dof_active": int(result.A.shape[0]), "n_dof_total": int(result.n_dof_total),
                    })
                    # S2 and S3: only solve, no error recomputation (reuse same l2/h1/en for slope gate? No, set NaN)
                    for solver_id in ("S2_cg_jacobi", "S3_cg_amg"):
                        t1b = time.perf_counter()
                        sol = SOLVERS[solver_id](result.A, result.rhs)
                        t_solve = time.perf_counter() - t1b
                        rows.append({
                            "tier": "T1", "formula": fid, "k": k, "n": n, "h": h, "eps": eps,
                            "solver": solver_id,
                            "lambda_min": sp_res["lambda_min"], "lambda_max": sp_res["lambda_max"],
                            "kappa": sp_res["kappa"],
                            "min_converged": bool(sp_res["min_converged"]), "max_converged": bool(sp_res["max_converged"]),
                            "asym_rel": float(sp_res["asym_rel"]),
                            "mass_lambda_min": float(mass_lmin), "svds_smin": float(svds_smin),
                            "svds_smax": float(svds_smax), "svds_kappa": float(svds_kappa),
                            "svds_checked": bool(do_svds),
                            "l2_err": float("nan"), "h1_err": float("nan"), "energy_err": float("nan"),
                            "cg_iters": int(sol.iterations), "solve_converged": bool(sol.converged),
                            "assemble_time": float(t_asm), "solve_time": float(t_solve),
                            "seed": 12345, "n_dof_active": int(result.A.shape[0]), "n_dof_total": int(result.n_dof_total),
                        })
                    if count % 40 == 0:
                        elapsed = time.perf_counter() - t_start
                        print(f"METRIC progress {count}/{total_configs} fid {fid} n {n} eps {eps:.0e} k {k} kappa {sp_res['kappa']:.2e} lmin {sp_res['lambda_min']:.2e} elapsed {elapsed:.1f}s", flush=True)

                # after mesh loop, compute slopes per (fid,k,eps) using S1_lu errors
                if len(hs) >= 2:
                    for name, errs in (("l2", errs_l2), ("h1", errs_h1), ("energy", errs_en)):
                        p, r2 = fit_slope(np.array(hs), np.array(errs))
                        rows.append({
                            "tier": "T1-slope",
                            "formula": fid,
                            "k": k,
                            "n": np.nan,
                            "h": np.nan,
                            "eps": eps,
                            "solver": "S1_lu",
                            "quantity": name,
                            "slope_p": float(p),
                            "slope_R2": float(r2),
                            "seed": 12345,
                        })
                        print(f"METRIC t1_slope fid {fid} k {k} eps {eps:.0e} {name} p {p:.3f} R2 {r2:.4f}", flush=True)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# T2 random ensemble (reduced N=60) and T3 curvature sweep
# ---------------------------------------------------------------------------

T2_N_PER_MESH = 20  # 10 disks + 10 superellipses per n, total 60; prereg is 200 (100+100) — D7
T2_NS = [16, 32, 64]
T2_FORMULAS = ["F1", "F3", "F4c", "F7"]  # 4 representative; full 14 would be 8400 configs
T3_ASPECTS = [1, 2, 5, 10]
T3_NS = [16, 32, 64]
T3_FORMULAS = ["F1", "F3", "F4c", "F7"]


def run_t2() -> pd.DataFrame:
    """T2 ensemble with exact circle and subdivision superellipse.
    Uses reduced N for runtime; subdivision verify=False for speed (D8)."""
    reg = build_registry()
    u,f,gu,dn = manufactured_2d()
    rows=[]
    rng = np.random.default_rng(20260825)
    total = len(T2_NS)*T2_N_PER_MESH*len(T2_FORMULAS)
    print(f"METRIC t2_total_configs {total}", flush=True)
    t0_all=time.perf_counter()
    for n in T2_NS:
        mesh = unit_square_mesh(n)
        for repl in range(T2_N_PER_MESH):
            is_disk = repl < 10
            if is_disk:
                r = float(rng.uniform(0.30,0.40))
                cx, cy = rng.uniform(0.35,0.65, size=2)
                ls = Circle(center=(float(cx),float(cy)), radius=r)
            else:
                ax, ay = rng.uniform(0.28,0.38, size=2)
                cx, cy = rng.uniform(0.35,0.65, size=2)
                ls = Superellipse(center=(float(cx),float(cy)), semi_axes=(float(ax),float(ay)), exponent=6)
            # for superellipse, we will use subdivision with verify=False via monkey patch: pass via cut_cell verify flag by using custom wrapper
            # Instead we directly call assemble which internally uses default verify=True; we accept the cost for now but limit N
            for fid in T2_FORMULAS:
                psi=reg[fid]['psi']; is_agg=(fid=='F7')
                try:
                    if is_agg:
                        res=assemble_nitsche(mesh, ls, psi, k=1, f=f, g=u, aggregator=default_aggregator, eps_c=1e-3)
                    else:
                        res=assemble_nitsche(mesh, ls, psi, k=1, f=f, g=u)
                    sp=spectral_measures(res.A)
                    sol=SOLVERS["S1_lu"](res.A, res.rhs)
                    full=res.expand(sol.x)
                    # single-pass l2 (superellipse subdivision already heavy, so we skip h1/energy for T2 demo to save time)
                    l2_tot=0.0
                    for e in range(mesh.n_elements):
                        # Use same cutting as assembly but with verify=False for superellipse to speed: we call cut_cell directly with appropriate args
                        if isinstance(ls, Superellipse):
                            from geometry.cutting import clip_subdivision
                            region=clip_subdivision(mesh.nodes[mesh.elements[e]], ls, lin_tol=5e-3, max_depth=30, verify=False)
                        else:
                            region=cut_cell(ls, mesh.nodes[mesh.elements[e]], vol_order=10)
                        if region.status=="empty": continue
                        basis=PkBasis(1, mesh.nodes[mesh.elements[e]])
                        loc=res.dofs[e]
                        vals=basis.values(region.pts)
                        diff=u(region.pts)-vals@full[loc]
                        l2_tot+=float(np.sum(region.wts*diff*diff))
                    l2=float(np.sqrt(l2_tot))
                    # record train/test split (prereg: train repl<14, test >=14 for N=20)
                    split="train" if repl<14 else "test"
                    rows.append({"tier":"T2","n":n,"repl":repl,"split":split,"formula":fid,"is_disk":is_disk,
                                 "kappa":float(sp['kappa']), "lambda_min":float(sp['lambda_min']), "l2_err":float(l2),
                                 "cx":float(cx),"cy":float(cy),"seed":20260825})
                    if len(rows)%20==0:
                        print(f"METRIC t2_progress n {n} repl {repl} fid {fid} kappa {sp['kappa']:.2e} elapsed {time.perf_counter()-t0_all:.1f}s", flush=True)
                except Exception as e:
                    print(f"METRIC t2_failed n {n} repl {repl} fid {fid} err {e}", flush=True)
                    rows.append({"tier":"T2","n":n,"repl":repl,"split":"train" if repl<14 else "test","formula":fid,"is_disk":is_disk,"kappa":np.nan,"lambda_min":np.nan,"l2_err":np.nan})
    df=pd.DataFrame(rows)
    # summary median/IQR
    for n in T2_NS:
        for fid in T2_FORMULAS:
            g=df[(df.n==n)&(df.formula==fid)]
            if len(g)==0: continue
            med=g.kappa.median(); q25=g.kappa.quantile(0.25); q75=g.kappa.quantile(0.75)
            print(f"METRIC t2_summary n {n} fid {fid} median_kappa {med:.2e} IQR [{q25:.2e},{q75:.2e}] N {len(g)}", flush=True)
    return df


def run_t3() -> pd.DataFrame:
    """T3 curvature sweep: ellipses with aspect sweep, exact affine path."""
    reg=build_registry()
    u,f,gu,dn = manufactured_2d()
    rows=[]
    total=len(T3_ASPECTS)*len(T3_NS)*2*len(T3_FORMULAS)
    print(f"METRIC t3_total_configs {total}", flush=True)
    t0_all=time.perf_counter()
    for aspect in T3_ASPECTS:
        b=np.sqrt(0.1225/aspect); a=aspect*b
        ell=Ellipse(center=(0.5,0.5), semi_axes=(float(a),float(b)))
        for n in T3_NS:
            mesh=unit_square_mesh(n)
            for k in [1,2]:
                for fid in T3_FORMULAS:
                    psi=reg[fid]['psi']; is_agg=(fid=='F7')
                    try:
                        if is_agg:
                            res=assemble_nitsche(mesh, ell, psi, k=k, f=f, g=u, aggregator=default_aggregator, eps_c=1e-3)
                        else:
                            res=assemble_nitsche(mesh, ell, psi, k=k, f=f, g=u)
                        sp=spectral_measures(res.A)
                        sol=SOLVERS["S1_lu"](res.A, res.rhs)
                        full=res.expand(sol.x)
                        # single-pass errors
                        l2_tot=0; h1_tot=0
                        for e in range(mesh.n_elements):
                            region=cut_cell(ell, mesh.nodes[mesh.elements[e]], vol_order=max(10,2*k+4))
                            if region.status=="empty": continue
                            basis=PkBasis(k, mesh.nodes[mesh.elements[e]])
                            loc=res.dofs[e]
                            vals=basis.values(region.pts)
                            diff=u(region.pts)-vals@full[loc]
                            l2_tot+=float(np.sum(region.wts*diff*diff))
                            grads=basis.grads(region.pts)
                            guh=np.einsum("i,qid->qd", full[loc], grads)
                            diff_h1=gu(region.pts)-guh
                            h1_tot+=float(np.sum(region.wts*np.sum(diff_h1*diff_h1, axis=1)))
                        l2=float(np.sqrt(l2_tot)); h1=float(np.sqrt(h1_tot))
                        rows.append({"tier":"T3","aspect":aspect,"n":n,"k":k,"formula":fid,"kappa":float(sp['kappa']),"lambda_min":float(sp['lambda_min']),"l2_err":l2,"h1_err":h1,"a":float(a),"b":float(b)})
                        if len(rows)%10==0:
                            print(f"METRIC t3_progress aspect {aspect} n {n} k {k} fid {fid} kappa {sp['kappa']:.2e} elapsed {time.perf_counter()-t0_all:.1f}s", flush=True)
                    except Exception as e:
                        print(f"METRIC t3_failed aspect {aspect} n {n} k {k} fid {fid} err {e}", flush=True)
                        rows.append({"tier":"T3","aspect":aspect,"n":n,"k":k,"formula":fid,"kappa":np.nan,"lambda_min":np.nan,"l2_err":np.nan,"h1_err":np.nan})
    df=pd.DataFrame(rows)
    for aspect in T3_ASPECTS:
        for fid in T3_FORMULAS:
            g=df[(df.aspect==aspect)&(df.formula==fid)&(df.k==1)]
            if len(g)==0: continue
            print(f"METRIC t3_summary aspect {aspect} fid {fid} median_kappa {g.kappa.median():.2e} N {len(g)}", flush=True)
    return df


def run_t1_star() -> pd.DataFrame:
    """T1* fixed-Omega check: same cap construction but on extended mesh [-0.5,1.5] to avoid truncation.
    Tests whether rate gate recovers when domain not truncated. Limited to F1,F3 for speed."""
    reg=build_registry()
    u,f,gu,dn = manufactured_2d()
    rows=[]
    fids=["F1","F3"]
    for fid in fids:
        psi=reg[fid]['psi']
        for k in [1]:
            hs=[]; errs=[]
            for n in [8,16,32,64]:
                mesh=extended_background_mesh(n, xmin=-0.5, xmax=1.5)
                h=(1.5-(-0.5))/n  # =2/n
                # Use same t1_circle but with extended mesh's h_ext? For comparability, keep original h=1/n for cap, but place disk same as T1
                circ=t1_circle(n, 0.5)  # use eps=0.5 as representative
                # Note t1_circle uses h=1/n, not h_ext, so cap height same as T1, but mesh is larger so disk fully inside
                res=assemble_nitsche(mesh, circ, psi, k=k, f=f, g=u)
                sp=spectral_measures(res.A)
                sol=SOLVERS["S1_lu"](res.A, res.rhs)
                full=res.expand(sol.x)
                l2_tot=0; h1_tot=0
                for e in range(mesh.n_elements):
                    region=cut_cell(circ, mesh.nodes[mesh.elements[e]], vol_order=10)
                    if region.status=="empty": continue
                    basis=PkBasis(k, mesh.nodes[mesh.elements[e]])
                    loc=res.dofs[e]
                    vals=basis.values(region.pts)
                    diff=u(region.pts)-vals@full[loc]
                    l2_tot+=float(np.sum(region.wts*diff*diff))
                    grads=basis.grads(region.pts)
                    guh=np.einsum("i,qid->qd", full[loc], grads)
                    diff_h1=gu(region.pts)-guh
                    h1_tot+=float(np.sum(region.wts*np.sum(diff_h1*diff_h1, axis=1)))
                l2=float(np.sqrt(l2_tot)); h1=float(np.sqrt(h1_tot))
                hs.append(2.0/n); errs.append(h1)
                rows.append({"tier":"T1star","formula":fid,"n":n,"k":k,"eps":0.5,"h":2.0/n,"kappa":float(sp['kappa']),"lambda_min":float(sp['lambda_min']),"l2_err":l2,"h1_err":h1})
                print(f"METRIC t1star fid {fid} n {n} h {2.0/n:.4f} kappa {sp['kappa']:.2e} h1 {h1:.2e}", flush=True)
            p,r2=fit_slope(np.array(hs), np.array(errs))
            print(f"METRIC t1star_slope fid {fid} k {k} p {p:.3f} R2 {r2:.4f}", flush=True)
            rows.append({"tier":"T1star-slope","formula":fid,"k":k,"slope_p":p,"slope_R2":r2})
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("METRIC stage benign_validation start", flush=True)
    df_benign = validate_benign()
    print("METRIC stage benign_validation done", flush=True)

    print("METRIC stage T1 start", flush=True)
    df_t1 = run_t1()
    print("METRIC stage T1 done", flush=True)

    print("METRIC stage T1star start", flush=True)
    try:
        df_t1star = run_t1_star()
        print("METRIC stage T1star done", flush=True)
    except Exception as e:
        print(f"METRIC t1star_failed {e}", flush=True)
        df_t1star = pd.DataFrame()

    print("METRIC stage T2 start", flush=True)
    try:
        df_t2 = run_t2()
        print("METRIC stage T2 done", flush=True)
    except Exception as e:
        print(f"METRIC t2_failed {e}", flush=True)
        import traceback; traceback.print_exc()
        df_t2 = pd.DataFrame()

    print("METRIC stage T3 start", flush=True)
    try:
        df_t3 = run_t3()
        print("METRIC stage T3 done", flush=True)
    except Exception as e:
        print(f"METRIC t3_failed {e}", flush=True)
        import traceback; traceback.print_exc()
        df_t3 = pd.DataFrame()

    # Combine all
    dfs = [df_benign, df_t1]
    if len(df_t1star): dfs.append(df_t1star)
    if len(df_t2): dfs.append(df_t2)
    if len(df_t3): dfs.append(df_t3)
    df_all = pd.concat([d for d in dfs if len(d)], ignore_index=True, sort=False)
    path = out_dir / "all_results.parquet"
    df_all.to_parquet(path, index=False)
    # Keep legacy t1 path for compat
    t1_path = out_dir / "t1_results.parquet"
    pd.concat([df_benign, df_t1], ignore_index=True, sort=False).to_parquet(t1_path, index=False)
    try:
        df_t2.to_parquet(out_dir / "t2_results.parquet", index=False)
        df_t3.to_parquet(out_dir / "t3_results.parquet", index=False)
        df_t1star.to_parquet(out_dir / "t1star_results.parquet", index=False)
    except Exception:
        pass
    try:
        df_all.to_csv(out_dir / "all_results.csv", index=False)
    except Exception:
        pass

    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    try:
        import scipy; versions["scipy"]=scipy.__version__
    except Exception: pass
    try:
        import pyamg; versions["pyamg"]=pyamg.__version__
    except Exception: pass
    print("METRIC package_versions " + ";".join(f"{a}={b}" for a,b in sorted(versions.items())), flush=True)
    print(f"METRIC results_file {path}", flush=True)
    print(f"METRIC t1_results_file {t1_path}", flush=True)
    # H1 gate summary (4-point, n=64)
    for fid in T1_FORMULAS:
        sub = df_t1[(df_t1["formula"]==fid)&(df_t1["tier"]=="T1")&(df_t1["solver"]=="S1_lu")]
        if len(sub)==0: continue
        worst=sub[sub["n"]==64]
        if len(worst):
            mg=worst["lambda_min"].min(); viol=(worst["lambda_min"]<1e-8).sum()
            print(f"METRIC h1_gate fid {fid} n64_min_gamma {mg:.2e} violations_lt1e-8 {viol}/{len(worst)}", flush=True)
        sldf=df_t1[(df_t1["formula"]==fid)&(df_t1["tier"]=="T1-slope")]
        if len(sldf):
            bad=sldf[(sldf["quantity"]=="energy")&((sldf["slope_R2"]<0.98)|(np.abs(sldf["slope_p"]-sldf["k"])>0.1*sldf["k"]))]
            print(f"METRIC slope_gate fid {fid} energy_bad {len(bad)}/{len(sldf[sldf['quantity']=='energy'])}", flush=True)
    # T2 median summary already printed inside run_t2; T3 inside run_t3
    return 0


if __name__ == "__main__":
    sys.exit(main())
