"""Unfitted symmetric Nitsche assembler on cut background meshes.

Poisson problem: given f inside Omega and Dirichlet data g on
Gamma = dOmega, the bilinear form and load are

    a_h(u,v) = sum_T int_{T∩Omega} grad u . grad v
               - sum_T int_Gamma_T [ (dn u) v + u (dn v) ] + lambda_T int_Gamma_T u v,
    L_h(v)   = sum_T int_{T∩Omega} f v + int_Gamma_T g ( lambda_T v - dn v ).

Optional ghost-penalty stabilization and AGFE-style aggregation act as
post-processing operators on the assembled system. Per-cell geometry
(rho, kappa, aspect, eps_cut, chosen lambda) is recorded for logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from benchmarks.mesh import TriMesh
from formulas.registry import CellGeometry, PsiFn, C_k, stabilization_lambda
from geometry.cutting import CutRegion, cut_cell


# ---------------------------------------------------------------------------
# element basis
# ---------------------------------------------------------------------------


class PkBasis:
    """P1/P2 shape functions on one triangle with analytic gradients.

    P2 dof order: vertex functions lambda_i(2 lambda_i - 1), then edge
    midpoints 4 lambda_i lambda_j ordered (1,2),(0,2),(0,1) — matching the
    mesh edge convention (edge opposite vertex 0 first).
    """

    def __init__(self, k: int, verts: np.ndarray):
        self.k = k
        self.verts = np.asarray(verts, dtype=float)
        v0, v1, v2 = self.verts
        e1, e2 = v1 - v0, v2 - v0
        self.det = float(e1[0] * e2[1] - e1[1] * e2[0])
        inv = np.array([[e2[1], -e2[0]], [-e1[1], e1[0]]]) / self.det
        # rows: gradients of barycentric coordinates (lambda_0, lambda_1, lambda_2)
        self.grad_lambda = np.stack([-(inv[0] + inv[1]), inv[0], inv[1]])

    @property
    def n_dof(self) -> int:
        return 3 if self.k == 1 else 6

    def barycentric(self, x: np.ndarray) -> np.ndarray:
        v0, v1, v2 = self.verts
        e1, e2 = v1 - v0, v2 - v0
        d = x - v0
        det = self.det
        lam1 = (d[:, 0] * e2[1] - d[:, 1] * e2[0]) / det
        lam2 = (-d[:, 0] * e1[1] + d[:, 1] * e1[0]) / det
        return np.stack([1.0 - lam1 - lam2, lam1, lam2], axis=1)

    def values(self, x: np.ndarray) -> np.ndarray:
        lam = self.barycentric(x)
        if self.k == 1:
            return lam
        l0, l1, l2 = lam[:, 0], lam[:, 1], lam[:, 2]
        return np.stack([
            l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1),
            4 * l1 * l2, 4 * l0 * l2, 4 * l0 * l1,
        ], axis=1)

    def grads(self, x: np.ndarray) -> np.ndarray:
        g = self.grad_lambda
        if self.k == 1:
            return np.broadcast_to(g[None, :, :], (len(x), 3, 2))
        lam = self.barycentric(x)
        l0, l1, l2 = lam[:, 0], lam[:, 1], lam[:, 2]
        out = np.empty((len(x), 6, 2))
        out[:, 0] = (4 * l0 - 1)[:, None] * g[0]
        out[:, 1] = (4 * l1 - 1)[:, None] * g[1]
        out[:, 2] = (4 * l2 - 1)[:, None] * g[2]
        out[:, 3] = 4 * (l1[:, None] * g[2] + l2[:, None] * g[1])
        out[:, 4] = 4 * (l0[:, None] * g[2] + l2[:, None] * g[0])
        out[:, 5] = 4 * (l0[:, None] * g[1] + l1[:, None] * g[0])
        return out


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def tri_area(elem: np.ndarray) -> float:
    return abs(
        (elem[1][0] - elem[0][0]) * (elem[2][1] - elem[0][1])
        - (elem[2][0] - elem[0][0]) * (elem[1][1] - elem[0][1])
    ) / 2.0


@dataclass
class AssemblyResult:
    A: sp.csr_matrix
    rhs: np.ndarray
    dofs: np.ndarray
    active_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    n_dof_total: int = 0
    cell_table: list[dict] = field(default_factory=list)

    def expand(self, x_active: np.ndarray) -> np.ndarray:
        """Scatter a solution on the reduced system into the full dof vector.

        Dropped (inactive) dofs correspond to basis functions supported wholly
        outside Omega; their value is irrelevant and set to zero.
        """
        full = np.zeros(self.n_dof_total)
        if len(self.active_ids):
            full[self.active_ids] = x_active
        return full


def assemble_nitsche(
    mesh: TriMesh,
    levelset,
    psi: PsiFn,
    k: int = 1,
    f=None,
    g=None,
    c_k: float | None = None,
    vol_order: int | None = None,
    bnd_order: int = 12,
    sigma_gp: float | None = None,
    aggregator=None,
    eps_c: float | None = None,
) -> AssemblyResult:
    """Assemble the unfitted symmetric Nitsche system.

    ``sigma_gp`` activates ghost-penalty stabilization with the
    preregistered scaling. When ``eps_c`` is given together with
    ``aggregator``, elements with cut fraction below ``eps_c`` are merged by
    the aggregator before static condensation (the F7 hybrid path).
    """
    if f is None or g is None:
        raise ValueError("manufactured-data callables f and g are required")
    if c_k is None:
        c_k = C_k(k)
    if vol_order is None:
        vol_order = max(8, 2 * k + 4)

    n_dof = mesh.dof_count(k)
    dofs = mesh.element_dofs(k)
    hT_all = mesh.diameters()

    regions = [
        cut_cell(levelset, mesh.nodes[mesh.elements[e]],
                 vol_order=vol_order, bnd_order=bnd_order)
        for e in range(mesh.n_elements)
    ]

    cell_table: list[dict] = []
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n_dof)

    for e, region in enumerate(regions):
        if region.status == "empty":
            continue
        elem = mesh.nodes[mesh.elements[e]]
        basis = PkBasis(k, elem)
        loc = dofs[e]
        nd = len(loc)

        K = np.zeros((nd, nd))
        vals_q = basis.values(region.pts)
        grads_q = basis.grads(region.pts)
        w = region.wts
        K += np.einsum("q,qid,qjd->ij", w, grads_q, grads_q)
        Floc = w @ (vals_q * f(region.pts)[:, None])

        lam_T = 0.0
        if region.status == "cut":
            cell_geom = CellGeometry(
                rho=region.cut_ratio,
                h_T=float(hT_all[e]),
                k=k,
                kappa=region.curvature_mean,
                aspect=float(region.bbox_aspect),
                eps_cut=region.area / tri_area(elem),
            )
            lam_T = stabilization_lambda(psi, cell_geom, c_k)
            bp, bw, bn = region.bnd_pts, region.bnd_wts, region.bnd_nrm
            vb = basis.values(bp)
            gb = basis.grads(bp)
            dn = np.einsum("qid,qd->qi", gb, bn)
            K += (
                -np.einsum("q,qi,qj->ij", bw, dn, vb)
                - np.einsum("q,qi,qj->ij", bw, vb, dn)
                + lam_T * np.einsum("q,qi,qj->ij", bw, vb, vb)
            )
            g_pts = g(bp)[:, None]  # (q,1): Dirichlet data, broadcasts over dofs
            Floc += bw @ (vb * (lam_T * g_pts))
            Floc -= bw @ (dn * g_pts)

        cell_table.append({
            "element": e,
            "cell_id": int(mesh.cell_id[e]),
            "status": region.status,
            "area": region.area,
            "gamma_length": region.gamma_length,
            "rho": region.cut_ratio if region.area > 0 else np.inf,
            "aspect": float(region.bbox_aspect),
            "curvature": region.curvature_mean,
            "lambda": lam_T,
            "method": region.method,
            "self_check_rel": region.meta.get("self_check_rel"),
        })

        for a in range(nd):
            base = len(rows)
            rows.extend(int(loc[a]) for _ in range(nd))
            cols.extend(int(loc[b]) for b in range(nd))
            vals.extend(K[a].tolist())
            rhs[loc[a]] += Floc[a]

    A = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

    if sigma_gp is not None:
        A = (A + ghost_penalty_matrix(mesh, regions, dofs, k, sigma_gp)).tocsr()

    if aggregator is not None:
        assert eps_c is not None, "aggregation requires eps_c"
        groups = aggregator(mesh, regions, dofs, eps_c)
        A, rhs, _ = apply_aggregation(A, rhs, groups, dofs, mesh, k)
        # Aggregated systems are expressed on free dofs of their own reduced
        # numbering; the active-space restriction below still applies safely
        # because condensed rows were eliminated exactly.

    A.sum_duplicates()
    # Restrict to the active space: drop dofs with entirely empty rows.
    alive = np.diff(A.indptr) > 0
    n_alive = int(alive.sum())
    active_ids = np.flatnonzero(alive)
    A_red = A[alive][:, alive].tocsr()
    rhs_red = rhs[alive]
    if n_alive != n_dof:
        print(f"METRIC dropped_inactive_dofs {n_dof - n_alive}", flush=True)
    return AssemblyResult(A=A_red, rhs=rhs_red, dofs=dofs,
                          active_ids=active_ids, n_dof_total=n_dof,
                          cell_table=cell_table)


# ---------------------------------------------------------------------------
# ghost penalty
# ---------------------------------------------------------------------------


def ghost_penalty_matrix(
    mesh: TriMesh, regions: list[CutRegion], dofs: np.ndarray, k: int, sigma: float
) -> sp.csr_matrix:
    """s_h = sigma sum_F h_F^{2k+1-d} int_F [dn u][dn v].

    Faces are interior edges with at least one adjacent cut element and two
    active neighbours; d = 2. Normal-derivative jumps use each side's own
    outward face normal.
    """
    n_dof = mesh.dof_count(k)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    seen: set[tuple[int, int]] = set()
    for e, region in enumerate(regions):
        if region.status != "cut":
            continue
        for other in mesh.element_adjacency[e]:
            key = (min(e, other), max(e, other))
            if key in seen or regions[other].status == "empty":
                continue
            seen.add(key)
            shared = set(int(i) for i in mesh.edge_ids[e]) & \
                set(int(i) for i in mesh.edge_ids[other])
            if not shared:
                continue
            fid = shared.pop()
            p0, p1 = mesh.nodes[mesh.edge_nodes[fid]]
            hF = float(np.linalg.norm(p1 - p0))
            scale = sigma * hF ** (2 * k - 1)  # 2k+1-d with d=2
            t = np.linspace(0.25, 0.75, 4)
            pts = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
            gw = np.full(len(t), hF / len(t))

            ba = PkBasis(k, mesh.nodes[mesh.elements[e]])
            bb = PkBasis(k, mesh.nodes[mesh.elements[other]])
            n_a = _face_normal_toward_exterior(ba.verts, p0, p1)
            dna = np.einsum("qid,d->qi", ba.grads(pts), n_a)
            dnb = np.einsum("qid,d->qi", bb.grads(pts), -n_a)
            la, lb = dofs[e], dofs[other]
            nda, ndb = len(la), len(lb)
            Da = np.concatenate([dna, np.zeros((len(t), ndb))], axis=1)
            Db = np.concatenate([np.zeros((len(t), nda)), dnb], axis=1)
            Dj = Da - Db
            Jloc = np.einsum("qi,qj,q->ij", Dj, Dj, gw)
            both = np.concatenate([la, lb]).astype(int)
            n = len(both)
            rows.extend(np.repeat(both, n).tolist())
            cols.extend(np.tile(both, n).tolist())
            vals.extend((scale * Jloc).ravel().tolist())
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def _face_normal_toward_exterior(elem_verts: np.ndarray, p0, p1) -> np.ndarray:
    """Unit normal of face (p0,p1) pointing away from the element centroid."""
    centroid = elem_verts.mean(axis=0)
    tang = p1 - p0
    n = np.array([tang[1], -tang[0]])
    n /= np.linalg.norm(n)
    if float((centroid - p0) @ n) > 0.0:
        n = -n
    return n


# ---------------------------------------------------------------------------
# aggregation (AGFE-style static condensation)
# ---------------------------------------------------------------------------


def default_aggregator(
    mesh: TriMesh, regions: list[CutRegion], dofs: np.ndarray, eps_c: float
) -> list[list[int]]:
    """Merge every sufficiently-sliver cut element into its active neighbour
    with the largest physical area sharing an edge."""
    areas = np.array([r.area for r in regions])
    assigned: dict[int, int] = {}
    groups: list[list[int]] = []
    for e, region in enumerate(regions):
        if region.status != "cut" or region.area <= 0.0:
            continue
        if region.area / tri_area(mesh.nodes[mesh.elements[e]]) >= eps_c:
            continue
        if e in assigned:
            continue
        candidates = [
            o for o in mesh.element_adjacency[e]
            if regions[o].status in ("full", "cut") and o not in assigned
        ]
        if not candidates:
            continue
        host = max(candidates, key=lambda o: areas[o])
        groups.append([host, e])
        assigned[e] = host
        assigned[host] = host
    return groups


def dof_coordinates(mesh: TriMesh, dofs_vec: np.ndarray, k: int) -> np.ndarray:
    """Physical coordinates of global dof ids (vertices, then edge mids)."""
    nv = len(mesh.nodes)
    out = np.empty((len(dofs_vec), 2))
    for j, dof in enumerate(dofs_vec):
        d = int(dof)
        if d < nv:
            out[j] = mesh.nodes[d]
        else:
            out[j] = mesh.nodes[mesh.edge_nodes[d - nv]].mean(axis=0)
    return out


def apply_aggregation(A, rhs, groups, dofs, mesh: TriMesh, k: int):
    """Constrain aggregated dofs to the host polynomial space and condense.

    Constraint: u_c = sum_j w_j u_{h_j}, w_j = phi^{host}_j(x_c) (exact for
    P1 hosts, interpolation-consistent for P2). Reduced system:
    A_red = P^T A P, b_red = P^T b with P = [[I],[W]] over (free, constrained).
    """
    n_dof = mesh.dof_count(k)
    constrained: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for group in groups:
        host = group[0]
        host_dofs = dofs[host].astype(int)
        hb = PkBasis(k, mesh.nodes[mesh.elements[host]])
        for e in group[1:]:
            edofs = dofs[e].astype(int)
            coords = dof_coordinates(mesh, edofs, k)
            wrows = hb.values(coords)
            for j, dof in enumerate(edofs):
                if dof not in set(host_dofs.tolist()):
                    constrained[int(dof)] = (host_dofs, wrows[j])

    cons_ids = np.array(sorted(constrained), dtype=int)
    free_ids = np.array(sorted(set(range(n_dof)) - set(cons_ids.tolist())), dtype=int)
    nf, nc = len(free_ids), len(cons_ids)

    W = sp.lil_matrix((nc, nf))
    for cj, dof in enumerate(cons_ids):
        hosts, weights = constrained[int(dof)]
        for hd, wt in zip(hosts, weights):
            fi = int(np.searchsorted(free_ids, int(hd)))
            if fi < nf and free_ids[fi] == int(hd):
                W[cj, fi] = wt
    W = W.tocsr()  # u_cons = W u_free

    App = A.tocsr()[np.concatenate([free_ids, cons_ids])][:,
                  np.concatenate([free_ids, cons_ids])].tocsr()
    Aff = App[:nf, :nf]
    Afc = App[:nf, nf:]
    Acf = App[nf:, :nf]
    Acc = App[nf:, nf:]
    b = rhs[np.concatenate([free_ids, cons_ids])]
    bf, bc = b[:nf], b[nf:]

    A_red = (Aff + Afc @ W + (W.T @ Acf) + (W.T @ (Acc @ W))).tocsr()
    b_red = bf + W.T @ bc

    # map back to original numbering space (free dofs keep their ids)
    keep_map = {int(d): i for i, d in enumerate(free_ids)}
    remap = np.array(sorted(keep_map), dtype=int)
    return A_red.tocsr(), b_red, remap
