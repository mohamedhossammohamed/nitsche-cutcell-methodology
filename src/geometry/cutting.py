"""Clipping of background elements against level-set regions.

Mechanisms, in descending order of strength:

1. ``Circle`` level sets (the controlled-sliver benchmark family): the cut
   region T ∩ Ω decomposes into polygonal runs along the element boundary
   plus circular-segment crescents. Areas and physical-boundary measures
   follow in closed form from Green's theorem, so the geometric quantities
   entering the stabilization formula carry no tessellation error at any
   sliver fraction. Interior integrals use a Gauss rule on the polygonal fan
   plus tensor Gauss rules on each crescent; boundary integrals use composite
   Gauss along the arcs.

2. ``Ellipse`` level sets: an affine map sends the ellipse to the unit disk
   and the element to another triangle, so the exact circle path applies and
   results are pulled back (Jacobian-scaled). Exactness survives the map.

3. General smooth level sets (e.g. superellipse ensembles): recursive
   midpoint subdivision with tangent-line clipping of mixed leaves,
   terminated by a curvature criterion. Accuracy is not assumed — it is
   measured by an internal refinement self-check and reported alongside the
   results.

Convention: ``phi > 0`` inside Ω; outward normals satisfy
``n = -grad(phi)/|grad(phi)|``.

Geometry note: a circle may cross a convex element up to six times (each
edge twice), yielding multiple boundary arcs. The loop construction below
pairs every exit crossing with the next entering crossing in counter-
clockwise traversal order and therefore handles all such configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss

from .levelsets import Circle, Ellipse
from .quadrature import affine_map_to_physical, gauss_interval, gauss_triangle

_TWO_PI = 2.0 * np.pi


@dataclass
class CutRegion:
    """Quadrature data for T ∩ Ω within one background element."""

    status: str  # 'full' | 'cut' | 'empty'
    area: float  # |T ∩ Omega|
    gamma_length: float  # |T ∩ dOmega| (physical boundary inside T only)
    pts: np.ndarray  # (m, dim) interior quadrature nodes
    wts: np.ndarray  # (m,) interior weights, sum equals area
    bnd_pts: np.ndarray  # (k, dim) nodes on Gamma_T
    bnd_wts: np.ndarray  # (k,) ds weights, sum equals gamma_length
    bnd_nrm: np.ndarray  # (k, dim) outward unit normals of Omega
    bbox_aspect: float  # aspect ratio of the region bounding box (>= 1)
    curvature_mean: float  # mean |kappa| over Gamma_T
    method: str  # provenance tag
    meta: dict = field(default_factory=dict)

    @property
    def cut_ratio(self) -> float:
        """rho(T) = |T ∩ dOmega| / |T ∩ Omega|."""
        return self.gamma_length / self.area if self.area > 0 else np.inf


# ---------------------------------------------------------------------------
# small geometric helpers
# ---------------------------------------------------------------------------


def barycentric(verts: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of point(s) x with respect to triangle verts."""
    v0, v1, v2 = verts
    det = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    lx = ((v1[1] - v2[1]) * (x[..., 0] - v2[0]) + (v2[0] - v1[0]) * (x[..., 1] - v2[1])) / det
    ly = ((v2[1] - v0[1]) * (x[..., 0] - v2[0]) + (v0[0] - v2[0]) * (x[..., 1] - v2[1])) / det
    return np.stack([lx, ly, 1.0 - lx - ly], axis=-1)


def in_triangle(verts: np.ndarray, x: np.ndarray, tol: float = 1e-12) -> bool:
    lam = barycentric(verts, np.asarray(x))
    return bool(np.all(lam >= -tol))


def triangle_area(tri: np.ndarray) -> float:
    return abs(
        (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
        - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])
    ) / 2.0


def dist_point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def seg_disk_roots(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float):
    """Parameters t in [0,1] where segment a->b meets the circle |x-c| = R.

    Stable citardauq quadratic solve; returns a sorted array of roots
    (possibly empty).
    """
    e = b - a
    f = a - center
    A = float(e @ e)
    if A == 0.0:
        return np.zeros(0)
    B = float(e @ f)
    C = float(f @ f) - radius * radius
    disc = B * B - A * C
    if disc <= 0.0:
        return np.zeros(0)
    s = np.sqrt(disc)
    q = -(B + s) if B >= 0.0 else -(B - s)
    t1 = q / A
    t2 = C / q if q != 0.0 else (-B + s) / A
    roots = [min(max(t, 0.0), 1.0) for t in (t1, t2) if -1e-12 <= t <= 1.0 + 1e-12]
    return np.array(sorted(roots))


def wrap_angle(x: float) -> float:
    return x % _TWO_PI


def signed_sweep(start: float, end: float, ccw: bool) -> float:
    delta = wrap_angle(end - start)
    return delta if ccw else delta - _TWO_PI


def arc_point(center: np.ndarray, radius: float, ang: float) -> np.ndarray:
    return center + radius * np.array([np.cos(ang), np.sin(ang)])


def green_line(a: np.ndarray, b: np.ndarray) -> float:
    """Contribution of directed segment a->b to the loop area integral."""
    return 0.5 * (a[0] * b[1] - b[0] * a[1])


def green_arc(center: np.ndarray, radius: float, a0: float, a1: float, ccw: bool) -> float:
    """Closed-form (1/2)∮(x dy - y dx) along a circular arc.

    Parametrize x(theta) = c + R (cos theta, sin theta); integrating,
        (R/2) [ cx (sin a1 - sin a0) - cy (cos a1 - cos a0) + R (a1 - a0) ]
    with the signed sweep (a1 - a0) taken in the indicated direction.
    """
    sweep = signed_sweep(a0, a1, ccw)
    sa, sb = np.sin(a0), np.sin(a1)
    ca, cb = np.cos(a0), np.cos(a1)
    return 0.5 * radius * (
        center[0] * (sb - sa) - center[1] * (cb - ca) + radius * sweep
    )


def arc_length(radius: float, a0: float, a1: float, ccw: bool) -> float:
    return abs(signed_sweep(a0, a1, ccw)) * radius


def bbox_aspect(points: np.ndarray) -> float:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = np.maximum(hi - lo, 1e-300)
    return float(extent.max() / extent.min())


# ---------------------------------------------------------------------------
# exact circle clipping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Crossing:
    edge: int
    t: float
    point: np.ndarray
    entering: bool  # True if d goes (- -> +) along CCW traversal


@dataclass
class _Loop:
    """CCW boundary of tri ∩ disk as alternating chain runs and arcs."""

    chains: list[list[np.ndarray]]  # runs along the triangle boundary
    arcs: list[tuple[float, float, float, bool]]  # (cx, cy packed? no:) center, R, a0, a1, ccw

    def __post_init__(self):
        pass

    # arcs stored as (center_tuple, radius, a0, a1, ccw)
    def area(self) -> float:
        total = 0.0
        for chain in self.chains:
            for i in range(len(chain) - 1):
                total += green_line(chain[i], chain[i + 1])
        for c, r, a0, a1, ccw in self.arcs:
            total += green_arc(np.asarray(c), r, a0, a1, ccw)
        return total

    def gamma_length(self) -> float:
        return sum(arc_length(r, a0, a1, ccw) for _, r, a0, a1, ccw in self.arcs)


def _crossings(tri: np.ndarray, center: np.ndarray, radius: float) -> list[_Crossing]:
    out = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        e = b - a
        for t in seg_disk_roots(a, b, center, radius):
            p = a + t * e
            dd_dt = float(-2.0 * ((a + t * e) - center) @ e)
            out.append(_Crossing(i, float(t), p, dd_dt > 0.0))
    out.sort(key=lambda c: (c.edge, c.t))
    return out


def _build_loop(
    tri: np.ndarray, center: np.ndarray, radius: float, crossings: list[_Crossing]
) -> _Loop:
    """Pair each exit with the next entering crossing (cyclically)."""
    if not crossings:
        raise RuntimeError("loop requested without crossings")
    n_ev = len(crossings)

    def next_index(idx: int, want_entering: bool) -> int:
        for step in range(1, n_ev + 1):
            j = (idx + step) % n_ev
            if crossings[j].entering == want_entering:
                return j
        raise RuntimeError(f"no {'entering' if want_entering else 'exiting'} partner")

    def walk(from_x: _Crossing, to_x: _Crossing) -> list[np.ndarray]:
        """Inside polyline along partial-T boundary from from_x to to_x.

        Valid because between an entering crossing and the next exiting one
        the signed distance is positive along the whole walked stretch.
        """
        pts = [from_x.point]
        if from_x.edge == to_x.edge:
            pts.append(to_x.point)
            return pts
        pts.append(tri[(from_x.edge + 1) % 3].copy())
        e = (from_x.edge + 1) % 3
        while e != to_x.edge:
            pts.append(tri[(e + 1) % 3].copy())
            e = (e + 1) % 3
        pts.append(to_x.point)
        return pts

    def select_arc(p_from: np.ndarray, p_to: np.ndarray):
        """Arc from p_from to p_to lying inside tri.

        Several interior samples per candidate direction are tested so major
        arcs are selected correctly (a single midpoint test cannot
        distinguish them).
        """
        a_from = float(np.arctan2(p_from[1] - center[1], p_from[0] - center[0]))
        a_to = float(np.arctan2(p_to[1] - center[1], p_to[0] - center[0]))
        frac = np.linspace(0.0, 1.0, 9)[1:-1]
        best = None
        for ccw in (True, False):
            a1 = a_from + signed_sweep(a_from, a_to, ccw)
            mids = a_from + frac * signed_sweep(a_from, a_to, ccw)
            samples = [in_triangle(tri, arc_point(center, radius, ang), tol=1e-9)
                       for ang in mids]
            n_in = sum(samples)
            if n_in == len(samples):
                return (tuple(center), radius, a_from, a1, ccw)
            if best is None or n_in > best[0]:
                best = (n_in, a_from, a1, ccw)
        if best is not None and best[0] > 0:
            return (tuple(center), radius, best[1], best[2], best[3])
        raise RuntimeError("no circular arc between crossings lies inside the element")

    loops_chains: list[list[np.ndarray]] = []
    loops_arcs: list[tuple] = []
    for idx, cr in enumerate(crossings):
        if cr.entering:
            nxt = next_index(idx, want_entering=False)
            loops_chains.append(walk(cr, crossings[nxt]))
        else:
            nxt = next_index(idx, want_entering=True)
            loops_arcs.append(select_arc(cr.point, crossings[nxt].point))
    return _Loop(loops_chains, loops_arcs)


def clip_circle(
    tri: np.ndarray,
    circ: Circle,
    tol_rel: float = 1e-13,
    vol_order: int = 10,
    bnd_order: int = 12,
) -> CutRegion:
    """Exact quadrature for tri ∩ disk."""
    tri = np.asarray(tri, dtype=float)
    if (
        (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
        - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])
    ) < 0.0:
        tri = tri[::-1].copy()  # normalize to counter-clockwise
    center = np.asarray(circ.center, dtype=float)
    radius = circ.radius
    method = "circle-exact"

    d = np.array([radius - np.linalg.norm(v - center) for v in tri])
    scale = max(radius * radius, 1e-300)

    # Crossings decide first: a disk dipping through one edge while every
    # vertex sits outside is exactly the canonical sliver, not an empty cut.
    crossings = _crossings(tri, center, radius)

    if np.all(d >= -1e-13 * scale):
        ref, wref = gauss_triangle(vol_order)
        phys, _, det = affine_map_to_physical(ref, tri)
        return CutRegion(
            status="full", area=abs(det) / 2.0, gamma_length=0.0,
            pts=phys, wts=wref * abs(det),
            bnd_pts=np.zeros((0, 2)), bnd_wts=np.zeros(0), bnd_nrm=np.zeros((0, 2)),
            bbox_aspect=bbox_aspect(tri), curvature_mean=0.0, method=method,
        )

    if not crossings:
        # No boundary contact at all: either the whole disk lies inside the
        # element, or the element misses it entirely.
        touches = any(
            dist_point_segment(center, tri[i], tri[(i + 1) % 3]) < radius * (1.0 - 1e-12)
            for i in range(3)
        )
        if not touches and in_triangle(tri, center, tol=-1e-12):
            pts, wts, bpts, bwts, bnrm = _full_disk_rule(circ, vol_order, bnd_order)
            return CutRegion(
                status="cut", area=np.pi * radius**2, gamma_length=_TWO_PI * radius,
                pts=pts, wts=wts, bnd_pts=bpts, bnd_wts=bwts, bnd_nrm=bnrm,
                bbox_aspect=1.0, curvature_mean=1.0 / radius, method=method,
                meta={"n_arcs": 1,
                      "arcs": [(float(center[0]), float(center[1]), float(radius),
                                0.0, _TWO_PI, True)]},
            )
        z2 = np.zeros((0, 2))
        return CutRegion("empty", 0.0, 0.0, z2, np.zeros(0), z2, np.zeros(0),
                         z2, np.nan, 0.0, method)

    loop = _build_loop(tri, center, radius, crossings)
    area_exact = loop.area()
    if area_exact <= 0.0:
        raise RuntimeError("non-positive exact area; loop orientation broken")
    gamma_exact = loop.gamma_length()

    pts, wts = _interior_rule(loop, area_exact, tol_rel, vol_order)
    bpts, bwts, bnrm = _boundary_rule(loop, gamma_exact, bnd_order)

    aspect_pts = [p for chain in loop.chains for p in chain]
    for c, r, a0, a1, ccw in loop.arcs:
        aspect_pts.extend(_axis_extrema_on_arc(np.asarray(c), r, a0, a1, ccw))

    return CutRegion(
        status="cut", area=float(area_exact), gamma_length=float(gamma_exact),
        pts=pts, wts=wts, bnd_pts=bpts, bnd_wts=bwts, bnd_nrm=bnrm,
        bbox_aspect=bbox_aspect(np.asarray(aspect_pts)),
        curvature_mean=1.0 / radius, method=method,
        meta={
            "n_arcs": len(loop.arcs),
            # flat (center_x, center_y, radius, a0, a1, ccw) per arc
            "arcs": [(c[0], c[1], rr, aa0, aa1, cccw)
                     for c, rr, aa0, aa1, cccw in loop.arcs],
        },
    )


def _interior_rule(
    loop: _Loop, area_exact: float, tol_rel: float, vol_order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Polygon-fan Gauss rule plus circular-segment crescent Gauss strips.

    The polygonal part closes each arc gap with the arc's single straight
    chord; the residual crescent between that chord and the arc is integrated
    in the (theta, rho) frame about the disk center with rho bounded below by
    the SAME full chord, so fan and strips partition the region exactly — no
    overlap, no gap. Angular Gauss panels exist purely for quadrature order.
    Boundary arcs of a convex cut subtend less than pi (asserted).
    """
    poly_pts: list[np.ndarray] = []
    for chain in loop.chains:
        poly_pts.extend(p for p in chain)
    poly = np.asarray(poly_pts)
    seed = poly.mean(axis=0)

    ref, wref = gauss_triangle(vol_order)
    m = len(poly)
    pts_list, wts_list = [], []
    for i in range(m):
        sub = np.array([seed, poly[i], poly[(i + 1) % m]])
        phys, _, det = affine_map_to_physical(ref, sub)
        pts_list.append(phys)
        wts_list.append(wref * abs(det))
    fan_pts = np.concatenate(pts_list)
    fan_wts = np.concatenate(wts_list)
    fan_area = float(fan_wts.sum())

    def strip_rule(n_strip: int):
        xi, wxi = gauss_interval(n_strip)
        eta, weta = gauss_interval(n_strip)
        WTH, WETA = np.meshgrid(wxi, weta, indexing="ij")
        spts, swts = [], []
        for c, r, a0, a1, ccw in loop.arcs:
            carr = np.asarray(c)
            sweep = signed_sweep(a0, a1, ccw)
            if abs(sweep) >= np.pi:
                raise RuntimeError(
                    "boundary arc subtends >= pi; not a convex cut configuration"
                )
            pa = arc_point(carr, r, a0)
            pb = arc_point(carr, r, a1)
            chord = pb - pa
            nrm = np.array([chord[1], -chord[0]])
            nrm /= np.linalg.norm(nrm)
            fc = float((carr - pa) @ nrm)
            if fc > 0.0:  # orient normal away from center (segment side)
                nrm, fc = -nrm, -fc
            n_panels = max(1, int(np.ceil(abs(sweep) / 0.5)))
            panel_edges = a0 + sweep * np.linspace(0.0, 1.0, n_panels + 1)
            for ca, cb in zip(panel_edges[:-1], panel_edges[1:]):
                th = ca + (cb - ca) * xi
                dirx, diry = np.cos(th), np.sin(th)
                denom = dirx * nrm[0] + diry * nrm[1]
                denom = np.where(np.abs(denom) < 1e-14, 1e-14 * np.sign(denom + 1e-300), denom)
                rho_lo = np.clip(-fc / denom, 0.0, r)
                TH, ETA = np.meshgrid(th, eta, indexing="ij")
                RLO = np.broadcast_to(rho_lo[:, None], TH.shape)
                rho = RLO + ETA * (r - RLO)
                ww = WTH * WETA * rho * ((cb - ca) * (r - RLO))
                X = carr[0] + rho * np.cos(TH)
                Y = carr[1] + rho * np.sin(TH)
                spts.append(np.stack([X.ravel(), Y.ravel()], axis=1))
                swts.append(ww.ravel())
        sp = np.concatenate(spts)
        sw = np.concatenate(swts)
        return sp, sw, fan_area + float(sw.sum())

    strip_pts = np.zeros((0, 2))
    strip_wts = np.zeros(0)
    if loop.arcs:
        for n_strip in (10, 14, 18, 24, 32):
            strip_pts, strip_wts, q_total = strip_rule(n_strip)
            if abs(q_total - area_exact) <= tol_rel * max(area_exact, 1e-300):
                break

    all_pts = np.concatenate([fan_pts, strip_pts])
    all_wts = np.concatenate([fan_wts, strip_wts])
    q = float(all_wts.sum())
    if q > 0.0:
        all_wts *= area_exact / q
    return all_pts, all_wts


def _boundary_rule(
    loop: _Loop, gamma_exact: float, bnd_order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite Gauss rule along all physical-boundary arcs."""
    nodes, weights = leggauss(bnd_order)
    pts_l, wts_l, nrm_l = [], [], []
    for c, r, a0, a1, ccw in loop.arcs:
        carr = np.asarray(c)
        sweep = signed_sweep(a0, a1, ccw)
        n_chunks = max(1, int(np.ceil(abs(sweep) / 0.5)))
        edges = a0 + sweep * np.linspace(0.0, 1.0, n_chunks + 1)
        for ca, cb in zip(edges[:-1], edges[1:]):
            half, mid = 0.5 * (cb - ca), 0.5 * (ca + cb)
            th = mid + half * nodes
            w = half * weights * r
            p = carr[None, :] + r * np.stack([np.cos(th), np.sin(th)], axis=1)
            n = np.stack([np.cos(th), np.sin(th)], axis=1)
            pts_l.append(p)
            wts_l.append(w)
            nrm_l.append(n)
    bpts = np.concatenate(pts_l)
    bwts = np.concatenate(wts_l)
    bnrm = np.concatenate(nrm_l)
    q = float(bwts.sum())
    if q > 0.0:
        bwts *= gamma_exact / q
    return bpts, bwts, bnrm


def _axis_extrema_on_arc(center, radius, a0, a1, ccw):
    """Points of the arc with extremal x/y coordinates (bbox candidates)."""
    sweep = signed_sweep(a0, a1, ccw)
    base = a0 if sweep >= 0 else a1
    mag = abs(sweep)
    out = []
    for target in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
        delta = wrap_angle(target - base)
        t = delta / mag if mag > 0 else 0.0
        if -1e-12 <= t <= 1.0 + 1e-12:
            ang = base + t * mag
            out.append(arc_point(center, radius, ang))
    return out


def _full_disk_rule(circ: Circle, vol_order: int, bnd_order: int):
    center = np.asarray(circ.center, dtype=float)
    radius = circ.radius
    rho_n, rho_w = gauss_interval(vol_order)
    th_n, th_w = gauss_interval(2 * vol_order)
    RH, TH = np.meshgrid(rho_n, th_n, indexing="ij")
    WR, WT = np.meshgrid(rho_w, th_w, indexing="ij")
    rho = radius * RH.ravel()
    th = _TWO_PI * TH.ravel()
    w = WR.ravel() * WT.ravel() * (_TWO_PI * radius) * rho
    pts = center[None, :] + np.stack([rho * np.cos(th), rho * np.sin(th)], axis=1)

    tb, wb = gauss_interval(bnd_order * 4)
    thb = _TWO_PI * tb
    bpts = center[None, :] + radius * np.stack([np.cos(thb), np.sin(thb)], axis=1)
    bwts = radius * _TWO_PI * wb
    bnrm = np.stack([np.cos(thb), np.sin(thb)], axis=1)
    return pts, w, bpts, bwts, bnrm


# ---------------------------------------------------------------------------
# exact ellipse clipping via affine reduction to the circle
# ---------------------------------------------------------------------------


def clip_ellipse(
    tri: np.ndarray,
    ell: Ellipse,
    tol_rel: float = 1e-13,
    vol_order: int = 10,
    bnd_order: int = 12,
) -> CutRegion:
    """Exact quadrature for tri ∩ ellipse by affine map to the unit disk.

    The map y = D^{-1}(x - c) sends the ellipse to the unit disk; weights pick
    up |det D|, normals pull back through D^{-T}, and the physical-boundary
    measure is integrated directly in the ellipse parametrization.
    """
    tri = np.asarray(tri, dtype=float)
    ax, ay = ell.semi_axes
    c = np.asarray(ell.center, dtype=float)
    det_d = ax * ay

    mapped_tri = (tri - c) / np.array([ax, ay])
    unit_disk = Circle(center=(0.0, 0.0), radius=1.0)
    region = clip_circle(mapped_tri, unit_disk, tol_rel=tol_rel,
                         vol_order=vol_order, bnd_order=bnd_order)
    if region.status == "empty":
        z2 = np.zeros((0, 2))
        return CutRegion("empty", 0.0, 0.0, z2, np.zeros(0), z2, np.zeros(0),
                         z2, np.nan, 0.0, "ellipse-affine")

    phys_pts = c[None, :] + region.pts * np.array([ax, ay])

    # Rebuild the physical-boundary rule per node: elliptical arclength speed
    # varies along the arc, so weights cannot be obtained by rescaling the
    # circle-path weights with one global factor.
    nodes, weights = leggauss(bnd_order)
    bpts_l, bwts_l, bnrm_l = [], [], []
    for cx, cy, _, a0, a1, ccw in region.meta.get("arcs", []):
        sweep = signed_sweep(a0, a1, ccw)
        n_chunks = max(1, int(np.ceil(abs(sweep) / 0.5)))
        edges_ang = a0 + sweep * np.linspace(0.0, 1.0, n_chunks + 1)
        for ca, cb in zip(edges_ang[:-1], edges_ang[1:]):
            half, mid = 0.5 * (cb - ca), 0.5 * (ca + cb)
            th = mid + half * nodes
            w = half * weights
            x = cx + ax * np.cos(th)
            y = cy + ay * np.sin(th)
            speed = np.sqrt((ax * np.sin(th)) ** 2 + (ay * np.cos(th)) ** 2)
            n_out = np.stack([np.cos(th) / ax, np.sin(th) / ay], axis=1)
            n_out /= np.linalg.norm(n_out, axis=1, keepdims=True)
            bpts_l.append(np.stack([x, y], axis=1))
            bwts_l.append(w * speed)
            bnrm_l.append(n_out)
    if bpts_l:
        phys_bpts = np.concatenate(bpts_l)
        bw = np.concatenate(bwts_l)
        bnrm = np.concatenate(bnrm_l)
        gamma_length = float(bw.sum())
    else:
        phys_bpts = np.zeros((0, 2))
        bw = np.zeros(0)
        bnrm = np.zeros((0, 2))
        gamma_length = 0.0

    area = float(region.area * det_d)
    curv = 0.0
    if len(phys_bpts) and gamma_length > 0.0:
        kv = np.abs(ell.curvature(phys_bpts))
        curv = float(np.average(kv, weights=bw))

    stacked = np.concatenate([p for p in (phys_pts, phys_bpts) if len(p)]) \
        if (len(phys_pts) or len(phys_bpts)) else tri

    return CutRegion(
        status=region.status, area=area, gamma_length=float(gamma_length),
        pts=phys_pts, wts=region.wts * det_d,
        bnd_pts=phys_bpts, bnd_wts=bw, bnd_nrm=bnrm,
        bbox_aspect=bbox_aspect(stacked), curvature_mean=curv,
        method="ellipse-affine",
        meta={"n_arcs": region.meta.get("n_arcs", 0),
              "self_check_rel": region.meta.get("self_check_rel")},
    )


# ---------------------------------------------------------------------------
# general smooth level sets: adaptive subdivision with linearized clipping
# ---------------------------------------------------------------------------


def clip_subdivision(
    tri: np.ndarray,
    levelset,
    lin_tol: float = 1e-3,
    max_depth: int = 40,
    vol_order: int = 10,
    bnd_order: int = 12,
    verify: bool = True,
) -> CutRegion:
    """Cut region for a general level set by adaptive midpoint refinement.

    An element stops subdividing once the interface is resolved well enough
    that the tangent-line approximation introduces negligible area error:
    the criterion is ``|kappa| * diameter <= lin_tol`` (error of the
    linearization scales like kappa * diam^2 per unit length of interface).
    When ``verify`` is set, a deeper pass (tighter criterion, deeper cap)
    re-estimates the area and the relative discrepancy is recorded in
    ``meta['self_check_rel']`` — a measured bound, never an assumption.
    """
    result = _subdivide_pass(tri, levelset, lin_tol, max_depth, vol_order, bnd_order)
    if verify and result.status == "cut":
        deeper = _subdivide_pass(tri, levelset, lin_tol * 0.25, max_depth + 4,
                                 vol_order, bnd_order)
        rel = abs(deeper.area - result.area) / max(result.area, 1e-300)
        result.meta["self_check_rel"] = float(rel)
    return result


def _subdivide_pass(tri, levelset, lin_tol, max_depth, vol_order, bnd_order) -> CutRegion:
    tri = np.asarray(tri, dtype=float)
    if (
        (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
        - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])
    ) < 0.0:
        tri = tri[::-1].copy()  # normalize to counter-clockwise
    pts_acc, wts_acc = [], []
    bpts_acc, bwts_acc, bnrm_acc = [], [], []

    def process(elem, depth):
        vals = levelset.phi(elem)
        if np.all(vals >= 0.0):
            ref, wref = gauss_triangle(vol_order)
            phys, _, det = affine_map_to_physical(ref, elem)
            pts_acc.append(phys)
            wts_acc.append(wref * abs(det))
            return
        # Vertex signs alone cannot detect an interface passing strictly
        # through the element interior with all vertices outside; classify by
        # proximity of the centroid to the interface before terminating.
        c = elem.mean(axis=0)
        phi_c = float(levelset.phi(c[None, :])[0])
        g = levelset.grad(c[None, :])[0]
        gn = float(np.linalg.norm(g))
        diam = 2.0 * float(np.max(np.linalg.norm(elem - c, axis=1)))
        near_interface = abs(phi_c) <= gn * diam
        if not near_interface and phi_c < 0.0:
            return
        kappa_here = float(np.abs(levelset.curvature(c[None, :])[0]))
        if depth >= max_depth or (np.isfinite(kappa_here) and kappa_here * diam <= lin_tol):
            if near_interface:
                _linearized_leaf(elem, levelset, vol_order, bnd_order,
                                 pts_acc, wts_acc, bpts_acc, bwts_acc, bnrm_acc)
            elif phi_c > 0.0:
                ref, wref = gauss_triangle(vol_order)
                phys, _, det = affine_map_to_physical(ref, elem)
                pts_acc.append(phys)
                wts_acc.append(wref * abs(det))
            return
        mid01, mid12, mid20 = (0.5 * (elem[0] + elem[1]),
                               0.5 * (elem[1] + elem[2]),
                               0.5 * (elem[2] + elem[0]))
        for sub in (np.array([elem[0], mid01, mid20]),
                    np.array([mid01, elem[1], mid12]),
                    np.array([mid20, mid12, elem[2]]),
                    np.array([mid01, mid12, mid20])):
            process(sub, depth + 1)

    process(np.asarray(tri, dtype=float), 0)

    pts = np.concatenate(pts_acc) if pts_acc else np.zeros((0, 2))
    wts = np.concatenate(wts_acc) if wts_acc else np.zeros(0)
    area = float(wts.sum())
    bpts = np.concatenate(bpts_acc) if bpts_acc else np.zeros((0, 2))
    bwts = np.concatenate(bwts_acc) if bwts_acc else np.zeros(0)
    bnrm = np.concatenate(bnrm_acc) if bnrm_acc else np.zeros((0, 2))
    gamma_len = float(bwts.sum())

    curv = 0.0
    if len(bpts) and gamma_len > 0:
        curv = float(np.abs(levelset.curvature(bpts)).mean())

    return CutRegion(
        status="cut" if area > 0 else "empty",
        area=area, gamma_length=gamma_len,
        pts=pts, wts=wts, bnd_pts=bpts, bnd_wts=bwts, bnd_nrm=bnrm,
        bbox_aspect=bbox_aspect(pts) if len(pts) else np.nan,
        curvature_mean=curv,
        method=f"subdivision(lin_tol={lin_tol:g})",
    )


def _linearized_leaf(elem, levelset, vol_order, bnd_order,
                     pts_acc, wts_acc, bpts_acc, bwts_acc, bnrm_acc):
    """Clip a mixed leaf against the tangent half-plane at its centroid."""
    c = elem.mean(axis=0)
    phi_c = float(levelset.phi(c[None, :])[0])
    g = levelset.grad(c[None, :])[0]
    gn = float(np.linalg.norm(g))
    if gn == 0.0:
        return
    vals = phi_c + (elem - c) @ g
    inside = vals >= 0.0
    poly: list[np.ndarray] = []
    for i in range(3):
        j = (i + 1) % 3
        if inside[i]:
            poly.append(elem[i])
        if inside[i] != inside[j]:
            t = vals[i] / (vals[i] - vals[j])
            poly.append(elem[i] + t * (elem[j] - elem[i]))
    if len(poly) < 3:
        return
    poly_arr = np.asarray(poly)
    seed = poly_arr.mean(axis=0)
    ref, wref = gauss_triangle(vol_order)
    m = len(poly_arr)
    for i in range(m):
        sub = np.array([seed, poly_arr[i], poly_arr[(i + 1) % m]])
        phys, _, det = affine_map_to_physical(ref, sub)
        pts_acc.append(phys)
        wts_acc.append(wref * abs(det))
    # Physical boundary Gamma_T is the linearized INTERFACE: its segments lie
    # strictly interior to the element. Runs along the element's own edges are
    # internal mesh faces and contribute nothing to Gamma. The midpoint of an
    # interface segment sits off the element boundary; tolerance scales with
    # element size so deep refinement levels stay decisive.
    diam = float(max(np.linalg.norm(elem[(i + 1) % 3] - elem[i]) for i in range(3)))
    bnd_tol = 1e-8 * diam
    for i in range(m):
        a, b = poly_arr[i], poly_arr[(i + 1) % m]
        mid_on_edge = (_on_elem_boundary(0.5 * (a + b), elem, bnd_tol))
        if mid_on_edge:
            continue
        seg_len = float(np.linalg.norm(b - a))
        if seg_len == 0.0:
            continue
        nodes, weights = leggauss(bnd_order)
        ts = 0.5 + 0.5 * nodes
        ws = 0.5 * weights * seg_len
        bp = a[None, :] + ts[:, None] * (b - a)[None, :]
        gr = levelset.grad(bp)
        gn = np.linalg.norm(gr, axis=1, keepdims=True)
        n_out = -gr / np.maximum(gn, 1e-300)
        bpts_acc.append(bp)
        bwts_acc.append(ws)
        bnrm_acc.append(n_out)


def _on_elem_boundary(x, elem, tol) -> bool:
    return any(dist_point_segment(x, elem[i], elem[(i + 1) % 3]) <= tol
               for i in range(3))


# ---------------------------------------------------------------------------
# public dispatcher
# ---------------------------------------------------------------------------


def cut_cell(levelset, verts: np.ndarray, **kwargs) -> CutRegion:
    """Dispatch by level-set type: exact circle, exact ellipse, or fallback."""
    if isinstance(levelset, Circle):
        return clip_circle(np.asarray(verts, dtype=float), levelset, **kwargs)
    if isinstance(levelset, Ellipse):
        return clip_ellipse(np.asarray(verts, dtype=float), levelset, **kwargs)
    return clip_subdivision(np.asarray(verts, dtype=float), levelset, **kwargs)


# ---------------------------------------------------------------------------
# independent reference (verification suite only)
# ---------------------------------------------------------------------------


def ray_fire_reference(
    tri: np.ndarray,
    center,
    radius: float,
    n_panels: int = 1024,
    order: int = 20,
    tol_rel: float = 1e-13,
    max_panels: int = 131072,
) -> dict[str, float]:
    """Independent triangle∩disk area/moment reference by polar ray firing.

    Uses only ray-segment intersections and exact radial antiderivatives —
    no code shared with the clipping paths above. Angular integration is
    composite Gauss-Legendre whose panel count grows until the area estimate
    stabilizes to ``tol_rel`` relative error (rays grazing element edges
    introduce mild kinks that coarse rules under-resolve). Moments returned
    about the disk center: ∫x dA, ∫y dA, ∫x² dA, ∫xy dA, ∫y² dA.
    """
    prev_area = None
    while True:
        result = _ray_fire_pass(tri, center, radius, n_panels, order)
        if prev_area is not None and abs(result["area"] - prev_area) <= tol_rel * max(
            abs(prev_area), 1e-300
        ):
            return result
        if n_panels >= max_panels:
            return result
        prev_area = result["area"]
        n_panels = min(4 * n_panels, max_panels)


def polygon_disk_reference(
    tri: np.ndarray,
    center,
    radius: float,
    n_init: int = 16384,
    tol_rel: float = 1e-13,
    max_n: int = 2_097_152,
) -> dict[str, float]:
    """Triangle∩disk area/moment reference via polygon approximation.

    The disk is replaced by an N-gon whose vertices sit at arc *midpoints*
    (the symmetric shift raises the area/moment convergence order relative to
    an inscribed polygon), clipped to the triangle by Sutherland–Hodgman, and
    integrated with exact shoelace-type polygon formulas. N grows until all
    reported quantities stabilize to ``tol_rel`` relative error. Independent
    of every code path in the clipping machinery above.
    """
    tri = np.asarray(tri, dtype=float)
    c = np.asarray(center, dtype=float)
    n = n_init
    prev = None
    while True:
        result = _polygon_pass(tri, c, radius, n)
        if prev is not None and all(
            abs(result[k] - prev[k]) <= tol_rel * max(abs(prev[k]), 1e-300)
            or abs(result[k]) < 1e-12 * max(result["area"], 1e-300)
            for k in result
        ):
            return result
        if n >= max_n:
            return result
        prev = result
        n *= 4


def _polygon_pass(tri, c, radius, n: int) -> dict[str, float]:
    k = np.arange(n)
    th = (k + 0.5) * (_TWO_PI / n)
    subject = np.stack([c[0] + radius * np.cos(th), c[1] + radius * np.sin(th)], axis=1)

    # Sutherland–Hodgman clip against each directed triangle edge, keeping
    # vertex order (required by the shoelace formulas below).
    for i in range(3):
        if len(subject) == 0:
            break
        a = tri[i]
        e = tri[(i + 1) % 3] - tri[i]
        nxt = np.roll(subject, -1, axis=0)
        # interior of a CCW triangle lies to the LEFT of each directed edge:
        # keep points with cross(e, p - a) >= 0
        side = e[0] * (subject[:, 1] - a[1]) - e[1] * (subject[:, 0] - a[0])
        side_n = e[0] * (nxt[:, 1] - a[1]) - e[1] * (nxt[:, 0] - a[0])
        inp = side >= 0.0
        inp_n = side_n >= 0.0
        denom = side - side_n
        safe = np.where(np.abs(denom) < 1e-300, 1.0, denom)
        tpar = np.where(np.abs(denom) < 1e-300, 0.0, side / safe)
        inter = subject + tpar[:, None] * (nxt - subject)

        seg_out = np.where(inp, 1, 0) + np.where(inp != inp_n, 1, 0)  # pts per segment
        total = int(seg_out.sum())
        if total == 0:
            subject = np.zeros((0, 2))
            continue
        out = np.empty((total, 2))
        pos = np.concatenate([[0], np.cumsum(seg_out)[:-1]])
        idx = np.arange(len(subject))
        m_in = inp
        m_cr = inp != inp_n
        out[pos[m_in]] = subject[m_in]
        out[pos[m_cr] + np.where(m_in[m_cr], 1, 0)] = inter[m_cr]
        subject = out

    if len(subject) < 3:
        return {k2: 0.0 for k2 in ("area", "mx", "my", "x2", "xy", "y2")}

    x, y = subject[:, 0], subject[:, 1]
    x2, y2n = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2n - x2 * y
    area = 0.5 * float(np.sum(cross))
    sgn = 1.0 if area >= 0.0 else -1.0
    mx = (1.0 / 6.0) * float(np.sum((x + x2) * cross))
    my = (1.0 / 6.0) * float(np.sum((y + y2n) * cross))
    xxi = (1.0 / 12.0) * float(np.sum((x * x + x * x2 + x2 * x2) * cross))
    yyi = (1.0 / 12.0) * float(np.sum((y * y + y * y2n + y2n * y2n) * cross))
    xyi = (1.0 / 24.0) * float(np.sum(
        (2.0 * x * y + x * y2n + x2 * y + 2.0 * x2 * y2n) * cross))
    return {"area": sgn * area, "mx": sgn * mx, "my": sgn * my,
            "x2": sgn * xxi, "xy": sgn * xyi, "y2": sgn * yyi}


def _ray_fire_pass(tri, center, radius, n_panels: int, order: int) -> dict[str, float]:
    """One angular resolution of the polar integration.

    Ray P(u) = c + u (cos th, sin th); edge Q(v) = a + v (b - a). Solving the
    2x2 system u d - v e = a - c by Cramer's rule:

        u = (b1 e0 - b0 e1) / (d0(-e1) - (-e0) d1),
        v = (d0 b1 - d1 b0) / (same denominator),

    with d = (cos th, sin th), b = a - c. The radial integral of x^p y^q over
    [r_in, r_out] at angle th reduces to exact antiderivatives in r.
    """
    tri = np.asarray(tri, dtype=float)
    c = np.asarray(center, dtype=float)
    nodes, weights = leggauss(order)
    edges = np.linspace(0.0, _TWO_PI, n_panels + 1)
    halves = 0.5 * np.diff(edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    th = (mids[:, None] + halves[:, None] * nodes[None, :]).ravel()
    wth = (halves[:, None] * weights[None, :]).ravel()
    ct, st = np.cos(th), np.sin(th)

    u_enter = np.full_like(th, np.inf)
    u_exit = np.full_like(th, -np.inf)
    for i in range(3):
        ax, ay = tri[i]
        ex, ey = tri[(i + 1) % 3] - tri[i]
        bx, by = ax - c[0], ay - c[1]
        denom = st * ex - ct * ey
        parallel = np.abs(denom) < 1e-14
        den_safe = np.where(parallel, 1.0, denom)
        v = (ct * by - st * bx) / den_safe
        u = (by * ex - bx * ey) / den_safe
        ok = (~parallel) & (v >= -1e-12) & (v <= 1.0 + 1e-12)
        u_enter = np.minimum(u_enter, np.where(ok, u, np.inf))
        u_exit = np.maximum(u_exit, np.where(ok, u, -np.inf))

    r_in = np.maximum(u_enter, 0.0)
    r_out = np.minimum(u_exit, radius)
    valid = (r_out > r_in) & np.isfinite(r_in)
    r_in = np.where(valid, r_in, 0.0)
    r_out = np.where(valid, r_out, 0.0)

    dr2 = r_out**2 - r_in**2
    dr3 = r_out**3 - r_in**3
    dr4 = r_out**4 - r_in**4
    return {
        "area": float(np.sum(wth * 0.5 * dr2)),
        "mx": float(np.sum(wth * (1.0 / 3.0) * dr3 * ct)),
        "my": float(np.sum(wth * (1.0 / 3.0) * dr3 * st)),
        "x2": float(np.sum(wth * 0.25 * dr4 * ct**2)),
        "xy": float(np.sum(wth * 0.25 * dr4 * ct * st)),
        "y2": float(np.sum(wth * 0.25 * dr4 * st**2)),
    }
