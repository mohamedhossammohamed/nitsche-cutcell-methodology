"""Stabilization-parameter formula registry (preregistered F1-F8).

Every candidate has the form

    lambda_T = C_k * Psi(cell) / h_T,     C_k = 4 (k+1)^2,

with Psi a closed-form function of the local cut geometry. The fitted family
F8 keeps the harmonic shape of F4 and adjusts (C_k, rho_cap); its parameters
are fitted on the Tier-2 training split only and frozen thereafter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Callable


def C_k(k: int) -> float:
    return 4.0 * (k + 1) ** 2


@dataclass(frozen=True)
class CellGeometry:
    """Local regressors x(T) entering the stabilization formula."""

    rho: float  # |T ∩ dOmega| / |T ∩ Omega|
    h_T: float  # element diameter
    k: int  # polynomial degree
    kappa: float  # mean |curvature| of dOmega inside T
    aspect: float  # bounding-box aspect ratio of T ∩ Omega (>= 1)
    eps_cut: float  # |T ∩ Omega| / |T|


PsiFn = Callable[[CellGeometry], float]


def _baseline(c: CellGeometry) -> float:
    return c.rho


def _make_hard_clip(cap: float) -> PsiFn:
    return lambda c: min(c.rho, cap)


def _aspect_clip(c: CellGeometry) -> float:
    return min(c.rho, 10.0 * c.aspect)


def _make_harmonic(cap: float) -> PsiFn:
    return lambda c: c.rho * cap / (c.rho + cap)


def _log_damping(c: CellGeometry) -> float:
    return 1.0 + log1p(c.rho)


def _make_curvature(beta: float) -> PsiFn:
    return lambda c: c.rho * (1.0 + beta * c.h_T * c.kappa)


def stabilization_lambda(psi: PsiFn, cell: CellGeometry, c_k: float) -> float:
    return c_k * psi(cell) / cell.h_T


def _variant(pid: str, name: str, psi: PsiFn, params: dict) -> dict:
    return {"id": pid, "name": name, "psi": psi, "params": params}


def build_registry() -> dict[str, dict]:
    """The frozen preregistered candidate list, keyed by formula id."""
    reg: dict[str, dict] = {}
    reg["F1"] = _variant("F1", "baseline rho/h", _baseline, {})
    for tag, cap in zip("abcd", (10.0, 50.0, 100.0, 500.0)):
        reg[f"F2{tag}"] = _variant(
            f"F2{tag}", f"hard clip rho_cap={cap:g}",
            _make_hard_clip(cap), {"rho_cap": cap})
    reg["F3"] = _variant("F3", "aspect-ratio clip", _aspect_clip, {"rho0": 10.0})
    for tag, cap in zip("abcd", (10.0, 50.0, 100.0, 500.0)):
        reg[f"F4{tag}"] = _variant(
            f"F4{tag}", f"harmonic blend rho_cap={cap:g}",
            _make_harmonic(cap), {"rho_cap": cap})
    reg["F5"] = _variant("F5", "logarithmic damping", _log_damping, {})
    for tag, beta in zip("ab", (0.5, 1.0)):
        reg[f"F6{tag}"] = _variant(
            f"F6{tag}", f"curvature-augmented beta={beta:g}",
            _make_curvature(beta), {"beta": beta})
    # F7 shares F4c's Psi; the hybrid branch lives in the benchmark driver
    # (aggregation activates below eps_c).
    reg["F7"] = _variant("F7", "aggregation hybrid",
                         _make_harmonic(100.0),
                         {"rho_cap": 100.0, "eps_c": 1e-3})
    # F8: harmonic shape, (C_k, rho_cap) fitted on Tier-2 train; placeholders
    # until fitting assigns them (fit writes params back through the driver).
    reg["F8"] = _variant("F8", "fitted harmonic",
                         _make_harmonic(100.0),
                         {"rho_cap": 100.0, "c_k": None, "status": "unfitted"})
    return reg


CLOSED_FORM_IDS = [
    "F1",
    *[f"F2{t}" for t in "abcd"],
    "F3",
    *[f"F4{t}" for t in "abcd"],
    "F5",
    *["F6a", "F6b"],
]
