"""Structured background meshes with P1/P2 degrees of freedom."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TriMesh:
    """Triangulation of the unit square, two triangles per tensor cell.

    Every cell (i, j) splits along the lower-left -> upper-right diagonal;
    both triangles are counter-clockwise. ``cell_id`` maps each element to
    its structured cell index so benchmark drivers can target specific cells.
    """

    nodes: np.ndarray  # (nv, 2)
    elements: np.ndarray  # (ne, 3) CCW vertex indices
    cell_id: np.ndarray  # (ne,) structured cell index i + j * n
    n: int  # cells per axis
    edge_ids: np.ndarray  # (ne, 3) ids of edges opposite vertices 0, 1, 2
    edge_nodes: np.ndarray  # (n_edges, 2)
    element_adjacency: list[list[int]]  # edge-sharing neighbours

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    def diameters(self) -> np.ndarray:
        p = self.nodes[self.elements]
        return np.max(np.stack([
            np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
        ], axis=1), axis=1)

    def dof_count(self, k: int) -> int:
        return len(self.nodes) if k == 1 else len(self.nodes) + len(self.edge_nodes)

    def element_dofs(self, k: int) -> np.ndarray:
        """(ne, ndof) global dof indices.

        P1: vertex ids. P2: vertex ids followed by edge-midpoint dofs, whose
        global ids are offset by the vertex count (edge_ids are 0-based
        within their own namespace).
        """
        elems = self.elements
        if k == 1:
            return elems.copy()
        return np.concatenate(
            [elems, self.edge_ids + len(self.nodes)], axis=1)


def _build_edges(elements: np.ndarray):
    """Global edge registry.

    Local ordering convention: edge ``loc`` is opposite vertex ``loc``, i.e.
    (v1,v2), (v0,v2), (v0,v1). Returns per-element edge ids, the endpoint
    table, and the key -> id lookup.
    """
    ne = len(elements)
    pairs = [(1, 2), (0, 2), (0, 1)]
    lookup: dict[tuple[int, int], int] = {}
    edge_nodes_list: list[tuple[int, int]] = []
    edge_ids = np.empty((ne, 3), dtype=int)
    for e in range(ne):
        for loc, (i, j) in enumerate(pairs):
            key = (min(int(elements[e, i]), int(elements[e, j])),
                   max(int(elements[e, i]), int(elements[e, j])))
            eid = lookup.get(key)
            if eid is None:
                eid = len(edge_nodes_list)
                lookup[key] = eid
                edge_nodes_list.append(key)
            edge_ids[e, loc] = eid
    return (edge_ids, np.asarray(edge_nodes_list, dtype=int), lookup)


def unit_square_mesh(n: int) -> TriMesh:
    xs = np.linspace(0.0, 1.0, n + 1)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1)

    def nid(i: int, j: int) -> int:
        return i * (n + 1) + j

    elems: list[list[int]] = []
    cell_ids: list[int] = []
    for i in range(n):
        for j in range(n):
            v00, v10 = nid(i, j), nid(i + 1, j)
            v01, v11 = nid(i, j + 1), nid(i + 1, j + 1)
            elems.append([v00, v10, v11])
            cell_ids.append(i + j * n)
            elems.append([v00, v11, v01])
            cell_ids.append(i + j * n)
    elements = np.asarray(elems, dtype=int)

    edge_ids, edge_nodes, lookup = _build_edges(elements)

    edge_to_elements: dict[int, list[int]] = {}
    for e in range(len(elements)):
        for loc in range(3):
            edge_to_elements.setdefault(int(edge_ids[e, loc]), []).append(e)
    adjacency: list[list[int]] = []
    for e in range(len(elements)):
        nbrs: set[int] = set()
        for loc in range(3):
            for other in edge_to_elements[int(edge_ids[e, loc])]:
                if other != e:
                    nbrs.add(other)
        adjacency.append(sorted(nbrs))

    return TriMesh(nodes, elements, np.asarray(cell_ids, dtype=int),
                   n, edge_ids, edge_nodes, adjacency)
