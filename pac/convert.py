"""
Convert between molecular graphs and canonical PAC grids.

The reverse conversion is implemented by `pac.pac`.
"""

from collections.abc import Callable, Iterator
from copy import deepcopy
from functools import lru_cache
from itertools import chain
from typing import NamedTuple, cast

import networkx as nx
import numpy as np
import torch
from rdkit import Chem

import const
from pac.enums import (
    TRANSFORMATIONS,
    AtomType,
    EdgeDirection,
    Framework,
    FunctionalGroup,
    VertexDirection,
    vertex_direction_pair_to_edge_direction,
)
from pac.geometry import d6_orbit, reflected_channel_index, rotated_channel_index
from pac.matrix_manipulation import reflect, rotate

from .pac import PAC, PACMatrix


def mol_to_nx(mol: Chem.Mol) -> nx.Graph:
    """
    Convert an RDKit molecule into a labelled NetworkX graph.

    Hydrogens attached to heteroatoms are made explicit to preserve protonation and tautomeric state. Every atom node is
    annotated with atomic number, hybridisation, formal charge, aromaticity, and degree.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to convert.

    Returns
    -------
    networkx.Graph
        Molecular graph with atom and bond attributes.
    """
    atoms_to_add_hydrogens = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6
    ]
    if len(atoms_to_add_hydrogens) != 0:
        mol = Chem.AddHs(mol, onlyOnAtoms=atoms_to_add_hydrogens)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        atom: Chem.Atom
        G.add_node(
            atom.GetIdx(),
            atomic_number=atom.GetAtomicNum(),
            hybridization=atom.GetHybridization(),
            charge=atom.GetFormalCharge(),
            is_aromatic=atom.GetIsAromatic(),
            total_degree=atom.GetTotalDegree(),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType()
        )
    return G


EDGE_TO_C_VERTICES = {
    EdgeDirection.TOP_RIGHT: (0, 1),
    EdgeDirection.RIGHT: (1, 2),
    EdgeDirection.BOTTOM_RIGHT: (2, 3),
    EdgeDirection.BOTTOM_LEFT: (3, 4),
    EdgeDirection.LEFT: (4, 5),
    EdgeDirection.TOP_LEFT: (5, 0),
}


def _registered_tiles(pac: PAC, atom_index: int) -> list[tuple[int, int]]:
    """
    Return the tiles that contain a registered framework atom.

    Parameters
    ----------
    pac : PAC
        PAC grid under construction.
    atom_index : int
        Index of an atom of the molecular graph.

    Returns
    -------
    list of tuple of int
        Coordinates of each tile that contains the atom, in the order of registration.

    Raises
    ------
    ValueError
        Raised when the atom is not registered on a ring or linker tile.
    """
    tiles = pac.index_registry.locate_owner(atom_index)
    if tiles is None:
        raise ValueError(
            f"Atom {atom_index} is not registered on a ring or linker tile."
        )
    return tiles


def _registered_vertex_directions(
    pac: PAC, atom_index: int
) -> dict[tuple[int, int], VertexDirection]:
    """
    Return the vertex direction of a registered framework atom in each hexagon that contains the atom.

    Parameters
    ----------
    pac : PAC
        PAC grid under construction.
    atom_index : int
        Index of an atom of the molecular graph.

    Returns
    -------
    dict of tuple of int to VertexDirection
        Vertex direction of the atom in each hexagon, keyed by the position of the hexagon.

    Raises
    ------
    ValueError
        Raised when the atom is not registered on a ring or linker tile.
    """
    vertex_directions = pac.index_registry.locate_vertex_directions(atom_index)
    if vertex_directions is None:
        raise ValueError(
            f"Atom {atom_index} is not registered on a ring or linker tile."
        )
    return vertex_directions


def _nx_to_pac_choices(
    g: nx.Graph,
    grid_size: int | None = None,
    chooser: _Chooser | None = None,
    require_straight_five_membered_rings: bool = True,
) -> PAC:
    """
    Attempt an embedding of a molecular graph on the hexagonal lattice.

    Branch decisions are supplied by `chooser`, making each attempted embedding deterministic for a given decision
    vector.

    Parameters
    ----------
    g : networkx.Graph
        Labelled molecular graph to embed.
    grid_size : int or None, default None
        Grid side length. The project default is used when this is None.
    chooser : _Chooser or None, default None
        Source of branch decisions during embedding. The first available option is selected at every branch when this is
        None.
    require_straight_five_membered_rings : bool, default True
        Place every 5-membered ring that is fused to 2 rings, which are not fused to each other, in the straight
        orientation. The angular orientation of these rings is also accepted when this is False.

    Returns
    -------
    PAC
        Completed PAC grid for this decision vector.
    """
    # Perhaps only God can fully explain this function at this point
    if grid_size is None:
        grid_size = const.GRID_SIZE
    if chooser is None:
        chooser = _Chooser([])
    pac = PAC(grid_size=grid_size, init_pos=(grid_size // 2, grid_size // 2))

    all_rings_indices = [
        c
        for c in nx.algorithms.cycles.simple_cycles(g, length_bound=6)
        if len(c) in (5, 6)
    ]
    all_rings_indices.sort(key=len, reverse=True)

    if not all_rings_indices:
        return pac

    root_index = chooser.pick(len(all_rings_indices))
    if root_index:
        all_rings_indices.insert(0, all_rings_indices.pop(root_index))

    def straight_gap_bond(ring_indices):
        """
        Return the bond that crosses the empty vertex of a straight 5-membered ring.

        A 5-membered ring can be fused to 2 rings that are not fused to each other, as in a 6-5-6 system. The ring is
        straight when its 2 fusion edges are on opposite sides of its hexagon, which requires that the bond between the
        atoms of the 2 neighboring rings crosses the empty vertex. In the angular orientation, the 2 fusion edges are at
        120 degrees.

        Returns
        -------
        set or None
            The 2 atoms of the bond that must cross the empty vertex. The result is None when the ring does not have 2
            neighboring rings that are not fused to each other or when `require_straight_five_membered_rings` is False.
        """
        if not require_straight_five_membered_rings:
            return None
        nbrs = [
            r
            for r in all_rings_indices
            if set(r) != set(ring_indices) and len(set(r) & set(ring_indices)) >= 2
        ]
        if len(nbrs) != 2:
            return None
        R1, R2 = set(nbrs[0]), set(nbrs[1])
        if len(R1 & R2) >= 2:
            return None
        n = len(ring_indices)
        for i, a in enumerate(ring_indices):
            b = ring_indices[(i + 1) % n]
            if (a in R1 - R2 and b in R2 - R1) or (a in R2 - R1 and b in R1 - R2):
                return {a, b}
        return None

    def gap_flanks(cand):
        drop = cand.index(None)
        return {cand[(drop - 1) % 6], cand[(drop + 1) % 6]}

    # Mirrored frame, for distance ranking only, not adjacency

    def get_tile_center(tile_column_index, tile_row_index):
        return (
            np.sqrt(3) * (tile_column_index + tile_row_index / 2.0),
            1.5 * tile_row_index,
        )

    def get_vertex_coord(tile_column_index, tile_row_index, d):
        center_x, center_y = get_tile_center(tile_column_index, tile_row_index)
        if isinstance(d, VertexDirection):
            d = d.value
        offsets = {
            0: (0, -1),
            1: (np.sqrt(3) / 2, -0.5),
            2: (np.sqrt(3) / 2, 0.5),
            3: (0, 1),
            4: (-np.sqrt(3) / 2, 0.5),
            5: (-np.sqrt(3) / 2, -0.5),
        }
        return center_x + offsets[d][0], center_y + offsets[d][1]

    def get_cartesian(C):
        tile_position, d = C
        return get_vertex_coord(tile_position[0], tile_position[1], d)

    def get_coinciding_vertices(C):
        xy, d = C
        coinciding_vertices = [(xy, d)]
        PATHS = {
            VertexDirection.TOP: (
                (EdgeDirection.TOP_LEFT, VertexDirection.BOTTOM_RIGHT),
                (EdgeDirection.TOP_RIGHT, VertexDirection.BOTTOM_LEFT),
            ),
            VertexDirection.TOP_RIGHT: (
                (EdgeDirection.TOP_RIGHT, VertexDirection.BOTTOM),
                (EdgeDirection.RIGHT, VertexDirection.TOP_LEFT),
            ),
            VertexDirection.BOTTOM_RIGHT: (
                (EdgeDirection.RIGHT, VertexDirection.BOTTOM_LEFT),
                (EdgeDirection.BOTTOM_RIGHT, VertexDirection.TOP),
            ),
            VertexDirection.BOTTOM: (
                (EdgeDirection.BOTTOM_RIGHT, VertexDirection.TOP_LEFT),
                (EdgeDirection.BOTTOM_LEFT, VertexDirection.TOP_RIGHT),
            ),
            VertexDirection.BOTTOM_LEFT: (
                (EdgeDirection.BOTTOM_LEFT, VertexDirection.TOP),
                (EdgeDirection.LEFT, VertexDirection.BOTTOM_RIGHT),
            ),
            VertexDirection.TOP_LEFT: (
                (EdgeDirection.LEFT, VertexDirection.TOP_RIGHT),
                (EdgeDirection.TOP_LEFT, VertexDirection.BOTTOM),
            ),
        }
        for next_edge, next_vertex in PATHS[d]:
            nxt_xy = pac.shift(xy[0], xy[1], next_edge)
            coinciding_vertices.append((nxt_xy, next_vertex))
        return coinciding_vertices

    def get_atom_type_enum(node):
        atomic_number = node["atomic_number"]
        hybridization = node["hybridization"]
        is_aromatic = node.get("is_aromatic", False)
        total_degree = node.get("total_degree", 0)

        if atomic_number == 6:
            if is_aromatic:
                return AtomType.CARBON_AROMATIC
            elif hybridization == Chem.HybridizationType.SP2:
                return AtomType.CARBON_SP2
            else:
                return AtomType.CARBON_SP3
        elif atomic_number == 16:
            return AtomType.SULFUR_SP3
        elif atomic_number == 7:
            if hybridization == Chem.HybridizationType.SP2 or is_aromatic:
                if total_degree == 3:
                    return AtomType.NITROGEN_SP2_2PI
                else:
                    return AtomType.NITROGEN_SP2_1PI
            else:
                return AtomType.NITROGEN_SP3
        elif atomic_number == 8:
            return AtomType.OXYGEN_SP3
        raise ValueError(f"Unknown atom type: {atomic_number}")

    def get_extra_neighbor(column_index, row_index, d):
        if d == 0:
            return pac.shift(column_index, row_index, EdgeDirection.TOP_LEFT) + (
                VertexDirection.TOP_RIGHT,
            )
        if d == 1:
            return pac.shift(column_index, row_index, EdgeDirection.TOP_RIGHT) + (
                VertexDirection.BOTTOM_RIGHT,
            )
        if d == 2:
            return pac.shift(column_index, row_index, EdgeDirection.RIGHT) + (
                VertexDirection.BOTTOM,
            )
        if d == 3:
            return pac.shift(column_index, row_index, EdgeDirection.BOTTOM_RIGHT) + (
                VertexDirection.BOTTOM_LEFT,
            )
        if d == 4:
            return pac.shift(column_index, row_index, EdgeDirection.BOTTOM_LEFT) + (
                VertexDirection.TOP_LEFT,
            )
        if d == 5:
            return pac.shift(column_index, row_index, EdgeDirection.LEFT) + (
                VertexDirection.TOP,
            )

    def get_canonical_coord(
        column_index, row_index, d
    ) -> tuple[tuple[int, int], VertexDirection]:
        if isinstance(d, int):
            d = VertexDirection(d)
        coinciding_vertices = get_coinciding_vertices(((column_index, row_index), d))
        coinciding_vertices.sort(
            key=lambda item: (item[0][0], item[0][1], item[1].value)
        )
        return coinciding_vertices[0]

    def get_neighbors(cx, cy, cd):
        n1 = get_canonical_coord(cx, cy, VertexDirection((cd.value + 1) % 6))
        n2 = get_canonical_coord(cx, cy, VertexDirection((cd.value - 1) % 6))
        ex, ey, ed = get_extra_neighbor(cx, cy, cd.value)
        n3 = get_canonical_coord(ex, ey, ed)
        return [n1, n2, n3]

    def get_tiles_sharing_edge(C_a, C_b):
        tiles = []
        coinciding_vertices_a = get_coinciding_vertices(C_a)
        coinciding_vertices_b = get_coinciding_vertices(C_b)
        for xy_a, d_a in coinciding_vertices_a:
            for xy_b, d_b in coinciding_vertices_b:
                if xy_a == xy_b:
                    tiles.append((xy_a, d_a, d_b))
        return tiles

    def is_trapped(cand_indices, tile_xy, out_atoms, occ_map):
        for d, atom_idx in enumerate(cand_indices):
            if atom_idx in out_atoms:
                c_coord = get_canonical_coord(
                    tile_xy[0], tile_xy[1], VertexDirection(d)
                )
                nbrs = get_neighbors(*c_coord[0], c_coord[1])
                has_free = False
                for nbr in nbrs:
                    # Check whether neighbor belongs to candidate ring
                    nbr_in_cand = False
                    for i_c, a_c in enumerate(cand_indices):
                        if (
                            a_c is not None
                            and get_canonical_coord(
                                tile_xy[0], tile_xy[1], VertexDirection(i_c)
                            )
                            == nbr
                        ):
                            nbr_in_cand = True
                            break
                    # Reject an external neighbor that occupies an earlier placement
                    if not nbr_in_cand and nbr not in occ_map:
                        has_free = True
                        break
                if not has_free:
                    return True
        return False

    unplaced_rings = all_rings_indices.copy()
    placed_rings = []
    occupied_canonical_to_atom = {}

    root_ring_indices = unplaced_rings.pop(0)

    if len(root_ring_indices) == 6:
        canonical_root: list[int | None] = list(root_ring_indices)
    else:
        # Root ring is 5-membered because `all_rings_indices` contains only 5- and 6-membered rings
        apex_node = None
        for n in root_ring_indices:
            atom = g.nodes[n]
            if (
                atom.get("atomic_number") != 6
                or atom.get("hybridization") == Chem.HybridizationType.SP3
            ):
                apex_node = n
                break
        if apex_node is None:
            apex_node = root_ring_indices[2]

        _req = straight_gap_bond(root_ring_indices)
        if _req is not None:
            # Apex faces gap bond across ring
            for _cand_apex in root_ring_indices:
                _i = root_ring_indices.index(_cand_apex)
                _r = root_ring_indices[_i:] + root_ring_indices[:_i]
                _r = _r[-2:] + _r[:-2]
                if {_r[0], _r[4]} == _req:
                    apex_node = _cand_apex
                    break
            else:
                raise ValueError(
                    f"Could not place seed five-membered ring {root_ring_indices} "
                    "with opposite fusion edges."
                )

        idx = root_ring_indices.index(apex_node)
        r = root_ring_indices[idx:] + root_ring_indices[:idx]
        r = r[-2:] + r[:-2]
        canonical_root = [None, r[4], r[3], r[2], r[1], r[0]]

    pac.index_registry.add_ring(*pac.init_pos, canonical_root)
    missing_count = canonical_root.count(None)
    if missing_count == 0:
        pac[0, pac.init_pos[0], pac.init_pos[1]] = 1
    elif missing_count == 1:
        missing_vertex = VertexDirection(canonical_root.index(None))
        pac[0, pac.init_pos[0], pac.init_pos[1]] = (2, missing_vertex)

    placed_rings.append(root_ring_indices)
    for d, atom_idx in enumerate(canonical_root):
        if atom_idx is not None:
            occupied_canonical_to_atom[
                get_canonical_coord(
                    pac.init_pos[0], pac.init_pos[1], VertexDirection(d)
                )
            ] = atom_idx

    rings_to_build = []
    for r in unplaced_rings.copy():
        if len(set(r) & set(root_ring_indices)) >= 2:
            rings_to_build.append(r)
            unplaced_rings.remove(r)

    INVERSE_DIR = {
        EdgeDirection.TOP_RIGHT: EdgeDirection.BOTTOM_LEFT,
        EdgeDirection.RIGHT: EdgeDirection.LEFT,
        EdgeDirection.BOTTOM_RIGHT: EdgeDirection.TOP_LEFT,
        EdgeDirection.BOTTOM_LEFT: EdgeDirection.TOP_RIGHT,
        EdgeDirection.LEFT: EdgeDirection.RIGHT,
        EdgeDirection.TOP_LEFT: EdgeDirection.BOTTOM_RIGHT,
    }

    SHARED_EDGE_VERTICES_IN_B = {
        EdgeDirection.TOP_RIGHT: (VertexDirection.TOP, VertexDirection.TOP_RIGHT),
        EdgeDirection.RIGHT: (VertexDirection.TOP_RIGHT, VertexDirection.BOTTOM_RIGHT),
        EdgeDirection.BOTTOM_RIGHT: (
            VertexDirection.BOTTOM_RIGHT,
            VertexDirection.BOTTOM,
        ),
        EdgeDirection.BOTTOM_LEFT: (
            VertexDirection.BOTTOM,
            VertexDirection.BOTTOM_LEFT,
        ),
        EdgeDirection.LEFT: (VertexDirection.BOTTOM_LEFT, VertexDirection.TOP_LEFT),
        EdgeDirection.TOP_LEFT: (VertexDirection.TOP_LEFT, VertexDirection.TOP),
    }

    indices_all_ring_atoms = list(chain.from_iterable(all_rings_indices))
    linker_atom_to_coord: dict[int, tuple[tuple[int, int], VertexDirection]] = {}

    _ring_atom_set = set(indices_all_ring_atoms)
    _ring_graph = nx.Graph()
    for _i, _r1 in enumerate(all_rings_indices):
        _ring_graph.add_node(_i)
        for _j, _r2 in enumerate(all_rings_indices):
            if _i != _j and len(set(_r1) & set(_r2)) >= 2:
                _ring_graph.add_edge(_i, _j)
    _fused_systems = []
    for _comp in nx.connected_components(_ring_graph):
        _atoms = set()
        for _ri in _comp:
            _atoms.update(all_rings_indices[_ri])
        _fused_systems.append(_atoms)

    grid_occupying_atoms = set(_ring_atom_set)
    _non_ring = [n for n in g.nodes if n not in _ring_atom_set]
    for _comp in nx.connected_components(g.subgraph(_non_ring)):
        _touched = set()
        for _n in _comp:
            for _nbr in g.neighbors(_n):
                for _si, _sys in enumerate(_fused_systems):
                    if _nbr in _sys:
                        _touched.add(_si)
        if len(_touched) > 1:
            grid_occupying_atoms.update(_comp)

    cycles_without_progress = 0
    while rings_to_build or unplaced_rings:
        if rings_to_build:
            ring_indices = rings_to_build.pop(0)

            shared_ring_indices = []
            for ring_index in ring_indices:
                if ring_index in pac.index_registry:
                    shared_ring_indices.append(ring_index)

            for i in range(len(ring_indices)):
                c1 = ring_indices[i]
                c2 = ring_indices[(i + 1) % len(ring_indices)]

                if c1 in shared_ring_indices and c2 in shared_ring_indices:
                    fused_rings_candidate_1 = pac.index_registry.index_to_ring[c1]
                    fused_rings_candidate_2 = pac.index_registry.index_to_ring[c2]
                    fused_rings_candidates = set(fused_rings_candidate_1) & set(
                        fused_rings_candidate_2
                    )

                    if len(fused_rings_candidates) == 0:
                        continue

                    fused_ring = next(iter(fused_rings_candidates))
                    fused_vertex_1 = _registered_vertex_directions(pac, c1)[fused_ring]
                    fused_vertex_2 = _registered_vertex_directions(pac, c2)[fused_ring]

                    if (fused_vertex_2.value - fused_vertex_1.value) % 6 not in (1, 5):
                        continue

                    consecutive_shared_index_1 = c1
                    consecutive_shared_index_2 = c2
                    break
            else:
                # No pair of consecutive ring atoms is shared with a placed ring
                rings_to_build.append(ring_indices)
                cycles_without_progress += 1
                if cycles_without_progress > len(rings_to_build):
                    raise ValueError(
                        f"Grid geometry conflict: cannot anchor ring {ring_indices}."
                    )
                continue
            cycles_without_progress = 0

            is_clockwise = (fused_vertex_2.value - fused_vertex_1.value) % 6 == 1
            direction = vertex_direction_pair_to_edge_direction[
                (fused_vertex_1, fused_vertex_2)
                if is_clockwise
                else (fused_vertex_2, fused_vertex_1)
            ]

            column_step, row_step = TRANSFORMATIONS[direction]
            xy_position_this_ring = (
                (fused_ring[0] + column_step) % pac.grid_size,
                (fused_ring[1] + row_step) % pac.grid_size,
            )

            inv_dir = INVERSE_DIR[direction]
            b_vertex1, b_vertex2 = SHARED_EDGE_VERTICES_IN_B[inv_dir]

            if is_clockwise:
                target_vertex_1, target_vertex_2 = b_vertex2.value, b_vertex1.value
            else:
                target_vertex_1, target_vertex_2 = b_vertex1.value, b_vertex2.value

            canonical_ring_indices: list[int | None] = [None] * 6

            if len(ring_indices) == 6:
                canonical_ring_indices[target_vertex_1] = consecutive_shared_index_1
                canonical_ring_indices[target_vertex_2] = consecutive_shared_index_2

                idx1_in_ring = ring_indices.index(consecutive_shared_index_1)
                idx2_in_ring = ring_indices.index(consecutive_shared_index_2)
                ring_is_forward = (idx2_in_ring - idx1_in_ring) % 6 == 1

                grid_step = 1 if (target_vertex_2 - target_vertex_1) % 6 == 1 else -1

                current_vertex = target_vertex_2
                current_ring_idx = idx2_in_ring

                for _ in range(4):
                    current_vertex = (current_vertex + grid_step) % 6
                    current_ring_idx = (
                        current_ring_idx + (1 if ring_is_forward else -1)
                    ) % 6
                    canonical_ring_indices[current_vertex] = ring_indices[
                        current_ring_idx
                    ]

            elif len(ring_indices) == 5:
                idx1 = ring_indices.index(consecutive_shared_index_1)
                idx2 = ring_indices.index(consecutive_shared_index_2)
                ring_is_forward = (idx2 - idx1) % len(ring_indices) == 1

                seq = []
                curr_idx = idx1
                for _ in range(5):
                    seq.append(ring_indices[curr_idx])
                    curr_idx = (curr_idx + (1 if ring_is_forward else -1)) % 5

                nodes_to_place = seq[2:]

                step = 1 if (target_vertex_2 - target_vertex_1) % 6 == 1 else -1

                vertices_to_fill = [
                    (target_vertex_2 + i * step) % 6 for i in range(1, 5)
                ]
                valid_candidates = []

                for drop_idx in range(4):
                    candidate: list[int | None] = [None] * 6
                    candidate[target_vertex_1] = seq[0]
                    candidate[target_vertex_2] = seq[1]

                    node_idx = 0
                    for i, vertex in enumerate(vertices_to_fill):
                        if i == drop_idx:
                            continue
                        candidate[vertex] = nodes_to_place[node_idx]
                        node_idx += 1

                    is_valid = True
                    column_index, row_index = xy_position_this_ring
                    for edge_dir in EdgeDirection:
                        dx_t, dy_t = TRANSFORMATIONS[edge_dir]
                        nx_val, ny_val = (
                            (column_index + dx_t) % pac.grid_size,
                            (row_index + dy_t) % pac.grid_size,
                        )

                        if (nx_val, ny_val) in pac.index_registry.indexes:
                            neighbor_ring = pac.index_registry.indexes[(nx_val, ny_val)]
                            n_vertex_a, n_vertex_b = EDGE_TO_C_VERTICES[
                                INVERSE_DIR[edge_dir]
                            ]
                            c_vertex_a, c_vertex_b = EDGE_TO_C_VERTICES[edge_dir]

                            exp_node_0 = neighbor_ring.indexes[n_vertex_a]
                            exp_node_1 = neighbor_ring.indexes[n_vertex_b]

                            if (
                                exp_node_0 is not None
                                and exp_node_0 != candidate[c_vertex_b]
                            ):
                                is_valid = False
                                break
                            if (
                                exp_node_1 is not None
                                and exp_node_1 != candidate[c_vertex_a]
                            ):
                                is_valid = False
                                break
                            if exp_node_0 is None and candidate[c_vertex_b] is not None:
                                is_valid = False
                                break
                            if exp_node_1 is None and candidate[c_vertex_a] is not None:
                                is_valid = False
                                break

                    for d_vertex, atom_idx in enumerate(candidate):
                        if atom_idx is not None:
                            c_coord = get_canonical_coord(
                                column_index, row_index, VertexDirection(d_vertex)
                            )
                            if (
                                c_coord in occupied_canonical_to_atom
                                and occupied_canonical_to_atom[c_coord] != atom_idx
                            ):
                                is_valid = False
                                break

                    if is_valid:
                        valid_candidates.append((candidate, vertices_to_fill[drop_idx]))

                if not valid_candidates:
                    raise ValueError(
                        "No valid configuration found for 5-membered ring fusion."
                    )

                required_gap = straight_gap_bond(ring_indices)
                if required_gap is not None:
                    straight_only = [
                        (c, d)
                        for c, d in valid_candidates
                        if gap_flanks(c) == required_gap
                    ]
                    if not straight_only:
                        raise ValueError(
                            f"No straight placement for fused 5-membered ring {ring_indices}."
                        )
                    valid_candidates = straight_only

                future_candidates = []
                for cand, drop_vertex in valid_candidates:
                    node_a = cand[(drop_vertex - 1) % 6]
                    node_b = cand[(drop_vertex + 1) % 6]

                    is_shared_with_future = False
                    for other_ring in all_rings_indices:
                        if set(other_ring) == set(ring_indices):
                            continue
                        if node_a in other_ring and node_b in other_ring:
                            is_shared_with_future = True
                            break

                    if not is_shared_with_future:
                        future_candidates.append((cand, drop_vertex))

                if future_candidates:
                    valid_candidates = future_candidates

                parent_ring_atoms = pac.index_registry.indexes[fused_ring].indexes

                outward_atoms_5 = set()
                for other_ring in all_rings_indices:
                    if set(other_ring) == set(ring_indices):
                        continue
                    if set(other_ring) == set(parent_ring_atoms):
                        continue
                    outward_atoms_5.update(set(ring_indices) & set(other_ring))

                for node in ring_indices:
                    for nbr in g.neighbors(node):
                        if nbr not in ring_indices and nbr not in parent_ring_atoms:
                            outward_atoms_5.add(node)

                untrapped_candidates = []
                for cand, drop_vertex in valid_candidates:
                    if not is_trapped(
                        cand,
                        xy_position_this_ring,
                        outward_atoms_5,
                        occupied_canonical_to_atom,
                    ):
                        untrapped_candidates.append((cand, drop_vertex))

                if untrapped_candidates:
                    valid_candidates = untrapped_candidates

                # Rank fusion sites by ring atoms, excluding (???)
                ring_fusion_atoms = set()
                for other_ring in all_rings_indices:
                    if set(other_ring) == set(ring_indices):
                        continue
                    if set(other_ring) == set(parent_ring_atoms):
                        continue
                    ring_fusion_atoms.update(set(ring_indices) & set(other_ring))

                px, py = get_tile_center(fused_ring[0], fused_ring[1])

                def rank_candidate(item):
                    cand, drop_vertex = item
                    hx, hy = xy_position_this_ring

                    sum_x, sum_y = 0, 0
                    for i in range(6):
                        if i == drop_vertex:
                            continue
                        nx_c, ny_c = get_vertex_coord(hx, hy, i)
                        sum_x += nx_c
                        sum_y += ny_c
                    centroid_x, centroid_y = sum_x / 5.0, sum_y / 5.0

                    dist_to_parent = (centroid_x - px) ** 2 + (centroid_y - py) ** 2

                    opposite_1 = (target_vertex_1 + 3) % 6
                    opposite_2 = (target_vertex_2 + 3) % 6

                    future_shared = sum(
                        1
                        for vertex in (opposite_1, opposite_2)
                        if cand[vertex] in ring_fusion_atoms
                    )

                    score = (-future_shared, -dist_to_parent)
                    return score

                valid_candidates.sort(key=rank_candidate)

                canonical_ring_indices = valid_candidates[
                    chooser.pick(len(valid_candidates))
                ][0]

            # Reject placements on occupied vertices
            for d_vertex, atom_idx in enumerate(canonical_ring_indices):
                if atom_idx is None:
                    continue
                c_coord = get_canonical_coord(
                    xy_position_this_ring[0],
                    xy_position_this_ring[1],
                    VertexDirection(d_vertex),
                )
                if (
                    c_coord in occupied_canonical_to_atom
                    and occupied_canonical_to_atom[c_coord] != atom_idx
                ):
                    raise ValueError(
                        f"Grid collision placing fused ring {ring_indices}: vertex {c_coord} already contains atom "
                        f"{occupied_canonical_to_atom[c_coord]}, wanted {atom_idx}."
                    )

            pac.index_registry.add_ring(
                xy_position_this_ring[0],
                xy_position_this_ring[1],
                canonical_ring_indices,
            )
            missing_count = canonical_ring_indices.count(None)
            if missing_count == 0:
                pac[0, xy_position_this_ring[0], xy_position_this_ring[1]] = 1
            elif missing_count == 1:
                missing_vertex = VertexDirection(canonical_ring_indices.index(None))
                pac[0, xy_position_this_ring[0], xy_position_this_ring[1]] = (
                    2,
                    missing_vertex,
                )

            placed_rings.append(ring_indices)
            for d, atom_idx in enumerate(canonical_ring_indices):
                if atom_idx is not None:
                    occupied_canonical_to_atom[
                        get_canonical_coord(
                            xy_position_this_ring[0],
                            xy_position_this_ring[1],
                            VertexDirection(d),
                        )
                    ] = atom_idx
            for r in unplaced_rings.copy():
                if len(set(r) & set(ring_indices)) >= 2:
                    rings_to_build.append(r)
                    unplaced_rings.remove(r)
        else:
            placed_atoms = set(chain.from_iterable(placed_rings))
            unplaced_atoms = set(chain.from_iterable(unplaced_rings))

            queue = []
            for p in placed_atoms:
                queue.append((p, [p]))
            visited = set(placed_atoms)

            linker_path = None
            while queue:
                curr, path = queue.pop(0)
                if curr in unplaced_atoms and curr not in placed_atoms:
                    linker_path = path
                    break

                for neighbor in g.neighbors(curr):
                    if neighbor not in visited and (
                        neighbor not in indices_all_ring_atoms
                        or neighbor in unplaced_atoms
                    ):
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

            if not linker_path:
                raise ValueError("Disconnected molecular graph or invalid linker!")

            target_ring = next(r for r in unplaced_rings if linker_path[-1] in r)

            u_0 = linker_path[0]
            if u_0 in linker_atom_to_coord:
                start_C = linker_atom_to_coord[u_0]
            else:
                ring_xy = _registered_tiles(pac, u_0)[0]
                vertex_direction = _registered_vertex_directions(pac, u_0)[ring_xy]
                start_C = get_canonical_coord(ring_xy[0], ring_xy[1], vertex_direction)

            grid_queue = [(start_C, [start_C])]
            valid_paths = []

            while grid_queue:
                curr_C, grid_path = grid_queue.pop(0)
                curr_atom_idx = len(grid_path) - 1

                if len(grid_path) == len(linker_path):
                    v_0_C = grid_path[-1]
                    prev_C = grid_path[-2]

                    coinciding_vertices = get_coinciding_vertices(v_0_C)
                    valid_ring_tiles = []
                    for coinciding_xy, coinciding_direction in coinciding_vertices:
                        if pac.framework[coinciding_xy] == 0:
                            c_plus = get_canonical_coord(
                                coinciding_xy[0],
                                coinciding_xy[1],
                                VertexDirection((coinciding_direction.value + 1) % 6),
                            )
                            c_minus = get_canonical_coord(
                                coinciding_xy[0],
                                coinciding_xy[1],
                                VertexDirection((coinciding_direction.value - 1) % 6),
                            )
                            if c_plus != prev_C and c_minus != prev_C:
                                collision = False
                                for i in range(6):
                                    cand_c = get_canonical_coord(
                                        coinciding_xy[0],
                                        coinciding_xy[1],
                                        VertexDirection(i),
                                    )
                                    if (
                                        cand_c in occupied_canonical_to_atom
                                        and occupied_canonical_to_atom[cand_c]
                                        != linker_path[-1]
                                    ):
                                        collision = True
                                        break
                                if not collision:
                                    valid_ring_tiles.append(
                                        (coinciding_xy, coinciding_direction)
                                    )

                    for rt in valid_ring_tiles:
                        valid_paths.append((grid_path, rt))
                    continue

                nxt_atom = linker_path[curr_atom_idx + 1]
                required_C = linker_atom_to_coord.get(nxt_atom)

                neighbors_found = get_neighbors(*curr_C[0], curr_C[1])

                for nxt_C in neighbors_found:
                    if nxt_C in grid_path:
                        continue

                    if required_C is not None:
                        if nxt_C != required_C:
                            continue
                    else:
                        if (
                            nxt_C in occupied_canonical_to_atom
                            and occupied_canonical_to_atom[nxt_C] != nxt_atom
                        ):
                            continue

                        # Consecutive occupied vertices of a hexagon become bonded during decoding. Reject placements
                        # adjacent to unbonded atoms.
                        fragmentation = False
                        for tile_xy, d_prev, d_curr in get_tiles_sharing_edge(
                            curr_C, nxt_C
                        ):
                            for delta in (-1, 1):
                                adj_vertex = (d_curr.value + delta) % 6
                                if adj_vertex == d_prev.value:
                                    continue
                                if pac.atom_types[adj_vertex][tile_xy] == 0:
                                    continue
                                adj_C = get_canonical_coord(
                                    tile_xy[0], tile_xy[1], VertexDirection(adj_vertex)
                                )
                                other_atom = occupied_canonical_to_atom.get(adj_C)
                                if other_atom is None or other_atom == nxt_atom:
                                    continue
                                if not g.has_edge(other_atom, nxt_atom):
                                    fragmentation = True
                                    break
                            if fragmentation:
                                break
                        if fragmentation:
                            continue

                    tiles = get_tiles_sharing_edge(curr_C, nxt_C)
                    if any(pac.framework[t[0]] in (0, 3) for t in tiles):
                        grid_queue.append((nxt_C, grid_path + [nxt_C]))

            if not valid_paths:
                raise ValueError("Could not route linker on the hexagonal grid.")

            core_x, core_y = 0, 0
            for rx, ry in pac.index_registry.indexes.keys():
                cx, cy = get_tile_center(rx, ry)
                core_x += cx
                core_y += cy
            if len(pac.index_registry.indexes) > 0:
                core_x /= len(pac.index_registry.indexes)
                core_y /= len(pac.index_registry.indexes)

            outward_atoms = set()
            for r in unplaced_rings:
                if r == target_ring:
                    continue
                outward_atoms.update(set(target_ring) & set(r))

            for node in target_ring:
                for nbr in g.neighbors(node):
                    if (
                        nbr not in target_ring
                        and nbr not in placed_atoms
                        and nbr not in linker_path
                    ) and nbr in grid_occupying_atoms:
                        outward_atoms.add(node)

            # Try heteroatom opposite empty vertex first. Try other vertices if this blocks a required fusion
            # edge.
            if len(target_ring) == 5:
                heuristic_apex = None
                for n in target_ring:
                    atom = g.nodes[n]
                    if (
                        atom.get("atomic_number") != 6
                        or atom.get("hybridization") == Chem.HybridizationType.SP3
                    ):
                        heuristic_apex = n
                        break
                if heuristic_apex is None:
                    heuristic_apex = target_ring[2]
                apex_options = [heuristic_apex] + [
                    n for n in target_ring if n != heuristic_apex
                ]
            else:
                apex_options = [None]

            ranked = []
            order = 0
            for apex_rank, apex_node in enumerate(apex_options):
                for gp, rt in valid_paths:
                    tile_xy, d_v0 = rt
                    idx_v0 = target_ring.index(linker_path[-1])

                    for step in [1, -1]:
                        order += 1
                        cand_indices: list[int | None] = [None] * 6
                        if len(target_ring) == 6:
                            for i in range(6):
                                cand_indices[(d_v0.value + i * step) % 6] = target_ring[
                                    (idx_v0 + i) % 6
                                ]
                        elif len(target_ring) == 5:
                            if apex_node is None:
                                raise ValueError(
                                    f"The 5-membered ring {target_ring} does not have an apex atom."
                                )
                            idx = target_ring.index(apex_node)
                            r = target_ring[idx:] + target_ring[:idx]
                            r = r[-2:] + r[:-2]
                            canonical_root_5 = [None, r[4], r[3], r[2], r[1], r[0]]
                            idx_in_canonical = canonical_root_5.index(linker_path[-1])

                            for i in range(6):
                                if canonical_root_5[i] is not None:
                                    dist = i - idx_in_canonical
                                    vertex = (d_v0.value + dist * step) % 6
                                    cand_indices[vertex] = canonical_root_5[i]

                        # Check both orientations for collisions
                        is_valid = True
                        for d, atom_idx in enumerate(cand_indices):
                            if atom_idx is not None:
                                c_coord = get_canonical_coord(
                                    tile_xy[0], tile_xy[1], VertexDirection(d)
                                )
                                if (
                                    c_coord in occupied_canonical_to_atom
                                    and occupied_canonical_to_atom[c_coord] != atom_idx
                                ):
                                    is_valid = False
                                    break
                        if not is_valid:
                            continue

                        if len(target_ring) == 5:
                            req = straight_gap_bond(target_ring)
                            if req is not None and gap_flanks(cand_indices) != req:
                                continue

                        if is_trapped(
                            cand_indices,
                            tile_xy,
                            outward_atoms,
                            occupied_canonical_to_atom,
                        ):
                            continue

                        c1 = get_cartesian(gp[0])
                        c2 = get_cartesian(gp[-1])
                        dist_linker = (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2

                        rt_c = get_tile_center(tile_xy[0], tile_xy[1])
                        dist_to_core = (rt_c[0] - core_x) ** 2 + (rt_c[1] - core_y) ** 2

                        future_dist = 0
                        for d, atom_idx in enumerate(cand_indices):
                            if atom_idx in outward_atoms:
                                nx_c, ny_c = get_vertex_coord(tile_xy[0], tile_xy[1], d)
                                future_dist += (nx_c - core_x) ** 2 + (
                                    ny_c - core_y
                                ) ** 2

                        score = dist_linker + dist_to_core * 10 + future_dist * 100
                        ranked.append((apex_rank, -score, order, gp, rt, cand_indices))

            if not ranked:
                raise ValueError(
                    "No collision-free orientation for target ring at the end of linker."
                )

            ranked.sort(key=lambda item: item[:3])
            _, _, _, grid_path, ring_tile, canonical_ring_indices = ranked[
                chooser.pick(len(ranked))
            ]

            for i in range(1, len(grid_path)):
                C_prev = grid_path[i - 1]
                C_curr = grid_path[i]
                atom_prev = linker_path[i - 1]
                atom_curr = linker_path[i]

                if atom_curr not in linker_atom_to_coord:
                    tiles = get_tiles_sharing_edge(C_prev, C_curr)
                    for tile_xy, d_prev, d_curr in tiles:
                        if pac.framework[tile_xy] == 0:
                            pac[0, tile_xy[0], tile_xy[1]] = 3

                        if pac.framework[tile_xy] in (1, 2, 3):
                            pac[d_prev.value + 7, tile_xy[0], tile_xy[1]] = (
                                get_atom_type_enum(g.nodes[atom_prev]).value
                            )
                            pac[d_curr.value + 7, tile_xy[0], tile_xy[1]] = (
                                get_atom_type_enum(g.nodes[atom_curr]).value
                            )

                    if atom_curr not in indices_all_ring_atoms:
                        occupied_canonical_to_atom[C_curr] = atom_curr
                        linker_atom_to_coord[atom_curr] = C_curr

            tile_xy, d_v0 = ring_tile

            pac.index_registry.add_ring(tile_xy[0], tile_xy[1], canonical_ring_indices)
            missing_count = canonical_ring_indices.count(None)
            if missing_count == 0:
                pac[0, tile_xy[0], tile_xy[1]] = 1
            elif missing_count == 1:
                missing_vertex = VertexDirection(canonical_ring_indices.index(None))
                pac[0, tile_xy[0], tile_xy[1]] = (2, missing_vertex)

            placed_rings.append(target_ring)
            unplaced_rings.remove(target_ring)
            for d, atom_idx in enumerate(canonical_ring_indices):
                if atom_idx is not None:
                    occupied_canonical_to_atom[
                        get_canonical_coord(tile_xy[0], tile_xy[1], VertexDirection(d))
                    ] = atom_idx

            for r in unplaced_rings.copy():
                if len(set(r) & set(target_ring)) >= 2:
                    rings_to_build.append(r)
                    unplaced_rings.remove(r)

    for node in g.nodes:
        pac.index_registry.node_attributes[node] = {
            "atomic_number": g.nodes[node].get("atomic_number", 6),
            "hybridization": g.nodes[node].get(
                "hybridization", Chem.HybridizationType.SP2
            ),
            "charge": g.nodes[node].get("charge", 0),
        }

    for ring_xy, ring_reg in pac.index_registry.indexes.items():
        missing_count = ring_reg.indexes.count(None)
        if missing_count == 0:
            pac[0, ring_xy[0], ring_xy[1]] = Framework.SIX_MEMBERED.value
        elif missing_count == 1:
            missing_vertex = VertexDirection(ring_reg.indexes.index(None))
            pac[0, ring_xy[0], ring_xy[1]] = (2, missing_vertex)

    indices_all_ring_atoms = list(chain.from_iterable(all_rings_indices))
    for ring_atom_index in indices_all_ring_atoms:
        node = g.nodes[ring_atom_index]
        new_type = get_atom_type_enum(node)

        ring_xy = _registered_tiles(pac, ring_atom_index)[0]
        vertex_direction = _registered_vertex_directions(pac, ring_atom_index)[ring_xy]
        vertex_direction = vertex_direction.value
        pac.index_registry.update_atom_type(*ring_xy, vertex_direction, new_type)
        pac[vertex_direction + 7, ring_xy[0], ring_xy[1]] = new_type.value

    ring_graph = nx.Graph()
    for i, ring1 in enumerate(all_rings_indices):
        ring_graph.add_node(i)
        for j, ring2 in enumerate(all_rings_indices):
            if i != j and len(set(ring1) & set(ring2)) >= 2:
                ring_graph.add_edge(i, j)

    fused_systems_indices = list(nx.connected_components(ring_graph))
    fused_systems = []
    for comp in fused_systems_indices:
        atoms = set()
        for ring_idx in comp:
            atoms.update(all_rings_indices[ring_idx])
        fused_systems.append(atoms)

    indices_all_nodes = set(g.nodes.keys())
    indices_all_ring_nodes = set(indices_all_ring_atoms)
    indices_all_non_ring_nodes = list(indices_all_nodes - indices_all_ring_nodes)
    non_ring_components = []

    while len(indices_all_non_ring_nodes) > 0:
        node_or_nodes: int | list[int] = indices_all_non_ring_nodes.pop()
        if isinstance(node_or_nodes, int):
            nodes = [node_or_nodes]
        else:
            nodes = node_or_nodes

        neighbors = set(g.neighbors(nodes[-1]))
        possibly_has_more_neighbors = False
        for neighbor in neighbors:
            if neighbor in indices_all_ring_nodes:
                continue
            if neighbor in indices_all_non_ring_nodes:
                indices_all_non_ring_nodes.remove(neighbor)
                indices_all_non_ring_nodes.append(nodes + [neighbor])
                possibly_has_more_neighbors = True
        if not possibly_has_more_neighbors:
            non_ring_components.append(nodes)

    for i in range(len(non_ring_components)):
        for j in range(i + 1, len(non_ring_components)):
            if set(non_ring_components[i]) & set(non_ring_components[j]):
                non_ring_components[i] = list(
                    set(non_ring_components[i]) | set(non_ring_components[j])
                )
                non_ring_components[j] = []

    non_ring_components = [idx for idx in non_ring_components if len(idx) > 0]

    linker_groups_indices = []
    functional_groups_indices = []

    for component in non_ring_components:
        connected_systems = set()
        for node in component:
            for neighbor in g.neighbors(node):
                for sys_idx, system_atoms in enumerate(fused_systems):
                    if neighbor in system_atoms:
                        connected_systems.add(sys_idx)

        if len(connected_systems) > 1:
            linker_groups_indices.append(component)
        else:
            functional_groups_indices.append(component)

    zero_atom_linkers = []
    for u, v in g.edges():
        u_sys = next((i for i, sys in enumerate(fused_systems) if u in sys), -1)
        v_sys = next((i for i, sys in enumerate(fused_systems) if v in sys), -1)
        if u_sys != -1 and v_sys != -1 and u_sys != v_sys:
            zero_atom_linkers.append((u, v))

    for functional_group_indices in functional_groups_indices:
        if len(functional_group_indices) == 1:
            if g.nodes[functional_group_indices[0]]["atomic_number"] == 6:
                fn_group = FunctionalGroup.METHYL
            elif g.nodes[functional_group_indices[0]]["atomic_number"] == 8:
                fn_group = FunctionalGroup.CARBONYL
            elif g.nodes[functional_group_indices[0]]["atomic_number"] == 9:
                fn_group = FunctionalGroup.FLUORIDE
            elif g.nodes[functional_group_indices[0]]["atomic_number"] == 17:
                fn_group = FunctionalGroup.CHLORIDE
            elif g.nodes[functional_group_indices[0]]["atomic_number"] == 1:
                fn_group = FunctionalGroup.HYDROGEN
            else:
                raise ValueError(
                    f"Unknown functional group with index {functional_group_indices[0]}."
                )
        elif len(functional_group_indices) == 2:
            atomic_numbers = [
                g.nodes[index]["atomic_number"] for index in functional_group_indices
            ]
            if atomic_numbers.count(6) == 2:
                fn_group = FunctionalGroup.ETHYL
            elif atomic_numbers.count(6) == 1 and atomic_numbers.count(7) == 1:
                fn_group = FunctionalGroup.CYANIDE
            elif atomic_numbers.count(6) == 1 and atomic_numbers.count(8) == 1:
                fn_group = FunctionalGroup.METHOXY
            else:
                raise ValueError(
                    f"Unknown functional group with indices {functional_group_indices}."
                )
        elif len(functional_group_indices) == 3:
            atomic_numbers = [
                g.nodes[index]["atomic_number"] for index in functional_group_indices
            ]
            if atomic_numbers.count(6) == 2 and atomic_numbers.count(7) == 1:
                fn_group = FunctionalGroup.DIMETHYLAMINO
            elif atomic_numbers.count(6) == 2 and atomic_numbers.count(8) == 1:
                fn_group = FunctionalGroup.ETHOXY
            elif atomic_numbers.count(7) == 1 and atomic_numbers.count(8) == 2:
                fn_group = FunctionalGroup.NITRO
            else:
                raise ValueError(
                    f"Unknown functional group with indices {functional_group_indices}."
                )
        else:
            raise ValueError(
                f"Unknown functional group with indices {functional_group_indices}."
            )

        ring_xy = None
        attached_index = None
        for index in functional_group_indices:
            neighbors = set(g.neighbors(index))
            for neighbor in neighbors:
                found = pac.index_registry.index_to_ring.get(neighbor, None)
                if found is not None:
                    assert len(found) == 1, (
                        "Functional group should be attached to only 1 ring."
                    )
                    ring_xy = found[0]
                    attached_index = neighbor
                    break

        if ring_xy is None or attached_index is None:
            raise ValueError(
                f"Functional group with indices {functional_group_indices} is not attached to a ring."
            )
        vertex_direction = _registered_vertex_directions(pac, attached_index)[ring_xy]
        vertex_direction = vertex_direction.value
        pac.index_registry.add_functional_group(
            ring_xy[0], ring_xy[1], vertex_direction, fn_group
        )
        pac.functional_groups[vertex_direction][ring_xy] = fn_group.value

    return _canonical_orientation(pac)


def _occupied_positions(
    tile_indices: np.ndarray, is_occupied: np.ndarray
) -> set[tuple[int, int]]:
    """
    Return the positions of the occupied tiles in a D6 image of a PAC grid.

    Parameters
    ----------
    tile_indices : numpy.ndarray of shape (grid_size, grid_size)
        Flat index ``row * grid_size + column`` of the input tile at each position of the image.
    is_occupied : numpy.ndarray of shape (grid_size * grid_size,)
        Occupancy of each input tile, in flat index order.

    Returns
    -------
    set of tuple of int
        ``(column_index, row_index)`` positions of the occupied tiles in the image.
    """
    row_indices, column_indices = np.nonzero(is_occupied[tile_indices])
    return set(zip(column_indices.tolist(), row_indices.tolist()))


def _d6_tile_indices(pac: PAC) -> list[np.ndarray]:
    """
    Calculate which input tile is at each position of the 12 D6 images of a PAC grid.

    An array of flat tile indices is transformed with `rotate`, `reflect`, and the translation of `PAC.realign` in the
    order of `d6_orbit`. The 13 channels of the PAC grid are not copied.

    Parameters
    ----------
    pac : PAC
        PAC grid to transform.

    Returns
    -------
    list of numpy.ndarray of shape (grid_size, grid_size)
        For each image, in the order of `d6_orbit`, the flat index ``row * grid_size + column`` of the input tile at
        each position.
    """
    grid_size = pac.grid_size
    is_occupied = pac.framework.toarray().reshape(-1) != 0

    def realign_tile_indices(tile_indices: np.ndarray) -> None:
        row_shift, column_shift = pac.calculate_realignment_shift(
            positions=_occupied_positions(tile_indices, is_occupied)
        )
        tile_indices[:] = np.roll(tile_indices, (row_shift, column_shift), axis=(0, 1))

    identity = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    return d6_orbit(
        identity, rotate=rotate, reflect=reflect, realign=realign_tile_indices
    )


def _d6_image(pac: PAC, tile_indices: np.ndarray, index: int) -> PAC:
    """
    Return a single D6 image of a PAC grid.

    The result is equal to entry `index` of `d6_orbit` for `PAC.rotated`, `PAC.reflected`, and `PAC.realign`.

    Parameters
    ----------
    pac : PAC
        PAC grid to transform.
    tile_indices : numpy.ndarray of shape (grid_size, grid_size)
        Flat index of the input tile at each position of the image, as returned by `_d6_tile_indices`.
    index : int
        Index of the image in the order of `d6_orbit`.

    Returns
    -------
    PAC
        The input PAC grid when `index` is 0, and a transformed copy otherwise.
    """
    if index == 0:
        return pac
    grid_size = pac.grid_size
    num_rotations = index % 6

    # Permute vertex channels in same order as `PAC.reflected` and `PAC.rotated`
    source_vertices = list(range(6))
    if index >= 6:
        source_vertices = [
            source_vertices[reflected_channel_index(vertex)] for vertex in range(6)
        ]
    for _ in range(num_rotations):
        source_vertices = [
            source_vertices[rotated_channel_index(vertex)] for vertex in range(6)
        ]

    pac_image = deepcopy(pac)
    pac_image.framework = PACMatrix(pac.framework.toarray().reshape(-1)[tile_indices])
    pac_image.functional_groups = [
        PACMatrix(pac.functional_groups[vertex].toarray().reshape(-1)[tile_indices])
        for vertex in source_vertices
    ]
    pac_image.atom_types = [
        PACMatrix(pac.atom_types[vertex].toarray().reshape(-1)[tile_indices])
        for vertex in source_vertices
    ]

    # `PAC.rotated` and `PAC.reflected` transform `init_pos` with grid, but `PAC.realign` won't
    unaligned_tile_indices = np.arange(grid_size * grid_size).reshape(
        grid_size, grid_size
    )
    if index >= 6:
        unaligned_tile_indices = reflect(unaligned_tile_indices)
    for _ in range(num_rotations):
        unaligned_tile_indices = rotate(unaligned_tile_indices)
    init_column, init_row = pac.init_pos
    new_init_pos = np.argwhere(
        unaligned_tile_indices == init_row * grid_size + init_column
    )
    pac_image.init_pos = (new_init_pos[0][1], new_init_pos[0][0])
    return pac_image


def _canonical_orientation(pac: PAC) -> PAC:
    """
    Return the realigned D6 image of a PAC grid with the smallest, most centred bounding box.

    The images are compared by 4 values in this order: the row span, the summed distance of the 2 row limits from the
    grid centre, the column span, and the summed distance of the 2 column limits from the grid centre. When 2 or more
    images are equal in all 4 values, the first of these images in the order of `d6_orbit` is returned. The values of
    the first image, which is the input PAC grid, are calculated before realignment, and the values of the other 11
    images after realignment. When the first image is selected, the input PAC grid is realigned in place with
    `PAC.realign` and returned.

    Parameters
    ----------
    pac : PAC
        PAC grid to orient.

    Returns
    -------
    PAC
        Selected D6 image, realigned.
    """
    grid_center = pac.grid_size // 2
    is_occupied = pac.framework.toarray().reshape(-1) != 0
    tile_index_maps = _d6_tile_indices(pac)

    images_criteria = []
    for tile_indices in tile_index_maps:
        bounding_box = pac.calculate_bounding_box(
            positions=_occupied_positions(tile_indices, is_occupied)
        )
        images_criteria.append(
            (
                bounding_box.row_max - bounding_box.row_min,
                abs(grid_center - bounding_box.row_min)
                + abs(bounding_box.row_max - grid_center),
                bounding_box.column_max - bounding_box.column_min,
                abs(grid_center - bounding_box.column_min)
                + abs(bounding_box.column_max - grid_center),
            )
        )
    min_index = images_criteria.index(min(images_criteria))
    image = _d6_image(pac, tile_index_maps[min_index], min_index)
    if min_index == 0:
        image.realign()
    return image


class _DeadEnd(ValueError):
    """
    Indicate that the current decision vector cannot complete an embedding.
    """


class _Chooser:
    """
    Replay a fixed sequence of embedding decisions.

    The first decision selects the root ring. Later decisions select the empty vertex of a fused 5-membered ring or the
    path, tile, and orientation used to embed a linker.
    """

    def __init__(self, choices):
        self.choices = choices
        self.counts = []
        self.path = []

    def pick(self, n_options):
        index = len(self.counts)
        self.counts.append(n_options)
        if n_options <= 0:
            raise _DeadEnd(f"Decision {index} cannot select an option")
        chosen = self.choices[index] if index < len(self.choices) else 0
        if chosen >= n_options:
            raise _DeadEnd(f"Decision {index} exhausted")
        self.path.append(chosen)
        return chosen


def _next_choice_vector(path, counts):
    """
    Advance the deepest decision that has an untested alternative.
    """
    for i in range(len(path) - 1, -1, -1):
        if path[i] + 1 < counts[i]:
            return path[:i] + [path[i] + 1]
    return None


def nx_to_pac(
    g: nx.Graph,
    grid_size: int | None = None,
    validator: Callable[[PAC], bool] | None = None,
    max_attempts: int = 400,
) -> PAC:
    """
    Embed a labelled molecular graph on the hexagonal PAC grid.

    Candidate ring and linker placements are explored by backtracking in 2 passes. The first pass places every
    5-membered ring that is fused to 2 rings, which are not fused to each other, in the straight orientation. If the
    first pass does not build a PAC grid that passes `validator`, the second pass also accepts the angular orientation
    of these rings. Valid embeddings are compared by molecular bounding-box area, and the canonical orientation is
    returned.

    Parameters
    ----------
    g : networkx.Graph
        Labelled molecular graph to embed.
    grid_size : int or None, default None
        Grid side length. The project default is used when this is None.
    validator : callable or None, default None
        Optional function that checks whether the completed grid reconstructs the input molecule.
    max_attempts : int, default 400
        Maximum number of embedding attempts in each pass.

    Returns
    -------
    PAC
        Canonically oriented PAC grid. When none of the built PAC grids passes `validator`, the first PAC grid that was
        built is returned.

    Raises
    ------
    ValueError
        Raised when neither pass builds a PAC grid.
    """
    first_built = None
    last_error = None

    for require_straight_five_membered_rings in (True, False):
        choices = []
        for _ in range(max_attempts):
            chooser = _Chooser(choices)
            pac = None
            try:
                pac = _nx_to_pac_choices(
                    g,
                    grid_size=grid_size,
                    chooser=chooser,
                    require_straight_five_membered_rings=require_straight_five_membered_rings,
                )
            except Exception as e:
                last_error = e

            if pac is not None:
                if validator is None:
                    return pac
                if first_built is None:
                    first_built = pac
                try:
                    if validator(pac):
                        return pac
                except Exception as e:
                    last_error = e

            choices = _next_choice_vector(chooser.path, chooser.counts)
            if choices is None:
                break

    if first_built is not None:
        return first_built
    raise ValueError(
        "A PAC embedding was not found for this molecule. Both the straight and the angular placement of the fused "
        "5-membered rings failed."
    ) from last_error


DEFAULT_MAX_EMBEDDING_ATTEMPTS = (
    10  # 10 is pretty fast, but more likely to fail. 400 is a decent middle ground.
)


def mol_to_pac(
    mol: Chem.Mol,
    grid_size: int | None = None,
    max_attempts: int = DEFAULT_MAX_EMBEDDING_ATTEMPTS,
) -> PAC:
    """
    Convert an RDKit molecule into a canonical PAC grid.

    The embedding search prefers a grid that decodes back to the canonical SMILES of the input. If none of the built
    grids matches, `nx_to_pac` returns the first built grid.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule to convert.
    grid_size : int or None, default None
        Grid side length. The project default is used when this is None.
    max_attempts : int, default DEFAULT_MAX_EMBEDDING_ATTEMPTS
        Maximum number of embedding attempts in each pass of `nx_to_pac`.

    Returns
    -------
    PAC
        Canonically oriented PAC grid.
    """
    if grid_size is None:
        grid_size = const.GRID_SIZE
    g = mol_to_nx(mol)

    target = Chem.MolToSmiles(mol)

    def _round_trips(pac: PAC) -> bool:
        rebuilt = Chem.MolFromSmiles(pac.smiles)
        return rebuilt is not None and Chem.MolToSmiles(rebuilt) == target

    return nx_to_pac(
        g, grid_size=grid_size, validator=_round_trips, max_attempts=max_attempts
    )


def smiles_to_pac(
    smiles: str,
    grid_size: int | None = None,
    max_attempts: int = DEFAULT_MAX_EMBEDDING_ATTEMPTS,
) -> PAC:
    """
    Convert a SMILES string into a canonical PAC grid.

    Parameters
    ----------
    smiles : str
        Molecular structure to convert.
    grid_size : int or None, default None
        Grid side length. The project default is used when this is None.
    max_attempts : int, default DEFAULT_MAX_EMBEDDING_ATTEMPTS
        Maximum number of embedding attempts in each pass of `nx_to_pac`.

    Returns
    -------
    PAC
        Canonically oriented PAC grid, with the occupied framework centred by `PAC.realign`.

    Raises
    ------
    ValueError
        Raised when RDKit cannot parse the SMILES string or the molecular graph cannot be embedded.
    """
    if grid_size is None:
        grid_size = const.GRID_SIZE
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES is invalid: {smiles}")
    return mol_to_pac(mol, grid_size=grid_size, max_attempts=max_attempts)


# Largest number of SMILES strings kept by `encode_grid`
ENCODED_GRID_CACHE_SIZE = 8192


@lru_cache(maxsize=ENCODED_GRID_CACHE_SIZE)
def encode_grid(smiles: str) -> torch.Tensor:
    """Cache the canonical PAC grid for a SMILES string as `torch.uint8`.

    Callers must not modify the cached tensor in place.

    Parameters
    ----------
    smiles : str
        Molecular structure to encode.

    Returns
    -------
    torch.Tensor
        Encoded grid in the channel order of `PAC.to_tensor`, as `torch.uint8`.

    Raises
    ------
    ValueError
        Raised when encoding fails. The exception is forwarded from `smiles_to_pac`.
    """
    return smiles_to_pac(smiles).to_tensor().to(torch.uint8)


class Decoding(NamedTuple):
    """
    Result of the auto-correction of 1 valid raw grid.

    Attributes
    ----------
    smiles : str
        SMILES string of the decoded molecule.
    decoded_grid : torch.Tensor of shape (13, grid_size, grid_size)
        Grid of the decoded `PAC` object after graph reconstruction, as `torch.uint8`.
    corrected_grid : torch.Tensor of shape (13, grid_size, grid_size)
        Grid of `encode_grid` for the SMILES string, as `torch.uint8`.
    """

    smiles: str
    decoded_grid: torch.Tensor
    corrected_grid: torch.Tensor


def decode_grid(grid: torch.Tensor) -> tuple[PAC, str] | None:
    """
    Convert a raw grid to a `PAC` object and the SMILES string of the molecule.

    The inverse conversion is `encode_grid`. The SMILES conversion reconstructs the molecular graph, which can modify
    the returned `PAC` object.

    Parameters
    ----------
    grid : torch.Tensor of shape (13, grid_size, grid_size)
        Raw grid.

    Returns
    -------
    tuple of (PAC, str) or None
        The `PAC` object and the SMILES string, or None when the conversion raises an exception or produces an empty
        SMILES string.
    """
    try:
        pac = PAC.from_tensor(grid)
        smiles = pac.smiles
    except Exception:
        return None
    if not smiles:
        return None
    return pac, smiles


def autocorrect_grid(grid: torch.Tensor) -> Decoding | None:
    """
    Decode a raw grid with `decode_grid` and encode the SMILES string as the corrected grid with `encode_grid`.

    Parameters
    ----------
    grid : torch.Tensor of shape (13, grid_size, grid_size)
        Raw grid.

    Returns
    -------
    Decoding or None
        The SMILES string, the decoded grid, and the corrected grid, or None when the raw grid cannot be decoded or the
        decoded molecule cannot be encoded.
    """
    decoded = decode_grid(grid)
    if decoded is None:
        return None
    pac, smiles = decoded
    try:
        corrected_grid = encode_grid(smiles)
    except ValueError:
        return None
    return Decoding(smiles, pac.to_tensor().to(torch.uint8), corrected_grid)
