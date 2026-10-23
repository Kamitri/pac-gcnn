"""PAC grid storage, molecular graph decoding, and D6 transformations."""

from copy import deepcopy
from typing import Any, Literal, NamedTuple, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import torch
from rdkit import Chem
from rdkit.Chem import BondType, HybridizationType, rdDepictor
from scipy.sparse import lil_matrix as matrix

import const
from pac.action import Action, get_possible_actions, take_action
from pac.enums import (
    TRANSFORMATIONS,
    AtomType,
    EdgeDirection,
    Framework,
    FunctionalGroup,
    VertexDirection,
)
from pac.geometry import (
    periodic_hexagonal_distance,
    reflected_channel_index,
    rotated_channel_index,
)
from pac.matrix_manipulation import reflect, rotate
from utils import get_fingerprint

rdDepictor.SetPreferCoordGen(True)


def _generate_str_for_serialization(raw: list[str]) -> str:
    """
    Apply run-length encoding to a flattened PAC grid channel.

    Consecutive equal values are encoded as ``value.count`` and separated by commas.

    Parameters
    ----------
    raw : list of str
        Non-empty sequence of string values from a channel in row-major order. Values must not contain a period or
        comma.

    Returns
    -------
    str
        Comma-separated run-length encoding of `raw`.

    Examples
    --------
    >>> _generate_str_for_serialization(["0", "0", "0", "2", "0"])
    '0.3,2.1,0.1'
    """
    result = []
    current_value = raw[0]
    current_value_count = 0
    for value in raw:
        if value == current_value:
            current_value_count += 1
        else:
            result.append(f"{current_value}.{current_value_count}")
            current_value = value
            current_value_count = 1
    result.append(f"{current_value}.{current_value_count}")
    return ",".join(result)


def _generate_arr_values_from_serialization(text: str) -> npt.NDArray[np.int8]:
    """
    Decode a run-length-encoded PAC grid channel.

    Parameters
    ----------
    text : str
        Channel string produced by `_generate_str_for_serialization`. Encoded values must be integers.

    Returns
    -------
    numpy.ndarray of shape (num_values,)
        Decoded channel values stored as int8.
    """
    result = []
    str_values = text.split(",")
    for str_value in str_values:
        value, count = str_value.split(".")
        result.extend([int(value)] * int(count))
    return np.array(result, dtype=np.int8)


def nx_to_mol(G: nx.Graph) -> Chem.RWMol:
    """
    Construct an editable RDKit molecule from a labelled NetworkX graph.

    Atomic numbers, hybridisation states, formal charges, and bond types are obtained from the graph attributes. Each
    RDKit atom is also assigned its original `pac_node_id`.

    Parameters
    ----------
    G : networkx.Graph
        Molecular graph with the atom and bond attributes used by the PAC representation.

    Returns
    -------
    rdkit.Chem.RWMol
        Unsanitised editable molecule.
    """
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, "atomic_number")
    node_hybridizations = nx.get_node_attributes(G, "hybridization")
    node_charges = nx.get_node_attributes(G, "charge")
    node_to_idx = {}

    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        if node_hybridizations.get(node) is not None:
            a.SetHybridization(node_hybridizations[node])
        if node_charges.get(node) is not None:
            a.SetFormalCharge(node_charges[node])
        idx = mol.AddAtom(a)

        mol.GetAtomWithIdx(idx).SetIntProp("pac_node_id", node)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, "bond_type")
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    return mol


def add_ring_indexes_to_adjacency_matrix(adj_matrix: list, indexes: list[int]) -> None:
    """
    Add the bonds of a closed ring to an edge list in place.

    The input must describe a complete ring and cannot contain an empty vertex represented by None.

    Parameters
    ----------
    adj_matrix : list
        Existing edge list to modify.
    indexes : list of int
        Atom indices in cyclic order.
    """
    adj_matrix += [
        (indexes[i], indexes[(i + 1) % len(indexes)]) for i in range(len(indexes))
    ]


class RingRegistry:
    """Store the atom indices of the six vertices of a hexagon in clockwise order from the top.

    `None` marks an empty vertex. A five-membered ring has 1 empty vertex, whereas a linker can have several.

    Parameters
    ----------
    indexes : list of int or None
        Six atom indices or None values. The input list is stored by reference.

    Attributes
    ----------
    indexes : list of int or None
        Atom indices or None values.
    """

    def __init__(self, indexes: list[int | None]) -> None:
        self.indexes = indexes

    @property
    def top(self) -> int | None:
        """Return the atom index at vertex 0, or None if empty."""
        return self.indexes[0]

    @property
    def top_right(self) -> int | None:
        """Return the atom index at vertex 1, or None if empty."""
        return self.indexes[1]

    @property
    def bottom_right(self) -> int | None:
        """Return the atom index at vertex 2, or None if empty."""
        return self.indexes[2]

    @property
    def bottom(self) -> int | None:
        """Return the atom index at vertex 3, or None if empty."""
        return self.indexes[3]

    @property
    def bottom_left(self) -> int | None:
        """Return the atom index at vertex 4, or None if empty."""
        return self.indexes[4]

    @property
    def top_left(self) -> int | None:
        """Return the atom index at vertex 5, or None if empty."""
        return self.indexes[5]

    def iter_clockwise(self) -> tuple[tuple[int | None, int | None], ...]:
        """Return six clockwise atom-index pairs along the edges of the hexagon."""
        return (
            (self.indexes[0], self.indexes[1]),
            (self.indexes[1], self.indexes[2]),
            (self.indexes[2], self.indexes[3]),
            (self.indexes[3], self.indexes[4]),
            (self.indexes[4], self.indexes[5]),
            (self.indexes[5], self.indexes[0]),
        )

    def __getitem__(self, item: int) -> int | None:
        return self.indexes[item]

    def __repr__(self) -> str:
        return f"RingRegistry(indexes={self.indexes})"


class MoleculeIndexRegistry:
    """
    Accumulate atom indices, bonds, and attributes during graph reconstruction.

    The registry numbers the atoms as the tiles are processed and records the bonds and the attributes of the NetworkX
    graph. An atom shared by fused rings has 1 molecular index at all coinciding vertices.

    The state applies to 1 grid conversion, and ring and linker tiles must be registered before their atom types or
    functional groups are assigned. ``atom_index in registry`` is True for an atom of a registered ring or linker tile,
    and ``registry[(column, row)]`` returns the `RingRegistry` of the tile, or None for an unregistered tile. Tile
    coordinates outside the grid must be wrapped by the caller.

    Parameters
    ----------
    grid_size : int or None, default None
        Number of tiles along each grid axis. `const.GRID_SIZE` is used when this is None.

    Attributes
    ----------
    grid_size : int
        Number of tiles along each grid axis.
    node_count : int
        Number of atoms currently registered and index assigned to the next new atom.
    indexes : dict of tuple of int to RingRegistry
        Mapping from wrapped tile coordinates to the six atom indices on that tile.
    index_to_ring : dict of int to list of tuple of int
        Mapping from each framework atom to its containing ring or linker tiles. Functional group atoms are not
        included.
    functional_groups_indexes : dict of int to tuple
        Mapping from the attachment atom to the atom indices and enum value of its functional group. A later functional
        group at the same attachment atom replaces the existing functional group.
    adjacency_matrix : list of tuple of int
        Molecular edge list in insertion order. Despite the attribute name, this is not an adjacency matrix.
    node_attributes : dict of int to dict
        NetworkX node attributes indexed by molecular atom index.
    edge_attributes : dict of tuple of int to rdkit.Chem.BondType
        RDKit bond types indexed by directed atom-index pairs.
    edge_to_framework : dict of tuple of int to set of int
        Framework values associated with each undirected bond. A fused bond may be associated with both ring tiles.
    """

    def __init__(self, grid_size: int | None = None) -> None:
        self.grid_size = grid_size if grid_size is not None else const.GRID_SIZE
        self.node_count = 0
        self.indexes = {}
        self.index_to_ring: dict[int, list[tuple[int, int]]] = {}
        self.functional_groups_indexes = {}
        self.adjacency_matrix = []
        self.node_attributes = {}
        self.edge_attributes = {}
        self.edge_to_framework = {}

    def add_ring(
        self,
        column_index: int,
        row_index: int,
        indexes: list[int | None],
        framework_type: int = 1,
    ) -> None:
        """
        Register a ring tile and its bonds.

        Indices greater than or equal to `node_count` are registered as new carbon atoms. Lower indices refer to atoms
        already introduced by an adjacent fused ring. The atoms are connected in a cycle, provisionally assigned
        aromatic bonds, and associated with this tile.

        In a 5-membered ring, the 2 atoms next to the empty vertex are bonded directly.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        indexes : list of int or None
            Six atom indices in clockwise order, with None at an empty vertex. New atoms must be numbered consecutively
            from `node_count`.
        framework_type : int, default 1
            Framework value associated with each bond in the ring.
        """
        new_indexes = [i for i in indexes if i is not None and i >= self.node_count]
        column_index = column_index % self.grid_size
        row_index = row_index % self.grid_size

        ring = RingRegistry(indexes)
        self.indexes[(column_index, row_index)] = ring

        valid_indexes = [i for i in indexes if i is not None]

        for i in range(len(valid_indexes)):
            u = valid_indexes[i]
            v = valid_indexes[(i + 1) % len(valid_indexes)]
            self.adjacency_matrix.append((u, v))
            self.edge_attributes[(u, v)] = BondType.AROMATIC

            edge_sorted = tuple(sorted((u, v)))
            if edge_sorted not in self.edge_to_framework:
                self.edge_to_framework[edge_sorted] = set()
            self.edge_to_framework[edge_sorted].add(framework_type)

        self.node_count += len(new_indexes)
        self.node_attributes.update({i: {"atomic_number": 6} for i in new_indexes})

        for index in valid_indexes:
            if index not in self.index_to_ring:
                self.index_to_ring[index] = []
            self.index_to_ring[index].append((column_index, row_index))

    def add_linker(
        self, column_index: int, row_index: int, indexes: list[int | None]
    ) -> None:
        """
        Register a linker tile as an open chain of atoms.

        Adjacent non-empty vertices are connected without closing a cycle. Each empty vertex breaks the chain. The bonds
        are provisionally assigned an aromatic type and associated with linker framework value 3 before final bond
        assignment in `PAC.graph`.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        indexes : list of int or None
            Six atom indices in clockwise order, with None at every empty vertex. New atoms must be numbered
            consecutively from `node_count`.
        """
        new_indexes = [i for i in indexes if i is not None and i >= self.node_count]
        column_index = column_index % self.grid_size
        row_index = row_index % self.grid_size

        ring = RingRegistry(indexes)
        self.indexes[(column_index, row_index)] = ring

        for i in range(6):
            node_a = indexes[i]
            node_b = indexes[(i + 1) % 6]
            if node_a is not None and node_b is not None:
                self.adjacency_matrix.append((node_a, node_b))
                self.edge_attributes[(node_a, node_b)] = BondType.AROMATIC

                edge_sorted = tuple(sorted((node_a, node_b)))
                if edge_sorted not in self.edge_to_framework:
                    self.edge_to_framework[edge_sorted] = set()
                self.edge_to_framework[edge_sorted].add(3)

        self.node_count += len(new_indexes)
        self.node_attributes.update({i: {"atomic_number": 6} for i in new_indexes})

        valid_indexes = [i for i in indexes if i is not None]
        for index in valid_indexes:
            if index not in self.index_to_ring:
                self.index_to_ring[index] = []
            self.index_to_ring[index].append((column_index, row_index))

    def add_cyanide(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach a cyano group to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
            (self.node_count, self.node_count + 1),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.edge_attributes[(self.node_count, self.node_count + 1)] = BondType.TRIPLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP,
        }
        self.node_attributes[self.node_count + 1] = {
            "atomic_number": 7,
            "hybridization": Chem.HybridizationType.SP,
        }
        self.functional_groups_indexes.update(
            {
                atom_index: (
                    [self.node_count, self.node_count + 1],
                    FunctionalGroup.CYANIDE,
                )
            }
        )
        self.node_count += 2

    def add_nitro(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach a nitro group to a framework atom.

        The nitro group is added in the charge-separated form.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
            (self.node_count, self.node_count + 1),
            (self.node_count, self.node_count + 2),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.edge_attributes[(self.node_count, self.node_count + 1)] = BondType.DOUBLE
        self.edge_attributes[(self.node_count, self.node_count + 2)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 7,
            "hybridization": Chem.HybridizationType.SP2,
            "charge": +1,
        }
        self.node_attributes[self.node_count + 1] = {
            "atomic_number": 8,
            "hybridization": Chem.HybridizationType.SP2,
        }
        self.node_attributes[self.node_count + 2] = {
            "atomic_number": 8,
            "hybridization": Chem.HybridizationType.SP3,
            "charge": -1,
        }
        self.functional_groups_indexes.update(
            {
                atom_index: (
                    [self.node_count, self.node_count + 1, self.node_count + 2],
                    FunctionalGroup.NITRO,
                )
            }
        )
        self.node_count += 3

    def add_dimethylamino(
        self, column_index: int, row_index: int, direction: int
    ) -> None:
        """
        Attach a dimethylamino group to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
            (self.node_count, self.node_count + 1),
            (self.node_count, self.node_count + 2),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.edge_attributes[(self.node_count, self.node_count + 1)] = BondType.SINGLE
        self.edge_attributes[(self.node_count, self.node_count + 2)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 7,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.node_attributes[self.node_count + 1] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.node_attributes[self.node_count + 2] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {
                atom_index: (
                    [self.node_count, self.node_count + 1, self.node_count + 2],
                    FunctionalGroup.DIMETHYLAMINO,
                )
            }
        )
        self.node_count += 3

    def add_methyl(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach a methyl group to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {atom_index: ([self.node_count], FunctionalGroup.METHYL)}
        )
        self.node_count += 1

    def add_hydrogen(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach an explicit hydrogen to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 1,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {atom_index: ([self.node_count], FunctionalGroup.HYDROGEN)}
        )
        self.node_count += 1

    def add_fluoride(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach fluorine to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 9,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {atom_index: ([self.node_count], FunctionalGroup.FLUORIDE)}
        )
        self.node_count += 1

    def add_chloride(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach chlorine to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 17,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {atom_index: ([self.node_count], FunctionalGroup.CHLORIDE)}
        )
        self.node_count += 1

    def add_ethyl(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach an ethyl group to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
            (self.node_count, self.node_count + 1),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.edge_attributes[(self.node_count, self.node_count + 1)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.node_attributes[self.node_count + 1] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {
                atom_index: (
                    [self.node_count, self.node_count + 1],
                    FunctionalGroup.ETHYL,
                )
            }
        )
        self.node_count += 2

    def add_methoxy(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach a methoxy group to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
            (self.node_count, self.node_count + 1),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.edge_attributes[(self.node_count, self.node_count + 1)] = BondType.SINGLE
        self.node_attributes[self.node_count] = {
            "atomic_number": 8,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.node_attributes[self.node_count + 1] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {
                atom_index: (
                    [self.node_count, self.node_count + 1],
                    FunctionalGroup.METHOXY,
                )
            }
        )
        self.node_count += 2

    def add_ethoxy(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Attach an ethoxy group to a framework atom.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
            (self.node_count, self.node_count + 1),
            (self.node_count + 1, self.node_count + 2),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.SINGLE
        self.edge_attributes[(self.node_count, self.node_count + 1)] = BondType.SINGLE
        self.edge_attributes[(self.node_count + 1, self.node_count + 2)] = (
            BondType.SINGLE
        )
        self.node_attributes[self.node_count] = {
            "atomic_number": 8,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.node_attributes[self.node_count + 1] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.node_attributes[self.node_count + 2] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP3,
        }
        self.functional_groups_indexes.update(
            {
                atom_index: (
                    [self.node_count, self.node_count + 1, self.node_count + 2],
                    FunctionalGroup.ETHOXY,
                )
            }
        )
        self.node_count += 3

    def add_carbonyl(self, column_index: int, row_index: int, direction: int) -> None:
        """
        Convert a framework atom into a carbonyl group.

        The attachment atom is reassigned as sp2 carbon, and its attribute dictionary is replaced. Carbonyl addition is
        the only functional group operation that changes the framework atom itself, and `_update_aromaticity`
        subsequently treats the entire ring tile as non-aromatic.

        The functional-group registry records 2 atom indices even though this method adds 1 atom. The second index
        therefore refers to the next atom created by the registry.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        """
        ring = self.indexes[(column_index % self.grid_size, row_index % self.grid_size)]
        atom_index = ring[direction]
        self.adjacency_matrix += [
            (atom_index, self.node_count),
        ]
        self.edge_attributes[(atom_index, self.node_count)] = BondType.DOUBLE
        self.node_attributes[atom_index] = {
            "atomic_number": 6,
            "hybridization": Chem.HybridizationType.SP2,
        }
        self.node_attributes[self.node_count] = {
            "atomic_number": 8,
            "hybridization": Chem.HybridizationType.SP2,
        }
        self.functional_groups_indexes.update(
            {
                atom_index: (
                    [self.node_count, self.node_count + 1],
                    FunctionalGroup.CARBONYL,
                )
            }
        )
        self.node_count += 1

    def add_functional_group(
        self,
        column_index: int,
        row_index: int,
        direction: int,
        functional_group: FunctionalGroup,
    ) -> None:
        """
        Attach the functional group encoded by a channel value.

        Values outside `FunctionalGroup` are ignored.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Attachment vertex, indexed clockwise from 0 at the top.
        functional_group : FunctionalGroup
            Functional group encoded in channels 1-6.
        """
        if functional_group == FunctionalGroup.HYDROGEN:
            self.add_hydrogen(column_index, row_index, direction)
            return
        if functional_group == FunctionalGroup.NITRO:
            self.add_nitro(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.CYANIDE:
            self.add_cyanide(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.DIMETHYLAMINO:
            self.add_dimethylamino(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.CARBONYL:
            self.add_carbonyl(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.METHYL:
            self.add_methyl(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.ETHYL:
            self.add_ethyl(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.METHOXY:
            self.add_methoxy(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.ETHOXY:
            self.add_ethoxy(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.FLUORIDE:
            self.add_fluoride(column_index, row_index, direction)
            return
        elif functional_group == FunctionalGroup.CHLORIDE:
            self.add_chloride(column_index, row_index, direction)
            return

    def update_atom_type(
        self, column_index: int, row_index: int, direction: int, atom_type: AtomType
    ) -> None:
        """
        Update the element and hybridisation of a registered framework atom.

        The molecular index and existing bonds are preserved while the atom attributes are updated from `atom_type`. The
        element is assigned from the carbon, nitrogen, oxygen, or sulfur enum member. Hybridisation is set to sp3 for
        `CARBON_SP3` and `NITROGEN_SP3`, while other atom types receive sp2 hybridisation only if it has not already
        been assigned. Aromatic carbon and sp2 nitrogen types are marked aromatic. The original enum value is retained
        as `pac_atom_type` for later bond assignment.

        Bond types are adjusted after the atom attributes. Bonds between 2 sp3 atoms become single, whereas a bond
        from an sp3 atom to a different type is provisionally aromatic. Selected bonds between sp2 heteroatoms are
        changed to single bonds when a functional group is already attached. `PAC.graph` performs the final kekulisation
        after all grid channels have been processed.

        The method returns without changing the registry when the tile has not been registered or the selected vertex is
        empty.

        Parameters
        ----------
        column_index : int
            Tile column, wrapped modulo `grid_size`.
        row_index : int
            Tile row, wrapped modulo `grid_size`.
        direction : int
            Vertex to update, indexed clockwise from 0 at the top.
        atom_type : AtomType
            Atom type encoded in channels 7-12.

        Raises
        ------
        ValueError
            Raised when `atom_type` is not an `AtomType` member. Zero is also invalid because it is an empty vertex.
        """
        ring = self.indexes.get(
            (column_index % self.grid_size, row_index % self.grid_size)
        )
        if ring is None:
            return
        atom_index = ring[direction]
        if atom_index is None:
            return
        hybridization = Chem.HybridizationType.SP2
        if atom_type in [
            AtomType.CARBON_AROMATIC,
            AtomType.CARBON_AROMATIC_5_MEMBERED,
            AtomType.CARBON_SP2,
            AtomType.CARBON_SP3,
        ]:
            atomic_number = 6
            if atom_type == AtomType.CARBON_SP3:
                hybridization = Chem.HybridizationType.SP3
        elif atom_type in [
            AtomType.NITROGEN_SP2_1PI,
            AtomType.NITROGEN_SP2_2PI,
            AtomType.NITROGEN_SP3,
        ]:
            atomic_number = 7
            if atom_type == AtomType.NITROGEN_SP3:
                hybridization = Chem.HybridizationType.SP3
        elif atom_type in [AtomType.OXYGEN_SP3]:
            atomic_number = 8
        elif atom_type in [AtomType.SULFUR_SP3]:
            atomic_number = 16
        else:
            raise ValueError(f"Invalid atom type: {atom_type}")

        self.node_attributes[atom_index]["atomic_number"] = atomic_number
        self.node_attributes[atom_index]["pac_atom_type"] = (
            atom_type.value
        )  # Store this for Kekulizer

        if hybridization == Chem.HybridizationType.SP3:
            self.node_attributes[atom_index]["hybridization"] = (
                Chem.HybridizationType.SP3
            )
        elif "hybridization" not in self.node_attributes[atom_index]:
            self.node_attributes[atom_index]["hybridization"] = hybridization

        is_aromatic_new = atom_type in [
            AtomType.CARBON_AROMATIC,
            AtomType.CARBON_AROMATIC_5_MEMBERED,
        ] or atom_type in [AtomType.NITROGEN_SP2_1PI, AtomType.NITROGEN_SP2_2PI]
        current_is_aromatic = self.node_attributes[atom_index].get("is_aromatic", False)
        self.node_attributes[atom_index]["is_aromatic"] = (
            current_is_aromatic or is_aromatic_new
        )

        for edge in self.adjacency_matrix:
            if atom_index not in edge:
                continue
            other_atom_index = edge[0] if edge[1] == atom_index else edge[1]
            other_atom_attributes = self.node_attributes.get(other_atom_index, {})
            if hybridization == HybridizationType.SP3:
                if (
                    other_atom_attributes.get("hybridization")
                    == Chem.HybridizationType.SP3
                ):
                    self.edge_attributes[edge] = BondType.SINGLE
                else:
                    self.edge_attributes[edge] = BondType.AROMATIC
            if hybridization == HybridizationType.SP2 and atomic_number != 6:
                if (
                    other_atom_attributes.get("hybridization")
                    == Chem.HybridizationType.SP2
                ):
                    functional_group_entry = self.functional_groups_indexes.get(
                        atom_index
                    )
                    if functional_group_entry is None:
                        continue
                    functional_group_atoms, _ = functional_group_entry
                    if other_atom_index in functional_group_atoms:
                        continue
                    self.edge_attributes[edge] = BondType.SINGLE

    def locate_owner(self, index: int) -> list[tuple[int, int]] | None:
        """
        Return the tiles containing a framework atom.

        A fused-ring atom may occur on several tiles. The tile on which the atom was first registered appears first.

        Parameters
        ----------
        index : int
            Atom index.

        Returns
        -------
        list of tuple of int or None
            Coordinates of every tile containing the atom. The result is None for an index that is not associated with a
            ring or linker tile.
        """
        return self.index_to_ring.get(index, None)

    def locate_vertex_directions(
        self, atom_index: int
    ) -> dict[tuple[int, int], VertexDirection] | None:
        """
        Return the vertex occupied by an atom in each associated hexagon.

        Parameters
        ----------
        atom_index : int
            Atom index.

        Returns
        -------
        dict of tuple of int to VertexDirection or None
            Mapping from tile coordinates to the corresponding vertex direction. The result is None when the atom is not
            associated with a framework tile.
        """
        rings_xy = self.index_to_ring.get(atom_index, None)
        if rings_xy is None:
            return None
        result = dict()
        for ring_xy in rings_xy:
            index_of_atom_index_in_this_ring = self.indexes[ring_xy].indexes.index(
                atom_index
            )
            if index_of_atom_index_in_this_ring == 0:
                vertex_direction = VertexDirection.TOP
            elif index_of_atom_index_in_this_ring == 1:
                vertex_direction = VertexDirection.TOP_RIGHT
            elif index_of_atom_index_in_this_ring == 2:
                vertex_direction = VertexDirection.BOTTOM_RIGHT
            elif index_of_atom_index_in_this_ring == 3:
                vertex_direction = VertexDirection.BOTTOM
            elif index_of_atom_index_in_this_ring == 4:
                vertex_direction = VertexDirection.BOTTOM_LEFT
            else:
                vertex_direction = VertexDirection.TOP_LEFT
            result[ring_xy] = vertex_direction
        return result

    def __contains__(self, item: int) -> bool:
        return item in self.index_to_ring.keys()

    def __getitem__(self, item: tuple[int, int]) -> RingRegistry | None:
        return self.indexes.get(item, None)

    def reset(self) -> None:
        """
        Reset the accumulated molecular graph state.

        Atom numbering, bonds, and attributes are cleared while the grid size is preserved. The next atom registered
        receives index 0.
        """
        self.__init__(grid_size=self.grid_size)


class GridBoundingBox(NamedTuple):
    """
    Describe the axial extent of the occupied PAC grid tiles.

    Attributes
    ----------
    column_min : int
        First occupied column along the shortest periodic interval.
    column_max : int
        Last occupied column along the shortest periodic interval.
    row_min : int
        First occupied row along the shortest periodic interval.
    row_max : int
        Last occupied row along the shortest periodic interval.
    """

    column_min: int
    column_max: int
    row_min: int
    row_max: int


class PACMatrix(matrix):
    """
    Store a sparse PAC grid channel with periodic coordinate indexing.

    Coordinate tuples use ``(column, row)`` order instead of the ``(row, column)`` order of the base sparse matrix
    class, and both coordinates wrap at the PAC grid boundary. An integer index has the meaning of the base class, and
    any other key raises IndexError.
    """

    def __getitem__(self, key: tuple[int, int] | int) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            column_index, row_index = key
            return super().__getitem__(
                (row_index % self.shape[1], column_index % self.shape[0])
            )
        elif isinstance(key, int) or isinstance(key, np.integer):
            return super().__getitem__(key)
        raise IndexError(
            "Item must be a tuple of (column_index, row_index) coordinates or a single int."
        )

    # SciPy calls 2nd parameter x; [] passes it by position
    def __setitem__(self, key: tuple[int, int] | int, value: Any) -> None:  # pyrefly: ignore[bad-override-param-name]
        if isinstance(key, tuple) and len(key) == 2:
            column_index, row_index = key
            super().__setitem__(
                (row_index % self.shape[1], column_index % self.shape[0]), value
            )
        elif isinstance(key, int):
            super().__setitem__(key, value)
        else:
            raise IndexError(
                "Key must be a tuple of (column_index, row_index) coordinates or a single int."
            )


class PAC:
    """
    Represent a polycyclic aromatic compound on a PAC grid.

    Channel 0 encodes the framework, channels 1 to 6 encode functional groups at the six vertices, and channels 7 to 12
    encode atom types and hybridisation states at the same vertices.

    Indexing with ``pac[channel, column, row]`` returns or sets the value of 1 channel at 1 tile, and the column and the
    row wrap modulo `grid_size`. An assignment of a ring to the framework channel sets the atom types of the occupied
    vertices to `CARBON_SP2` and the atom type of the empty vertex of a 5-membered ring to 0. The value of a 5-membered
    ring can be the pair ``(framework_value, empty_vertex)``, and the default empty vertex is the top vertex. An sp3
    carbon or a heteroatom assigned to 1 vertex is copied to every vertex of the same atom, while the ring-specific sp2
    and aromatic carbon types of the vertices stay independent.

    Parameters
    ----------
    grid_size : int or None, default None
        Number of tiles along each grid axis. `const.GRID_SIZE` is used when this is None.
    init_pos : tuple of int or None, default None
        Position of the initial 6-membered ring. The grid centre is used when this is None.

    Attributes
    ----------
    grid_size : int
        Number of tiles along each PAC grid axis.
    init_pos : tuple of int
        Stored position of the seed ring.
    framework : PACMatrix
        Framework channel of the PAC grid.
    functional_groups : list of PACMatrix
        Six functional-group channels in vertex order.
    atom_types : list of PACMatrix
        Six atom-type channels in vertex order.
    index_registry : MoleculeIndexRegistry
        Atom indices and bonds from the latest graph reconstruction.
    reg : MoleculeIndexRegistry
        Alias for `index_registry`.
    """

    def __init__(
        self,
        grid_size: int | None = None,
        init_pos: tuple[int, int] | None = None,
    ) -> None:
        self.grid_size = grid_size if grid_size is not None else const.GRID_SIZE
        self.framework = PACMatrix((self.grid_size, self.grid_size), dtype=np.int8)
        self.functional_groups = [
            PACMatrix((self.grid_size, self.grid_size), dtype=np.int8) for _ in range(6)
        ]
        self.atom_types = [
            PACMatrix((self.grid_size, self.grid_size), dtype=np.int8) for _ in range(6)
        ]
        if init_pos is None:
            self.init_pos = (self.grid_size // 2, self.grid_size // 2)
        else:
            self.init_pos = init_pos
        self[0, self.init_pos[0], self.init_pos[1]] = 1
        self.index_registry = MoleculeIndexRegistry(grid_size=grid_size)
        self.reg = self.index_registry

    def calculate_bounding_box(
        self,
        negative_indexing: bool = False,
        positions: set[tuple[int, int]] | None = None,
    ) -> GridBoundingBox:
        """
        Calculate the bounding box of the occupied framework tiles.

        The shortest periodic interval containing the molecule is calculated independently for rows and columns. A
        molecule that wraps around the grid boundary can therefore have a minimum index greater than its maximum.

        Parameters
        ----------
        negative_indexing : bool, default False
            Express the beginning of a wrapped interval as a negative index when True.
        positions : set of tuple of int or None, default None
            ``(column_index, row_index)`` positions to enclose. Every position with an occupied framework tile is
            enclosed when this is None.

        Returns
        -------
        GridBoundingBox
            Column and row limits of the shortest periodic bounding box.
        """
        if positions is None:
            row_non_zeros, column_non_zeros = self.framework.nonzero()
        else:
            column_non_zeros = np.array([column_index for column_index, _ in positions])
            row_non_zeros = np.array([row_index for _, row_index in positions])
        row_non_zeros = np.sort(np.unique(row_non_zeros))
        column_non_zeros = np.sort(np.unique(column_non_zeros))
        row_diff = np.diff(row_non_zeros)
        column_diff = np.diff(column_non_zeros)

        rows_wrap_around = not np.all(row_diff == 1)
        columns_wrap_around = not np.all(column_diff == 1)
        if rows_wrap_around:
            discontinuity_index = np.where(row_diff != 1)[0][0]
            row_min = row_non_zeros[discontinuity_index + 1]
            row_max = row_non_zeros[discontinuity_index]
        else:
            row_min = row_non_zeros[0]
            row_max = row_non_zeros[-1]
        if columns_wrap_around:
            discontinuity_index = np.where(column_diff != 1)[0][0]
            column_min = column_non_zeros[discontinuity_index + 1]
            column_max = column_non_zeros[discontinuity_index]
        else:
            column_min = column_non_zeros[0]
            column_max = column_non_zeros[-1]
        if negative_indexing:
            if row_min > row_max:
                row_min = row_min % -self.grid_size
            if column_min > column_max:
                column_min = column_min % -self.grid_size
        return GridBoundingBox(
            column_min=column_min.item(),
            column_max=column_max.item(),
            row_min=row_min.item(),
            row_max=row_max.item(),
        )

    def shift(
        self, column_index: int, row_index: int, edge_dir: EdgeDirection
    ) -> tuple[int, int]:
        """
        Return the coordinates of a neighbouring tile.

        Parameters
        ----------
        column_index : int
            Column coordinate of the starting tile. Values outside the grid are accepted.
        row_index : int
            Row coordinate of the starting tile.
        edge_dir : EdgeDirection
            Direction of the neighbour. The value must be an `EdgeDirection` member rather than a combination of flags.

        Returns
        -------
        tuple of int
            Wrapped column and row coordinates of the neighbour.
        """
        column_step, row_step = TRANSFORMATIONS[edge_dir]
        return (column_index + column_step) % self.grid_size, (
            row_index + row_step
        ) % self.grid_size

    def has_any_neighbour(
        self,
        channel: int,
        column_index: int,
        row_index: int,
        positions: EdgeDirection,
        mode: Literal["or", "and"] = "or",
    ) -> bool:
        """
        Test selected neighbouring tiles for non-zero channel values.

        Parameters
        ----------
        channel : int
            Grid channel to test. Channel 0 tests framework occupancy.
        column_index : int
            Column coordinate of the central tile.
        row_index : int
            Row coordinate of the central tile.
        positions : EdgeDirection
            Neighbour directions to test. Several flag members can be combined with ``|``.
        mode : {'or', 'and'}, default 'or'
            Logical operation applied across the selected directions.

        Returns
        -------
        bool
            Result of the requested logical test.

        Raises
        ------
        ValueError
            Raised when `mode` is not `"or"` or `"and"`.
        """
        if mode == "or":
            for position, (column_step, row_step) in TRANSFORMATIONS.items():
                if positions & position:
                    neighbour_column_index, neighbour_row_index = (
                        column_index + column_step,
                        row_index + row_step,
                    )
                    if self[channel, neighbour_column_index, neighbour_row_index] != 0:
                        return True
            return False
        elif mode == "and":
            for position in EdgeDirection:
                if positions & position:
                    column_step, row_step = TRANSFORMATIONS[position]
                    neighbour_column_index, neighbour_row_index = (
                        column_index + column_step,
                        row_index + row_step,
                    )
                    if self[channel, neighbour_column_index, neighbour_row_index] == 0:
                        return False
            return True
        raise ValueError(f'Invalid mode: {mode}. Expected "or" or "and".')

    def _update_aromaticity(self) -> None:
        """
        Reassign aromatic carbon types from the current grid structure.

        Structural edits can make the aromatic carbon codes on a ring tile inconsistent. The carbon type is recalculated
        on every occupied ring vertex, while linkers, empty tiles, and heteroatoms remain unchanged.

        A ring tile is treated as non-aromatic if any vertex of its hexagon is an sp3 carbon or has an attached carbonyl
        group. The remaining carbon vertices of the hexagon are assigned `CARBON_SP2`. Carbon vertices in the hexagons
        of all other ring tiles receive the aromatic code for the corresponding ring size. A carbonyl attachment atom is
        always assigned `CARBON_SP2`.

        `graph` applies this preliminary assignment before molecular graph reconstruction. Aromaticity is updated again
        after RDKit sanitisation.
        """
        # Scan all tiles because occupied bounding box can wrap across grid edges
        for column_index in range(self.grid_size):
            for row_index in range(self.grid_size):
                fw = self.framework[column_index, row_index]
                if fw not in (1, 2):
                    continue

                is_non_aromatic = False
                for i in range(6):
                    if fw == 2 and self.atom_types[i][column_index, row_index] == 0:
                        continue
                    if (
                        self.atom_types[i][column_index, row_index]
                        == AtomType.CARBON_SP3.value
                        or self.functional_groups[i][column_index, row_index]
                        == FunctionalGroup.CARBONYL.value
                    ):
                        is_non_aromatic = True
                        break

                target_aromatic = (
                    AtomType.CARBON_AROMATIC.value
                    if fw == 1
                    else AtomType.CARBON_AROMATIC_5_MEMBERED.value
                )

                for i in range(6):
                    if fw == 2 and self.atom_types[i][column_index, row_index] == 0:
                        continue

                    current_atom = self.atom_types[i][column_index, row_index]
                    if (
                        self.functional_groups[i][column_index, row_index]
                        == FunctionalGroup.CARBONYL.value
                    ):
                        self.atom_types[i][column_index, row_index] = (
                            AtomType.CARBON_SP2.value
                        )
                    elif not is_non_aromatic:
                        if current_atom in (
                            AtomType.CARBON_SP2.value,
                            AtomType.CARBON_AROMATIC.value,
                            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
                        ):
                            self.atom_types[i][column_index, row_index] = (
                                target_aromatic
                            )
                    else:
                        # Preserve sp2 codes on non-aromatic ring carbons
                        if current_atom in (
                            AtomType.CARBON_AROMATIC.value,
                            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
                            AtomType.CARBON_SP2.value,
                        ):
                            self.atom_types[i][column_index, row_index] = (
                                AtomType.CARBON_SP2.value
                            )

    def _framework_tile_components(self) -> list[set[tuple[int, int]]]:
        """
        Return the connected components of the occupied framework tiles on the torus.

        An atom is shared only by neighbouring tiles, and every bond lies within a single tile or joins a functional
        group to its tile so every piece of the molecular graph lies inside a single component.

        Returns
        -------
        list of set of tuple of int
            ``(column_index, row_index)`` positions of each component. The components are ordered by the smallest
            position they contain.
        """
        row_indexes, column_indexes = self.framework.nonzero()
        unassigned_positions = {
            (int(column_index), int(row_index))
            for row_index, column_index in zip(row_indexes, column_indexes)
        }
        components = []
        for start_position in sorted(unassigned_positions):
            if start_position not in unassigned_positions:
                continue
            unassigned_positions.remove(start_position)
            component = {start_position}
            frontier = [start_position]
            while frontier:
                column_index, row_index = frontier.pop()
                for edge_direction in TRANSFORMATIONS:
                    neighbour_position = self.shift(
                        column_index, row_index, edge_direction
                    )
                    if neighbour_position in unassigned_positions:
                        unassigned_positions.remove(neighbour_position)
                        component.add(neighbour_position)
                        frontier.append(neighbour_position)
            components.append(component)
        return components

    def _register_framework_tiles(self) -> None:
        """
        Register the atoms and bonds of every occupied framework tile.

        The molecular index registry is reset first. Each connected component of occupied framework tiles is then
        registered separately, in the order given by the bounding box of that component so the order within a
        component does not depend on external tiles. Within a component, the ring and linker tiles are registered
        from the bottom row upwards and from left to right within each row. The functional groups and atom types at each
        position are assigned immediately after its ring or linker tile is registered.

        Raises
        ------
        KeyError
            Raised when a component occupies every row and has neighbouring tiles in rows ``grid_size - 1`` and 0, or
            occupies every column and has neighbouring tiles in columns ``grid_size - 1`` and 0 of 1 row. In that order,
            a tile is registered before 1 of its left, bottom-left, or bottom-right neighbouring tiles.
        """
        self.index_registry.reset()
        for component in self._framework_tile_components():
            bounding_box = self.calculate_bounding_box(
                negative_indexing=True, positions=component
            )

            # Index lookup needs LEFT, BOTTOM_LEFT and BOTTOM_RIGHT neighbours registered first
            rows = np.array(range(bounding_box.row_min, bounding_box.row_max + 1))
            rows = np.where(rows < 0, rows + self.grid_size, rows)[::-1]
            cols = np.array(range(bounding_box.column_min, bounding_box.column_max + 1))
            cols = np.where(cols < 0, cols + self.grid_size, cols)

            for row in rows:
                for col in cols:
                    row_index, column_index = row, col
                    if (int(column_index), int(row_index)) not in component:
                        continue

                    missing_vertices = []
                    if self.framework[column_index, row_index] == 2:
                        for i in range(6):
                            if self.atom_types[i][column_index, row_index] == 0:
                                missing_vertices.append(i)
                                break
                    elif self.framework[column_index, row_index] == 3:
                        for i in range(6):
                            if self.atom_types[i][column_index, row_index] == 0:
                                missing_vertices.append(i)

                    template: list[int | None] = [None] * 6

                    if self.has_any_neighbour(
                        0, column_index, row_index, EdgeDirection.LEFT
                    ):
                        shared_l = self.get_ring_joint_indexes(
                            *self.shift(column_index, row_index, EdgeDirection.LEFT),
                            EdgeDirection.RIGHT,
                        )
                        if shared_l[0] is not None:
                            template[5] = shared_l[0]
                        if shared_l[1] is not None:
                            template[4] = shared_l[1]

                    if self.has_any_neighbour(
                        0, column_index, row_index, EdgeDirection.BOTTOM_LEFT
                    ):
                        shared_bl = self.get_ring_joint_indexes(
                            *self.shift(
                                column_index, row_index, EdgeDirection.BOTTOM_LEFT
                            ),
                            EdgeDirection.TOP_RIGHT,
                        )
                        if shared_bl[0] is not None and template[4] is None:
                            template[4] = shared_bl[0]
                        if shared_bl[1] is not None:
                            template[3] = shared_bl[1]

                    if self.has_any_neighbour(
                        0, column_index, row_index, EdgeDirection.BOTTOM_RIGHT
                    ):
                        shared_br = self.get_ring_joint_indexes(
                            *self.shift(
                                column_index, row_index, EdgeDirection.BOTTOM_RIGHT
                            ),
                            EdgeDirection.TOP_LEFT,
                        )
                        if shared_br[0] is not None and template[3] is None:
                            template[3] = shared_br[0]
                        if shared_br[1] is not None:
                            template[2] = shared_br[1]

                    for missing_vertex in missing_vertices:
                        template[missing_vertex] = None

                    current_new_id = self.reg.node_count
                    for i in range(6):
                        if i in missing_vertices:
                            template[i] = None
                        elif template[i] is None:
                            template[i] = current_new_id
                            current_new_id += 1

                    if self.framework[column_index, row_index] == 3:
                        self.reg.add_linker(column_index, row_index, template)
                    else:
                        self.reg.add_ring(
                            column_index,
                            row_index,
                            template,
                            framework_type=int(self.framework[column_index, row_index]),
                        )

                    for i in range(6):
                        if i in missing_vertices:
                            continue
                        if self[i + 1, column_index, row_index] != 0:
                            self.reg.add_functional_group(
                                column_index,
                                row_index,
                                i,
                                FunctionalGroup(self[i + 1, column_index, row_index]),
                            )
                        if self[i + 7, column_index, row_index] != 0:
                            self.reg.update_atom_type(
                                column_index,
                                row_index,
                                i,
                                AtomType(self[i + 7, column_index, row_index]),
                            )

    def _registry_graph(self) -> nx.Graph:
        """
        Build the molecular graph from the molecular index registry.

        Returns
        -------
        networkx.Graph
            Graph with a node for every atom that has at least 1 bond, the node attributes of the registry, and the
            registry bond types in the ``bond_type`` edge attribute. Self-loops are removed, and an atom whose only bond
            is a self-loop remains as a node without an edge.
        """
        molecular_graph = nx.Graph()
        molecular_graph.add_edges_from(self.reg.adjacency_matrix)
        nx.set_node_attributes(molecular_graph, self.reg.node_attributes)
        molecular_graph.remove_edges_from(nx.selfloop_edges(molecular_graph))
        nx.set_edge_attributes(molecular_graph, self.reg.edge_attributes, "bond_type")
        return molecular_graph

    def _defragment(self) -> None:
        """
        Keep the largest piece of the molecular graph and remove the other pieces from the grid.

        A piece is a connected component of the graph that `_registry_graph` builds from the registry. When there are 2
        or more pieces, the kept piece is the 1 with the most framework tiles. A framework tile counts toward every
        piece that has an atom at 1 of its vertices. A tie is resolved in favour of the piece closer to the central
        position ``(grid_size // 2, grid_size // 2)``, where the distance of a piece is the smallest
        `periodic_hexagonal_distance` from the central position to 1 of its framework tiles. Among pieces that also
        tie in distance, the piece registered first is kept. `_framework_tile_components` orders the components by their
        smallest position so the component that starts furthest to the left is registered first.

        At every position where the framework tile does not have an atom of the kept piece, the whole tile stack is set
        to 0. On a linker tile of the kept piece, the atom type tile and the functional group tile are set to 0 at every
        vertex whose atom is not in the kept piece. The registry is then rebuilt from the modified grid and contains the
        atoms and bonds of the kept piece only. When there is at most 1 piece, neither the grid nor the registry
        changes.

        Notes
        -----
        `MoleculeIndexRegistry.add_ring` bonds the atoms of a ring tile in a cycle so all of its atoms belong to a
        single piece. `add_linker` bonds only consecutive non-empty vertices so the atoms of a linker tile can belong to
        different pieces. A linker tile is therefore the only framework tile that can have some of its atoms
        kept and others removed.

        Every framework tile with an atom of the kept piece is kept so the cleared tile stacks and vertices do not have
        any atom of the kept piece. The rebuilt registry preserves the atom connectivity of the kept piece but assigns
        new atom indices.
        """
        # NetworkX stubs lose node type in _dispatchable; nodes are atom indices
        pieces = cast(
            list[set[int]], list(nx.connected_components(self._registry_graph()))
        )
        if len(pieces) <= 1:
            return

        central_position = (self.grid_size // 2, self.grid_size // 2)
        piece_positions = [
            {
                position
                for atom_index in piece
                for position in self.reg.index_to_ring.get(atom_index, [])
            }
            for piece in pieces
        ]
        rankings = [
            (
                -len(positions),
                min(
                    periodic_hexagonal_distance(
                        position, central_position, self.grid_size
                    )
                    for position in positions
                ),
            )
            for positions in piece_positions
        ]
        kept_index = rankings.index(min(rankings))
        kept_piece = pieces[kept_index]
        kept_positions = piece_positions[kept_index]
        for position, ring in self.reg.indexes.items():
            if position not in kept_positions:
                self.framework[position] = 0
                for vertex in range(6):
                    self.functional_groups[vertex][position] = 0
                    self.atom_types[vertex][position] = 0
                continue
            for vertex, atom_index in enumerate(ring.indexes):
                if atom_index is not None and atom_index not in kept_piece:
                    self.functional_groups[vertex][position] = 0
                    self.atom_types[vertex][position] = 0
        self._register_framework_tiles()

    @property
    def graph(self) -> nx.Graph:
        """
        Reconstruct a labelled molecular graph from the PAC grid.

        The `mol`, `smiles`, and `fingerprint` properties all depend on this reconstruction.

        When the registered atoms form 2 or more pieces, `_defragment` keeps the piece with the most framework tiles and
        removes the others from the grid before bond orders are assigned. Defragmentation is part of autocorrection.

        Graph reconstruction modifies the PAC instance. The preliminary and RDKit-derived aromaticity assignments are
        written back to the atom-type channels, the tiles of the removed pieces are set to 0, and the molecular index
        registry is rebuilt. Successive calls can therefore differ if the first call corrects an inconsistent grid.

        Returns
        -------
        networkx.Graph
            Molecular graph indexed by the atom numbers assigned during reconstruction. Nodes have element and
            electronic-structure attributes, and edges have RDKit bond types. Self-loops are removed. The graph has at
            most 1 connected component.

        Notes
        -----
        Bond orders are inferred because they are not encoded directly in the grid. Framework bonds are initially
        assigned as single bonds. Maximum weight matching then selects the double bonds among atoms that can accept
        these bonds. Atoms are excluded from the matching if they already have an external double bond or have an atom type
        incompatible with another double bond.

        The resulting graph is sanitised by RDKit. When sanitisation succeeds, RDKit aromaticity is copied to the graph
        nodes and atom-type channels, and bonds between aromatic framework atoms are assigned aromatic bond types unless
        they belong only to a linker. If sanitisation fails, the graph is returned with the bond orders obtained from
        the matching step.
        """
        self._update_aromaticity()
        self._register_framework_tiles()
        self._defragment()

        ring_atoms = set()
        for xy, ring in self.reg.indexes.items():
            if self.framework[xy] in (1, 2):
                for idx in ring.indexes:
                    if idx is not None:
                        ring_atoms.add(idx)

        # Copy atom types to every coinciding vertex representing same atom
        for node in range(self.reg.node_count):
            vertex_directions = self.reg.locate_vertex_directions(node)
            if not vertex_directions:
                continue

            is_carbonyl_node = False
            for (column_index, row_index), vertex_dir in vertex_directions.items():
                if (
                    self.functional_groups[vertex_dir.value][column_index, row_index]
                    == FunctionalGroup.CARBONYL.value
                ):
                    is_carbonyl_node = True
                    break

            if is_carbonyl_node:
                for (column_index, row_index), vertex_dir in vertex_directions.items():
                    self.atom_types[vertex_dir.value][column_index, row_index] = (
                        AtomType.CARBON_SP2.value
                    )

        # ======[ Assign Kekule Bonds by Maximum-Weight Matching ]======
        nodes_with_external_double_bonds = set()
        for edge, b_type in list(self.reg.edge_attributes.items()):
            if b_type == BondType.DOUBLE:
                u, v = edge
                if u in ring_atoms and v not in ring_atoms:
                    nodes_with_external_double_bonds.add(u)
                elif v in ring_atoms and u not in ring_atoms:
                    nodes_with_external_double_bonds.add(v)

        candidate_double_edges = []
        forbidden_double = {
            AtomType.CARBON_SP3.value,
            AtomType.NITROGEN_SP2_2PI.value,
            AtomType.NITROGEN_SP3.value,
            AtomType.OXYGEN_SP3.value,
            AtomType.SULFUR_SP3.value,
        }

        for edge in list(self.reg.adjacency_matrix):
            edge_sorted = tuple(sorted(edge))
            if edge_sorted not in self.reg.edge_to_framework:
                continue

            u, v = edge
            attr_u = self.reg.node_attributes.get(u, {})
            attr_v = self.reg.node_attributes.get(v, {})

            type_u = attr_u.get("pac_atom_type", AtomType.CARBON_SP2.value)
            type_v = attr_v.get("pac_atom_type", AtomType.CARBON_SP2.value)

            can_double_u = (u not in nodes_with_external_double_bonds) and (
                type_u not in forbidden_double
            )
            can_double_v = (v not in nodes_with_external_double_bonds) and (
                type_v not in forbidden_double
            )

            self.reg.edge_attributes[edge] = BondType.SINGLE

            if can_double_u and can_double_v:
                candidate_double_edges.append(edge)

        if candidate_double_edges:
            matching_graph = nx.Graph()
            for u, v in candidate_double_edges:
                matching_graph.add_edge(u, v, weight=10)
            matching = nx.max_weight_matching(matching_graph, maxcardinality=True)
            for u, v in matching:
                self.reg.edge_attributes[(u, v)] = BondType.DOUBLE
                if (v, u) in self.reg.edge_attributes:
                    self.reg.edge_attributes[(v, u)] = BondType.DOUBLE

        # ======[ Recompute Aromaticity with RDKit ]======
        temp_G = self._registry_graph()

        try:
            temp_mol = nx_to_mol(temp_G)
            Chem.SanitizeMol(temp_mol)

            for atom in temp_mol.GetAtoms():
                node = atom.GetIntProp("pac_node_id")
                is_aromatic = atom.GetIsAromatic()
                self.reg.node_attributes[node]["is_aromatic"] = is_aromatic

                vertex_directions = self.reg.locate_vertex_directions(node)
                if not vertex_directions:
                    continue

                is_carbonyl_node = False
                for (column_index, row_index), vertex_dir in vertex_directions.items():
                    if (
                        self.functional_groups[vertex_dir.value][
                            column_index, row_index
                        ]
                        == FunctionalGroup.CARBONYL.value
                    ):
                        is_carbonyl_node = True
                        break

                for (column_index, row_index), vertex_dir in vertex_directions.items():
                    fw = self.framework[column_index, row_index]
                    current_val = self.atom_types[vertex_dir.value][
                        column_index, row_index
                    ]

                    if is_carbonyl_node:
                        self.atom_types[vertex_dir.value][column_index, row_index] = (
                            AtomType.CARBON_SP2.value
                        )
                    elif is_aromatic:
                        if current_val == AtomType.CARBON_SP2.value:
                            if fw == 1:
                                self.atom_types[vertex_dir.value][
                                    column_index, row_index
                                ] = AtomType.CARBON_AROMATIC.value
                            elif fw == 2:
                                self.atom_types[vertex_dir.value][
                                    column_index, row_index
                                ] = AtomType.CARBON_AROMATIC_5_MEMBERED.value
                    else:
                        if current_val in (
                            AtomType.CARBON_AROMATIC.value,
                            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
                        ):
                            self.atom_types[vertex_dir.value][
                                column_index, row_index
                            ] = AtomType.CARBON_SP2.value

        except Exception:
            pass

        # ======[ Retain Aromatic Bond Types for Rendering ]======
        for edge in list(self.reg.adjacency_matrix):
            edge_sorted = tuple(sorted(edge))
            if edge_sorted not in self.reg.edge_to_framework:
                continue

            u, v = edge
            is_both_aromatic = self.reg.node_attributes.get(u, {}).get(
                "is_aromatic", False
            ) and self.reg.node_attributes.get(v, {}).get("is_aromatic", False)

            frameworks = self.reg.edge_to_framework.get(edge_sorted, set())
            is_linker_only = (
                (3 in frameworks) and (1 not in frameworks) and (2 not in frameworks)
            )

            if is_both_aromatic and not is_linker_only:
                self.reg.edge_attributes[edge] = BondType.AROMATIC
                if (v, u) in self.reg.edge_attributes:
                    self.reg.edge_attributes[(v, u)] = BondType.AROMATIC

        return self._registry_graph()

    def draw_graph(self) -> go.Figure:
        """
        Draw the reconstructed molecular graph with Plotly.

        Atom positions are generated with the Kamada-Kawai graph layout rather than the PAC lattice coordinates, and
        bond orders are not shown. Calling this method can modify the PAC grid during graph reconstruction.

        Returns
        -------
        plotly.graph_objects.Figure
            Molecular graph figure without axes or a legend.
        """
        G = self.graph
        pos = nx.kamada_kawai_layout(G)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color="#666666"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_colors = []
        node_text = []

        color_map = {
            1: "#FFFFFF",
            6: "#333333",
            7: "#00FF00",
            8: "#0000FF",
            16: "#FFD700",
        }

        for node in G.nodes():
            column_index, row_index = pos[node]
            node_x.append(column_index)
            node_y.append(row_index)
            atomic_num = G.nodes[node].get("atomic_number", 6)
            node_colors.append(color_map.get(atomic_num, "#333333"))
            node_text.append(str(node))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="bottom center",
            hoverinfo="text",
            marker=dict(showscale=False, color=node_colors, size=20, line_width=2),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
            ),
        )
        return fig

    def as_dict(self) -> dict:
        """
        Return the occupied framework positions, with each index above ``grid_size / 2`` shifted by ``-grid_size``.

        A structure that wraps around the boundary at index 0 has contiguous indices. Empty rows are omitted.

        Returns
        -------
        dict
            Mapping from the shifted row index to the sorted shifted column indices of the occupied tiles. Keys and
            values are NumPy integers.
        """
        non_zero_rows, non_zero_cols = self.framework.nonzero()
        rows_with_non_zero = np.unique(non_zero_rows)
        rows_with_non_zero = np.sort(
            np.where(
                rows_with_non_zero > (self.grid_size / 2),
                rows_with_non_zero - self.grid_size,
                rows_with_non_zero,
            )
        )
        lil = dict()
        for row in rows_with_non_zero:
            lil[row] = []
            nonzero_cols = non_zero_cols[
                non_zero_rows == (row if row >= 0 else row + self.grid_size)
            ]
            nonzero_cols = np.sort(
                np.where(
                    nonzero_cols > (self.grid_size / 2),
                    nonzero_cols - self.grid_size,
                    nonzero_cols,
                )
            )
            lil[row].extend(nonzero_cols)
        return lil

    def get_ring_joint_indexes(
        self, column_index: int, row_index: int, edge_direction: EdgeDirection
    ) -> list[int | None]:
        """
        Return the 2 molecular atom indices along 1 edge of the hexagon of a tile.

        The 2 indices are in clockwise order around the hexagon of the tile.

        Parameters
        ----------
        column_index : int
            Column coordinate of the registered tile.
        row_index : int
            Row coordinate of the registered tile.
        edge_direction : EdgeDirection
            Edge identified by the direction of the neighbouring tile. The value must be an `EdgeDirection` member.

        Returns
        -------
        list of int or None
            Atom indices at the 2 edge vertices. Either entry may be None for a 5-membered ring or linker.

        Raises
        ------
        ValueError
            Raised when the tile is empty or `edge_direction` is not a single valid edge direction.
        """
        if self.framework[column_index, row_index] not in (
            Framework.SIX_MEMBERED.value,
            2,
            3,
        ):
            raise ValueError(
                f"The tile at coordinates ({column_index}, {row_index}) has an invalid framework value: "
                f"{self.framework[column_index, row_index]}."
            )

        ring_indexes = self.reg.indexes[
            (column_index % self.grid_size, row_index % self.grid_size)
        ]

        if edge_direction == EdgeDirection.RIGHT:
            target_vertices = [VertexDirection.TOP_RIGHT, VertexDirection.BOTTOM_RIGHT]
            indices = [ring_indexes[1], ring_indexes[2]]
        elif edge_direction == EdgeDirection.TOP_LEFT:
            target_vertices = [VertexDirection.TOP_LEFT, VertexDirection.TOP]
            indices = [ring_indexes[5], ring_indexes[0]]
        elif edge_direction == EdgeDirection.TOP_RIGHT:
            target_vertices = [VertexDirection.TOP, VertexDirection.TOP_RIGHT]
            indices = [ring_indexes[0], ring_indexes[1]]
        elif edge_direction == EdgeDirection.BOTTOM_RIGHT:
            target_vertices = [VertexDirection.BOTTOM_RIGHT, VertexDirection.BOTTOM]
            indices = [ring_indexes[2], ring_indexes[3]]
        elif edge_direction == EdgeDirection.BOTTOM_LEFT:
            target_vertices = [VertexDirection.BOTTOM, VertexDirection.BOTTOM_LEFT]
            indices = [ring_indexes[3], ring_indexes[4]]
        elif edge_direction == EdgeDirection.LEFT:
            target_vertices = [VertexDirection.BOTTOM_LEFT, VertexDirection.TOP_LEFT]
            indices = [ring_indexes[4], ring_indexes[5]]
        else:
            raise ValueError(f"Invalid edge direction: {edge_direction}")

        if self.framework[column_index, row_index] == Framework.SIX_MEMBERED.value:
            return indices

        else:
            # Check at start of method leaves framework values 2 and 3
            aligned_indices = []
            for vertex_dir, idx in zip(target_vertices, indices):
                if self.atom_types[vertex_dir.value][column_index, row_index] == 0:
                    aligned_indices.append(None)
                else:
                    aligned_indices.append(idx)
            return aligned_indices

    def print_framework(self, shift_x: int = 0, shift_y: int = 0) -> None:
        """
        Print the framework channel as a numeric grid.

        Parameters
        ----------
        shift_x : int, default 0
            Circular column shift applied before printing.
        shift_y : int, default 0
            Circular row shift applied before printing.
        """
        dense_matrix = self.framework.toarray()
        dense_matrix = np.roll(dense_matrix, shift_x, axis=1)
        dense_matrix = np.roll(dense_matrix, shift_y, axis=0)

        for row in dense_matrix:
            print(" ".join(map(str, row)))

    def rotated(self) -> PAC:
        """
        Return a PAC copy rotated 60° counterclockwise.

        The copied index registry retains its original tile coordinates until reconstructed by `graph`.

        Returns
        -------
        PAC
            Deep copy in the rotated orientation.

        Raises
        ------
        ValueError
            If the transformed position of the seed ring cannot be recovered.
        """
        rotated_pac = deepcopy(self)
        magic_number = 42
        previous_value = rotated_pac.framework[self.init_pos]
        rotated_pac.framework[self.init_pos] = magic_number
        # Rotate about array center. `init_pos` changes under chained rotations.
        rotated_pac.framework = PACMatrix(rotate(rotated_pac.framework.toarray()))
        new_init_pos = np.argwhere(rotated_pac.framework.toarray() == magic_number)
        if new_init_pos.size == 0:
            raise ValueError("Cannot determine `init_pos` after rotation.")
        rotated_pac.init_pos = (
            new_init_pos[0][1],
            new_init_pos[0][0],
        )
        rotated_pac.framework[rotated_pac.init_pos] = previous_value

        for i in range(6):
            rotated_pac.atom_types[i] = PACMatrix(
                rotate(rotated_pac.atom_types[i].toarray())
            )
            rotated_pac.functional_groups[i] = PACMatrix(
                rotate(rotated_pac.functional_groups[i].toarray())
            )

        rotated_pac.functional_groups = [
            rotated_pac.functional_groups[rotated_channel_index(i)] for i in range(6)
        ]
        rotated_pac.atom_types = [
            rotated_pac.atom_types[rotated_channel_index(i)] for i in range(6)
        ]
        return rotated_pac

    def reflected(self) -> PAC:
        """
        Return a PAC copy reflected across the vertical axis.

        The copied index registry retains its original tile coordinates until reconstructed by `graph`.

        Returns
        -------
        PAC
            Deep copy in the reflected orientation.

        Raises
        ------
        ValueError
            If the transformed position of the seed ring cannot be recovered.
        """
        reflected_pac = deepcopy(self)
        magic_number = 42
        previous_value = reflected_pac.framework[self.init_pos]
        reflected_pac.framework[self.init_pos] = magic_number
        reflected_pac.framework = PACMatrix(reflect(reflected_pac.framework.toarray()))
        new_init_pos = np.argwhere(reflected_pac.framework.toarray() == magic_number)
        if new_init_pos.size == 0:
            raise ValueError("Cannot determine `init_pos` after reflection.")
        reflected_pac.init_pos = (
            new_init_pos[0][1],
            new_init_pos[0][0],
        )
        reflected_pac.framework[reflected_pac.init_pos] = previous_value

        for i in range(6):
            reflected_pac.atom_types[i] = PACMatrix(
                reflect(reflected_pac.atom_types[i].toarray())
            )
            reflected_pac.functional_groups[i] = PACMatrix(
                reflect(reflected_pac.functional_groups[i].toarray())
            )

        reflected_pac.functional_groups = [
            reflected_pac.functional_groups[reflected_channel_index(i)]
            for i in range(6)
        ]
        reflected_pac.atom_types = [
            reflected_pac.atom_types[reflected_channel_index(i)] for i in range(6)
        ]
        return reflected_pac

    def autocorrected(self) -> PAC:
        """Return a new PAC grid encoded from the decoded molecule.

        The result is the corrected grid of `autocorrect_grid`. This PAC grid is not modified.

        Returns
        -------
        PAC
            Re-encoded PAC grid.

        Raises
        ------
        ValueError
            The grid cannot decode, or the decoded molecule cannot be encoded.
        """
        decoding = autocorrect_grid(self.to_tensor())
        if decoding is None:
            raise ValueError(
                "The grid is undecodable or the decoded molecule cannot be encoded."
            )
        return PAC.from_tensor(decoding.corrected_grid)

    def shifted(self, row_shift: int, column_shift: int) -> PAC:
        """
        Return a circularly translated copy of the PAC grid.

        `init_pos` and the molecular index registry retain their previous coordinates, leaving the registry stale until
        reconstructed by `graph`.

        Parameters
        ----------
        row_shift : int
            Number of rows to shift downwards. The row shift is the first positional argument, unlike the coordinate
            order used elsewhere in `PAC`.
        column_shift : int
            Number of columns to shift to the right.

        Returns
        -------
        PAC
            Deep copy with the requested translation.
        """
        shifted_pac = deepcopy(self)
        shifted_pac.framework = PACMatrix(
            np.roll(
                shifted_pac.framework.toarray(), (row_shift, column_shift), axis=(0, 1)
            )
        )
        for i in range(6):
            shifted_pac.functional_groups[i] = PACMatrix(
                np.roll(
                    shifted_pac.functional_groups[i].toarray(),
                    (row_shift, column_shift),
                    axis=(0, 1),
                )
            )
            shifted_pac.atom_types[i] = PACMatrix(
                np.roll(
                    shifted_pac.atom_types[i].toarray(),
                    (row_shift, column_shift),
                    axis=(0, 1),
                )
            )
        return shifted_pac

    def print(self, channel: int, shift_x: int = 0, shift_y: int = 0) -> None:
        """
        Print a PAC channel as a numeric grid.

        A channel index outside 0-12 does not produce any output.

        Parameters
        ----------
        channel : int
            Channel index.
        shift_x : int, default 0
            Circular column shift applied before printing.
        shift_y : int, default 0
            Circular row shift applied before printing.
        """
        if channel == 0:
            self.print_framework(shift_x, shift_y)
        elif 1 <= channel <= 6:
            dense_matrix = self.functional_groups[channel - 1].toarray()
            dense_matrix = np.roll(dense_matrix, shift_x, axis=1)
            dense_matrix = np.roll(dense_matrix, shift_y, axis=0)
            for row in dense_matrix:
                print(" ".join(map(str, row)))
        elif 7 <= channel <= 12:
            dense_matrix = self.atom_types[channel - 7].toarray()
            dense_matrix = np.roll(dense_matrix, shift_x, axis=1)
            dense_matrix = np.roll(dense_matrix, shift_y, axis=0)
            for row in dense_matrix:
                print(" ".join(map(str, row)))

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the 13 PAC channels into a floating-point PyTorch tensor.

        Returns
        -------
        torch.Tensor
            PAC grid with shape ``(13, grid_size, grid_size)`` on the CPU.
        """
        base = self.framework.toarray().astype(np.float32)
        func_arrays = [f.toarray().astype(np.float32) for f in self.functional_groups]
        atom_arrays = [a.toarray().astype(np.float32) for a in self.atom_types]
        channels = (
            [torch.from_numpy(base)]
            + [torch.from_numpy(arr) for arr in func_arrays]
            + [torch.from_numpy(arr) for arr in atom_arrays]
        )
        return torch.stack(channels, dim=0)

    def __getitem__(self, tile_index: tuple[int, int, int]) -> Any:
        channel, column_index, row_index = tile_index
        column_index = column_index % self.grid_size
        row_index = row_index % self.grid_size
        if channel == 0:
            return self.framework[column_index, row_index]
        elif 1 <= channel <= 6:
            return self.functional_groups[channel - 1][column_index, row_index]
        else:
            return self.atom_types[channel - 7][column_index, row_index]

    def calculate_realignment_shift(
        self, positions: set[tuple[int, int]] | None = None
    ) -> tuple[int, int]:
        """
        Calculate the circular translation that centres the occupied framework.

        Parameters
        ----------
        positions : set of tuple of int or None, default None
            ``(column_index, row_index)`` positions of the occupied tiles. Every position with an occupied framework
            tile is used when this is None.

        Returns
        -------
        tuple of int
            Row shift and column shift that place the centre of the shortest periodic bounding box at the grid centre.
        """
        bounding_box = self.calculate_bounding_box(
            negative_indexing=True, positions=positions
        )
        occupied_row_center = (bounding_box.row_min + bounding_box.row_max) // 2
        occupied_column_center = (
            bounding_box.column_min + bounding_box.column_max
        ) // 2
        row_shift = (self.grid_size // 2 - occupied_row_center) % self.grid_size
        column_shift = (self.grid_size // 2 - occupied_column_center) % self.grid_size
        return row_shift, column_shift

    def realign(self) -> None:
        """
        Realign the PAC grid by centring its occupied framework.

        `init_pos` and the molecular index registry retain their previous coordinates, leaving the registry stale until
        reconstructed by `graph`.
        """
        row_shift, column_shift = self.calculate_realignment_shift()
        self.framework = PACMatrix(
            np.roll(self.framework.toarray(), (row_shift, column_shift), axis=(0, 1))
        )
        for i in range(6):
            self.functional_groups[i] = PACMatrix(
                np.roll(
                    self.functional_groups[i].toarray(),
                    (row_shift, column_shift),
                    axis=(0, 1),
                )
            )
            self.atom_types[i] = PACMatrix(
                np.roll(
                    self.atom_types[i].toarray(), (row_shift, column_shift), axis=(0, 1)
                )
            )

    def identify_coinciding_vertices(
        self, atom_type_tile_index: tuple[int, int, int]
    ) -> list[tuple[tuple[int, int], VertexDirection]]:
        """
        Return all coinciding vertices that encode the same framework atom.

        Parameters
        ----------
        atom_type_tile_index : tuple of int
            ``(atom_type_channel, column, row)`` of the atom type tile of the selected vertex. The channel must be
            between 7 and 12.

        Returns
        -------
        list of tuple
            ``((column, row), vertex_direction)`` pairs. The selected vertex is first, followed by the coinciding
            vertices of the 2 neighbouring hexagons, each included only when it encodes an atom.
        """
        parents = [
            (
                (atom_type_tile_index[1], atom_type_tile_index[2]),
                VertexDirection(atom_type_tile_index[0] - 7),
            )
        ]
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
        directions = PATHS[VertexDirection(atom_type_tile_index[0] - 7)]
        for next_edge_direction, next_vertex_direction in directions:
            next_position = self.shift(
                atom_type_tile_index[1], atom_type_tile_index[2], next_edge_direction
            )
            if self.framework[next_position] not in (
                Framework.SIX_MEMBERED.value,
                2,
                3,
            ):
                continue
            if self.framework[next_position] in (2, 3):
                if self.atom_types[next_vertex_direction.value][next_position] == 0:
                    continue
            parents.append((next_position, next_vertex_direction))
        return parents

    def __setitem__(self, tile_index: tuple[int, int, int], value: Any) -> None:
        channel, column_index, row_index = tile_index
        if channel == 0:
            if isinstance(value, tuple):
                framework_value, missing_vertex = value
            else:
                framework_value = value
                missing_vertex = VertexDirection.TOP
            self.framework[column_index, row_index] = framework_value

            if framework_value == Framework.SIX_MEMBERED.value:
                for i in range(6):
                    self.atom_types[i][column_index, row_index] = (
                        AtomType.CARBON_SP2.value
                    )
            elif framework_value == 2:
                for i in range(6):
                    if i == missing_vertex.value:
                        self.atom_types[i][column_index, row_index] = 0
                    else:
                        self.atom_types[i][column_index, row_index] = (
                            AtomType.CARBON_SP2.value
                        )
            elif framework_value == 3:
                pass

        elif 1 <= channel <= 6:
            self.functional_groups[channel - 1][column_index, row_index] = value
        else:
            self.atom_types[channel - 7][column_index, row_index] = value

            # Copy sp3 C and heteroatoms to coinciding vertices
            if value >= AtomType.CARBON_SP3.value:
                parents = self.identify_coinciding_vertices(
                    (channel, column_index, row_index)
                )
                for parent in parents:
                    px, py, vertex_direction = parent[0][0], parent[0][1], parent[1]
                    self.atom_types[vertex_direction.value][px, py] = value

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> PAC:
        """
        Reconstruct a PAC grid from a stacked channel tensor.

        Values are converted to int8 without rounding. Convert model output to categorical channel values before calling
        this method. The registry remains empty until molecular graph reconstruction.

        Parameters
        ----------
        tensor : torch.Tensor of shape (13, grid_size, grid_size)
            Channel tensor in the order returned by `to_tensor`. The tensor is detached and moved to the CPU before
            conversion.

        Returns
        -------
        PAC
            Reconstructed PAC grid. `init_pos` remains fixed at ``(7, 7)`` regardless of the tensor dimensions or
            molecular position.

        Raises
        ------
        ValueError
            Raised when the framework channel does not contain an occupied tile.
        """
        matrix_representation = tensor.detach().cpu().to(torch.int8).numpy()
        pac = PAC(grid_size=matrix_representation.shape[1])
        pac.framework = PACMatrix(matrix_representation[0])
        for i in range(6):
            pac.functional_groups[i] = PACMatrix(matrix_representation[i + 1])
            pac.atom_types[i] = PACMatrix(matrix_representation[i + 7])
        occupied_tiles = np.argwhere(pac.framework.toarray() > 0)
        if occupied_tiles.size == 0:
            raise ValueError("The tensor does not contain any occupied tiles.")
        pac.init_pos = (7, 7)
        return pac

    @property
    def smiles(self) -> str | None:
        """
        Return the canonical SMILES of the molecule.

        The string is canonical for the sanitised molecule. Calling this property can modify the PAC grid during graph
        reconstruction.

        Returns
        -------
        str or None
            Canonical SMILES string. The result is None if any conversion or sanitisation step fails.
        """
        try:
            return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(self.mol)))
        except:
            return None

    @property
    def fingerprint(self) -> torch.Tensor:
        """
        Return the Morgan count fingerprint of the molecule.

        The fingerprint is the radius-3, 2048-element Morgan count fingerprint of the sanitised molecule. Calling this
        property can modify the PAC grid during graph reconstruction.

        Returns
        -------
        torch.Tensor of shape (2048,)
            Morgan fingerprint counts stored as float32 on the CPU.
        """
        return get_fingerprint(Chem.MolFromSmiles(Chem.MolToSmiles(self.mol)))

    @property
    def mol(self) -> Chem.Mol:
        """
        Return the reconstructed molecule as an RDKit object.

        The molecule is built from `graph` without a final RDKit sanitisation step. Calling this property can modify the
        PAC grid during graph reconstruction.

        Returns
        -------
        rdkit.Chem.Mol
            Unsanitised molecule. Each atom has a `pac_node_id` integer property corresponding to its graph node.
        """
        G = self.graph
        return Chem.Mol(nx_to_mol(G))

    def serialize(self) -> str:
        """
        Serialise all 13 PAC grid channels with run-length encoding.

        Returns
        -------
        str
            Encoded channel strings separated by semicolons.
        """
        framework = _generate_str_for_serialization(
            [str(value) for value in self.framework.toarray().flatten()]
        )
        functionals = []
        for functional in self.functional_groups:
            functionals.append(
                _generate_str_for_serialization(
                    [str(value) for value in functional.toarray().flatten()]
                )
            )
        atom_types = []
        for atom_type in self.atom_types:
            atom_types.append(
                _generate_str_for_serialization(
                    [str(value) for value in atom_type.toarray().flatten()]
                )
            )
        return framework + ";" + ";".join(functionals) + ";" + ";".join(atom_types)

    @staticmethod
    def deserialize(text: str) -> PAC:
        """
        Reconstruct a PAC grid from its serialised representation.

        Parameters
        ----------
        text : str
            String returned by `serialize`.

        Returns
        -------
        PAC
            Reconstructed PAC grid.
        """
        pac = PAC()
        split_text = text.split(";")
        framework, functional_groups, atom_types = (
            split_text[0],
            split_text[1:7],
            split_text[7:13],
        )
        framework = _generate_arr_values_from_serialization(framework)
        functional_groups = [
            _generate_arr_values_from_serialization(functional).reshape(
                (pac.grid_size, pac.grid_size)
            )
            for functional in functional_groups
        ]
        atom_types = [
            _generate_arr_values_from_serialization(atom_type).reshape(
                (pac.grid_size, pac.grid_size)
            )
            for atom_type in atom_types
        ]
        pac.framework = PACMatrix(
            np.array(framework, dtype=np.int8).reshape((pac.grid_size, pac.grid_size))
        )
        pac.functional_groups = [
            PACMatrix(functional) for functional in functional_groups
        ]
        pac.atom_types = [PACMatrix(atom_type) for atom_type in atom_types]
        return pac

    @property
    def category(self) -> str:
        """
        Return the five-digit category of the molecule.

        The 5 binary digits indicate the presence of 6-membered rings, 5-membered rings, heteroatoms or sp3 carbons,
        linkers, and functional groups, in that order.

        Returns
        -------
        str
            Five binary digits describing the features present.
        """
        characteristics = self.has_category_x_characteristics
        return f"{int(characteristics[1])}{int(characteristics[2])}{int(characteristics[3])}{int(characteristics[4])}{int(characteristics[5])}"

    @property
    def has_category_x_characteristics(self) -> dict:
        """
        Return the features encoded by `category`.

        Returns
        -------
        dict
            Boolean feature flags indexed from 1 to 5:

            1. Presence of 6-membered rings. This value is always True.
            2. Presence of a 5-membered ring.
            3. Presence of an sp3 carbon or any nitrogen, oxygen, or sulfur atom in the framework.
            4. Presence of a linker.
            5. Presence of a functional group.

            Some values are NumPy booleans because they are calculated with `numpy.any`.
        """
        has_5_membered_rings = np.any(self.framework.toarray() == 2)
        has_linker = np.any(self.framework.toarray() == 3)
        has_functional_groups = any(
            np.any(func.toarray() != 0) for func in self.functional_groups
        )
        has_category_3 = any(
            np.any(atom_type.tocsr().data > AtomType.CARBON_SP2.value)
            for atom_type in self.atom_types
        )
        has_category_x_characteristics = {
            1: True,
            2: has_5_membered_rings,
            3: has_category_3,
            4: has_linker,
            5: has_functional_groups,
        }
        return has_category_x_characteristics

    @property
    def level(self) -> str:
        """
        Return `category` under the name `level`.

        Returns
        -------
        str
            Five characters, each of them '0' or '1'. See `category`.
        """
        return self.category

    @property
    def classification(self) -> str:
        """
        Return `category` under the name `classification`.

        Returns
        -------
        str
            Five characters, each of them '0' or '1'. See `category`.
        """
        return self.category

    def locate_all_neighbours_clockwise(
        self, channel: int, column_index: int, row_index: int
    ) -> tuple:
        """
        Return a channel value from each neighbouring tile.

        Parameters
        ----------
        channel : int
            Grid channel to inspect. Channel 0 returns framework occupancy.
        column_index : int
            Column coordinate of the central tile.
        row_index : int
            Row coordinate of the central tile.

        Returns
        -------
        tuple
            Six channel values in clockwise order from the upper right.
        """
        return (
            self[
                channel, *self.shift(column_index, row_index, EdgeDirection.TOP_RIGHT)
            ],
            self[channel, *self.shift(column_index, row_index, EdgeDirection.RIGHT)],
            self[
                channel,
                *self.shift(column_index, row_index, EdgeDirection.BOTTOM_RIGHT),
            ],
            self[
                channel, *self.shift(column_index, row_index, EdgeDirection.BOTTOM_LEFT)
            ],
            self[channel, *self.shift(column_index, row_index, EdgeDirection.LEFT)],
            self[channel, *self.shift(column_index, row_index, EdgeDirection.TOP_LEFT)],
        )

    def is_legal_benzene_position(self, column_index: int, row_index: int) -> bool:
        """
        Return whether a 6-membered ring can be placed on a tile.

        The target tile must be empty. For every occupied neighbouring tile, the 2 vertices of its hexagon on the
        shared edge must have compatible atom types and must not have functional groups. The excluded atom types cannot
        accept the additional framework bond introduced by ring fusion.

        Adjacency to the existing molecule is not checked here.

        Parameters
        ----------
        column_index : int
            Column coordinate of the candidate tile.
        row_index : int
            Row coordinate of the candidate tile.

        Returns
        -------
        bool
            True if a 6-membered ring can be placed on the tile.
        """
        if self.framework[column_index, row_index] != 0:
            return False
        neighbours = self.locate_all_neighbours_clockwise(0, column_index, row_index)
        count = sum(map(bool, neighbours))

        for i, neighbour in enumerate(neighbours):
            if neighbour == 0:
                continue
            DIR = {
                0: EdgeDirection.TOP_RIGHT,
                1: EdgeDirection.RIGHT,
                2: EdgeDirection.BOTTOM_RIGHT,
                3: EdgeDirection.BOTTOM_LEFT,
                4: EdgeDirection.LEFT,
                5: EdgeDirection.TOP_LEFT,
            }
            this_neighbour_position_relative_to_current = DIR[i]
            RELATIONSHIP = {
                EdgeDirection.TOP_RIGHT: (3, 4),
                EdgeDirection.RIGHT: (4, 5),
                EdgeDirection.BOTTOM_RIGHT: (5, 0),
                EdgeDirection.BOTTOM_LEFT: (0, 1),
                EdgeDirection.LEFT: (1, 2),
                EdgeDirection.TOP_LEFT: (2, 3),
            }
            channel_1, channel_2 = RELATIONSHIP[
                this_neighbour_position_relative_to_current
            ]
            neighbour_x, neighbour_y = self.shift(
                column_index, row_index, this_neighbour_position_relative_to_current
            )
            if (
                self.functional_groups[channel_1][neighbour_x, neighbour_y] != 0
                or self.functional_groups[channel_2][neighbour_x, neighbour_y] != 0
            ):
                return False
            if self.atom_types[channel_1][neighbour_x, neighbour_y] in (
                AtomType.OXYGEN_SP3,
                AtomType.SULFUR_SP3,
                AtomType.NITROGEN_SP2_1PI,
                AtomType.NITROGEN_SP2_2PI,
            ) or self.atom_types[channel_2][neighbour_x, neighbour_y] in (
                AtomType.OXYGEN_SP3,
                AtomType.SULFUR_SP3,
                AtomType.NITROGEN_SP2_1PI,
                AtomType.NITROGEN_SP2_2PI,
            ):
                return False
        return True

    def is_legal_five_membered_position(
        self, column_index: int, row_index: int
    ) -> bool:
        """
        Return whether a 5-membered ring can be placed on a tile.

        The target tile must be empty, and for every occupied neighbouring tile, the 2 vertices of its hexagon on the
        shared edge must not have a functional group. The tile must have exactly 1 neighbouring ring, or at least 1
        neighbouring linker if a neighbouring ring is not present.

        Restricting the placement to 1 neighbouring ring keeps the omitted vertex on the molecular perimeter. Unlike
        `is_legal_benzene_position`, this method does not test the atom types at the vertices on the shared edges.

        Parameters
        ----------
        column_index : int
            Column coordinate of the candidate tile.
        row_index : int
            Row coordinate of the candidate tile.

        Returns
        -------
        bool
            True if a 5-membered ring can occupy the tile. Permitted omitted vertices are determined separately by
            `FIVE_RING_CONFIGS`.
        """
        if self.framework[column_index, row_index] != 0:
            return False

        neighbors = self.locate_all_neighbours_clockwise(0, column_index, row_index)
        count_rings = sum(
            1 for n in neighbors if n in (Framework.SIX_MEMBERED.value, 2)
        )
        count_linkers = sum(1 for n in neighbors if n == Framework.LINKER.value)

        for i, neighbor in enumerate(neighbors):
            if neighbor == 0:
                continue
            this_neighbor_dir = list(EdgeDirection)[i]
            RELATIONSHIP = {
                EdgeDirection.TOP_RIGHT: (3, 4),
                EdgeDirection.RIGHT: (4, 5),
                EdgeDirection.BOTTOM_RIGHT: (5, 0),
                EdgeDirection.BOTTOM_LEFT: (0, 1),
                EdgeDirection.LEFT: (1, 2),
                EdgeDirection.TOP_LEFT: (2, 3),
            }
            ch1, ch2 = RELATIONSHIP[this_neighbor_dir]
            nx_coord, ny_coord = self.shift(column_index, row_index, this_neighbor_dir)
            if (
                self.functional_groups[ch1][nx_coord, ny_coord] != 0
                or self.functional_groups[ch2][nx_coord, ny_coord] != 0
            ):
                return False

        if count_rings > 1:
            return False
        if count_rings == 0 and count_linkers == 0:
            return False

        return True

    def is_legal_linker_position(self, column_index: int, row_index: int) -> bool:
        """
        Return whether a linker can be placed on a tile.

        The target tile must be empty and adjacent to at least 1 occupied tile, and for every occupied neighbouring
        tile, the 2 vertices of its hexagon on the shared edge must not have a functional group. Atom types and the
        framework values of neighbouring tiles are not restricted.

        Linkers are introduced during molecular graph conversion rather than by the current action space.

        Parameters
        ----------
        column_index : int
            Column coordinate of the candidate tile.
        row_index : int
            Row coordinate of the candidate tile.

        Returns
        -------
        bool
            True if the tile satisfies the linker placement rules.
        """
        if self.framework[column_index, row_index] != 0:
            return False

        neighbours = self.locate_all_neighbours_clockwise(0, column_index, row_index)
        if sum(map(bool, neighbours)) == 0:
            return False

        for i, neighbour in enumerate(neighbours):
            if neighbour == 0:
                continue
            this_neighbour_position_relative_to_current = list(EdgeDirection)[i]
            RELATIONSHIP = {
                EdgeDirection.TOP_RIGHT: (3, 4),
                EdgeDirection.RIGHT: (4, 5),
                EdgeDirection.BOTTOM_RIGHT: (5, 0),
                EdgeDirection.BOTTOM_LEFT: (0, 1),
                EdgeDirection.LEFT: (1, 2),
                EdgeDirection.TOP_LEFT: (2, 3),
            }
            channel_1, channel_2 = RELATIONSHIP[
                this_neighbour_position_relative_to_current
            ]
            neighbour_x, neighbour_y = self.shift(
                column_index, row_index, this_neighbour_position_relative_to_current
            )

            if (
                self.functional_groups[channel_1][neighbour_x, neighbour_y] != 0
                or self.functional_groups[channel_2][neighbour_x, neighbour_y] != 0
            ):
                return False

        return True

    def is_legal_functional_group_vertex(
        self, column_index: int, row_index: int, vertex_direction: VertexDirection
    ) -> tuple[bool, bool]:
        """
        Return whether a functional group can be attached at a vertex.

        The vertex must not already have a functional group, and both tiles outside its adjacent edges must be empty.
        This restricts attachment to the molecular perimeter. The framework atom must be an sp2 or aromatic carbon, sp3
        carbon, or sp3 nitrogen.

        Carbonyl addition requires 2 available valences and is therefore permitted only at an sp3 carbon or sp3
        nitrogen.

        Parameters
        ----------
        column_index : int
            Column coordinate of the hexagon that contains the vertex.
        row_index : int
            Row coordinate of the hexagon that contains the vertex.
        vertex_direction : VertexDirection
            Vertex direction to test.

        Returns
        -------
        tuple of bool
            Permission for a standard functional group followed by permission for a carbonyl group.
        """
        if self.functional_groups[vertex_direction.value][column_index, row_index] != 0:
            return False, False
        RELATIONSHIP = {
            VertexDirection.TOP: (EdgeDirection.TOP_LEFT, EdgeDirection.TOP_RIGHT),
            VertexDirection.TOP_RIGHT: (EdgeDirection.TOP_RIGHT, EdgeDirection.RIGHT),
            VertexDirection.BOTTOM_RIGHT: (
                EdgeDirection.RIGHT,
                EdgeDirection.BOTTOM_RIGHT,
            ),
            VertexDirection.BOTTOM: (
                EdgeDirection.BOTTOM_RIGHT,
                EdgeDirection.BOTTOM_LEFT,
            ),
            VertexDirection.BOTTOM_LEFT: (
                EdgeDirection.BOTTOM_LEFT,
                EdgeDirection.LEFT,
            ),
            VertexDirection.TOP_LEFT: (EdgeDirection.LEFT, EdgeDirection.TOP_LEFT),
        }
        dir1, dir2 = RELATIONSHIP[vertex_direction]
        neighbor1 = self.shift(column_index, row_index, dir1)
        neighbor2 = self.shift(column_index, row_index, dir2)
        if self.framework[*neighbor1] or self.framework[*neighbor2]:
            return False, False
        atom_type = self.atom_types[vertex_direction.value][column_index, row_index]
        if atom_type not in (
            AtomType.NITROGEN_SP3.value,
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_SP3.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
        ):
            return False, False
        if atom_type in (AtomType.NITROGEN_SP3.value, AtomType.CARBON_SP3.value):
            return True, True
        return True, False

    def is_legal_atom_type_sp3_vertex(
        self, column_index: int, row_index: int, vertex_direction: VertexDirection
    ) -> tuple[bool, bool, bool]:
        """
        Return the sp3 atom types permitted at a framework vertex.

        The current vertex must be empty or contain an sp2 or aromatic carbon. The remaining restrictions depend on its
        functional group, on whether its atom is shared with a neighbouring hexagon, and on the atom types of its two
        adjacent vertices:

        - A vertex with a functional group can only be assigned sp3 carbon.
        - Oxygen and sulfur cannot occupy a vertex of an atom shared with a neighbouring hexagon.
        - Nitrogen, oxygen, and sulfur require carbon atoms at both adjacent vertices of the same hexagon.

        Parameters
        ----------
        column_index : int
            Column coordinate of the hexagon that contains the vertex.
        row_index : int
            Row coordinate of the hexagon that contains the vertex.
        vertex_direction : VertexDirection
            Vertex direction to test.

        Returns
        -------
        tuple of bool
            Permission for sp3 carbon, sp3 nitrogen, and sp3 oxygen or sulfur, in that order. One result covers oxygen
            and sulfur because their placement rules are identical.
        """
        can_add_carbon, can_add_nitrogen, can_add_oxygen_sulfur = True, True, True
        if self.atom_types[vertex_direction.value][column_index, row_index] not in (
            0,
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
        ):
            can_add_carbon, can_add_nitrogen, can_add_oxygen_sulfur = (
                False,
                False,
                False,
            )
        if self.functional_groups[vertex_direction.value][column_index, row_index] != 0:
            can_add_nitrogen, can_add_oxygen_sulfur = False, False
        if (
            len(
                self.identify_coinciding_vertices(
                    (vertex_direction.value + 7, column_index, row_index)
                )
            )
            > 1
        ):
            can_add_oxygen_sulfur = False
        if self.atom_types[(vertex_direction.value - 1) % 6][
            column_index, row_index
        ] not in (
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
            AtomType.CARBON_SP3.value,
        ) or self.atom_types[(vertex_direction.value + 1) % 6][
            column_index, row_index
        ] not in (
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
            AtomType.CARBON_SP3.value,
        ):
            can_add_oxygen_sulfur = False
            can_add_nitrogen = False
        return can_add_carbon, can_add_nitrogen, can_add_oxygen_sulfur

    def is_legal_atom_type_sp2_vertex(
        self, column_index: int, row_index: int, vertex_direction: VertexDirection
    ) -> tuple[bool, bool]:
        """
        Return the sp2 atom types permitted at a framework vertex.

        The current vertex must be empty or contain an sp2 or aromatic carbon, and a functional group must not be
        attached.

        An sp2 nitrogen requires a carbon or nitrogen at both adjacent vertices and cannot be assigned to a vertex of an
        atom shared with a neighbouring hexagon. An sp2 carbon is permitted only at an empty vertex, such as the empty
        vertex of a 5-membered ring or an unassigned vertex of the hexagon of a linker tile.

        Parameters
        ----------
        column_index : int
            Column coordinate of the hexagon that contains the vertex.
        row_index : int
            Row coordinate of the hexagon that contains the vertex.
        vertex_direction : VertexDirection
            Vertex direction to test.

        Returns
        -------
        tuple of bool
            Permission for sp2 nitrogen followed by permission for sp2 carbon.
        """
        can_add_carbon = True
        can_add_nitrogen = True
        if self.atom_types[vertex_direction.value][column_index, row_index] not in (
            0,
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
        ):
            return False, False
        if self.functional_groups[vertex_direction.value][column_index, row_index] != 0:
            return False, False
        if (
            len(
                self.identify_coinciding_vertices(
                    (vertex_direction.value + 7, column_index, row_index)
                )
            )
            > 1
        ):
            can_add_nitrogen = False
        if self.atom_types[(vertex_direction.value - 1) % 6][
            column_index, row_index
        ] not in (
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
            AtomType.CARBON_SP3.value,
            AtomType.NITROGEN_SP2_1PI.value,
            AtomType.NITROGEN_SP2_2PI.value,
            AtomType.NITROGEN_SP3.value,
        ) or self.atom_types[(vertex_direction.value + 1) % 6][
            column_index, row_index
        ] not in (
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
            AtomType.CARBON_SP3.value,
            AtomType.NITROGEN_SP2_1PI.value,
            AtomType.NITROGEN_SP2_2PI.value,
            AtomType.NITROGEN_SP3.value,
        ):
            can_add_nitrogen = False
        if self.atom_types[vertex_direction.value][column_index, row_index] in (
            AtomType.CARBON_SP2.value,
            AtomType.CARBON_AROMATIC.value,
            AtomType.CARBON_AROMATIC_5_MEMBERED.value,
        ):
            can_add_carbon = False
        return can_add_nitrogen, can_add_carbon

    def copy(self) -> PAC:
        """
        Return an independent copy of the grid.

        All channels and the molecular index registry are copied.

        Returns
        -------
        PAC
            Independent copy of the PAC grid.
        """
        return deepcopy(self)

    @property
    def count_6_membered_rings(self) -> int:
        """
        Count the tiles containing a six-membered ring.

        Returns
        -------
        int
            Number of tiles with framework value 1.
        """
        return int((self.framework.tocsr().data == 1).sum())

    @property
    def count_5_membered_rings(self) -> int:
        """
        Count the tiles containing a five-membered ring.

        Returns
        -------
        int
            Number of tiles with framework value 2.
        """
        return int((self.framework.tocsr().data == 2).sum())

    @property
    def count_total_rings(self) -> int:
        """
        Count the rings of both sizes together.

        Returns
        -------
        int
            Number of tiles with framework value 1 or 2.
        """
        return self.count_6_membered_rings + self.count_5_membered_rings

    @property
    def count_functional_groups(self) -> dict[FunctionalGroup, int]:
        """
        Count the functional groups of each type.

        Returns
        -------
        dict
            Count of each functional group in enum declaration order. Groups that are absent have a count of zero.
        """
        all_vals = np.concatenate([f.tocsr().data for f in self.functional_groups])
        unique_vals, val_counts = np.unique(all_vals, return_counts=True)
        counts = {fg: 0 for fg in FunctionalGroup}
        for val, count in zip(unique_vals, val_counts):
            counts[FunctionalGroup(val)] = int(count)
        return counts

    @property
    def sum_functional_groups(self) -> int:
        """
        Count all functional groups together.

        Returns
        -------
        int
            Total number of non-zero entries across all functional-group channels.
        """
        return int(
            sum(
                (functional.tocsr().data != 0).sum()
                for functional in self.functional_groups
            )
        )

    def _count_atom_type(self, atom_type_value: int) -> int:
        """
        Count the atoms of an atom type without double-counting shared atoms.

        Parameters
        ----------
        atom_type_value : int
            `AtomType` integer value to count.

        Returns
        -------
        int
            Number of atoms of that type in the molecule.
        """
        count = 0
        belongs_to = list()
        for i, atom_type in enumerate(self.atom_types):
            pos_indices = np.argwhere(atom_type.toarray() == atom_type_value)
            pos_indices = [
                (column_index, row_index) for row_index, column_index in pos_indices
            ]
            for column_index, row_index in pos_indices:
                coinciding_vertices = tuple(
                    sorted(
                        self.identify_coinciding_vertices(
                            (i + 7, column_index, row_index)
                        )
                    )
                )
                if len(coinciding_vertices) > 1:
                    if coinciding_vertices not in belongs_to:
                        belongs_to.append(coinciding_vertices)
                        count += 1
                else:
                    count += 1
        return count

    @property
    def count_sp3_carbon(self) -> int:
        """
        Count the sp3 carbon atoms of the ring system.

        Returns
        -------
        int
            Number of framework atoms encoded as `AtomType.CARBON_SP3`, with shared atoms counted once.
        """
        return self._count_atom_type(AtomType.CARBON_SP3.value)

    @property
    def count_nitrogen(self) -> int:
        """
        Count the nitrogen atoms in the framework.

        The count includes both sp2 nitrogen types and sp3 nitrogen.

        Returns
        -------
        int
            Number of nitrogen atoms in the framework. Nitrogens within functional groups are excluded.
        """
        return (
            self._count_atom_type(AtomType.NITROGEN_SP2_1PI.value)
            + self._count_atom_type(AtomType.NITROGEN_SP2_2PI.value)
            + self._count_atom_type(AtomType.NITROGEN_SP3.value)
        )

    @property
    def count_oxygen(self) -> int:
        """
        Count the sp3 oxygen atoms in the framework.

        Returns
        -------
        int
            Number of framework atoms encoded as `AtomType.OXYGEN_SP3`, with shared atoms counted once.
        """
        return self._count_atom_type(AtomType.OXYGEN_SP3.value)

    @property
    def count_sulfur(self) -> int:
        """
        Count the sp3 sulfur atoms in the framework.

        Returns
        -------
        int
            Number of framework atoms encoded as `AtomType.SULFUR_SP3`, with shared atoms counted once.
        """
        return self._count_atom_type(AtomType.SULFUR_SP3.value)

    def get_possible_actions(self, *args: Any, **kwargs: Any) -> list[Action]:
        """
        Return all structural actions permitted by the current grid.

        Parameters
        ----------
        *args
            Positional arguments forwarded to `pac.action.get_possible_actions`.
        **kwargs
            Keyword arguments forwarded to `pac.action.get_possible_actions`.

        Returns
        -------
        list of Action
            Permitted ring, functional-group, and atom-type actions.

        See Also
        --------
        pac.action.get_possible_actions : Generate legal structural actions.
        """
        return get_possible_actions(self, *args, **kwargs)

    def take_action(self, action: Action) -> None:
        """
        Apply a structural action to the grid in place.

        The action is not checked against the output of `get_possible_actions`.

        Parameters
        ----------
        action : Action
            Structural action to apply.

        See Also
        --------
        pac.action.take_action : Apply an action to a PAC grid.
        """
        take_action(self, action)

    def take_actions(self, actions: list[Action]) -> None:
        """
        Apply a sequence of actions to the grid in order.

        The legality of the actions is not checked, and previous changes are not reverted if a later action fails.

        Parameters
        ----------
        actions : list of Action
            Structural actions in application order.
        """
        for action in actions:
            self.take_action(action)

    def display_channels(
        self, shift_x: int = 0, shift_y: int = 0, crop_window: int | None = None
    ) -> None:
        """
        Display all 13 PAC channels as tables in a notebook.

        If IPython is unavailable, the method prints an explanatory message and returns.

        Parameters
        ----------
        shift_x : int, default 0
            Circular column shift applied before display.
        shift_y : int, default 0
            Circular row shift applied before display.
        crop_window : int or None, default None
            Side length of a square window centred on the shifted grid. The full grid is displayed when this is None.
        """
        try:
            from IPython.display import HTML, display
        except ImportError:
            print("IPython is not available.")
            return

        def mat_to_html(mat, channel_type):
            rolled = np.roll(mat, shift_x, axis=1)
            rolled = np.roll(rolled, shift_y, axis=0)

            if crop_window is not None:
                cx, cy = rolled.shape[0] // 2, rolled.shape[1] // 2
                half = crop_window // 2
                rolled = rolled[
                    cx - half : cx + half + (crop_window % 2),
                    cy - half : cy + half + (crop_window % 2),
                ]

            html = '<table style="border-collapse: collapse; margin: 0 auto; padding: 0; line-height: 1;">'
            for row in rolled:
                html += "<tr>"
                for val in row:
                    if val == 0:
                        bg_color = "transparent"
                        color = "#e8e8e8"
                    else:
                        color = "#000000"
                        if channel_type == "fw":
                            bg_color = "#d9edf7"
                        elif channel_type == "func":
                            bg_color = "#dff0d8"
                        else:
                            bg_color = "#fcf8e3"

                    html += f'<td style="border: 1px solid #ccc; width: 18px; height: 18px; text-align: center; vertical-align: middle; font-size: 11px; color: {color}; background-color: {bg_color};">{val}</td>'
                html += "</tr>"
            html += "</table>"
            return html

        fw_html = mat_to_html(self.framework.toarray(), "fw")

        def build_hex_grid(channels, c_type):
            tables = [mat_to_html(ch.toarray(), c_type) for ch in channels]
            areas = ["top", "tr", "br", "bottom", "bl", "tl"]

            grid_html = """
            <div style="display: grid; grid-template-areas:
                '. top .'
                'tl . tr'
                'bl . br'
                '. bottom .'; gap: 6px; align-items: center; justify-items: center;">
            """
            for area, table in zip(areas, tables):
                grid_html += f'<div style="grid-area: {area};">{table}</div>'
            grid_html += "</div>"
            return grid_html

        func_html = build_hex_grid(self.functional_groups, "func")
        atom_html = build_hex_grid(self.atom_types, "atom")

        master_html = f"""
        <div style="display: flex; flex-direction: row; gap: 40px; align-items: center; justify-content: flex-start; font-family: sans-serif;">
            <div style="display: flex; flex-direction: column; align-items: center;">
                <h4 style="margin-bottom: 15px; color: #fff;">Framework</h4>
                {fw_html}
            </div>
            <div style="display: flex; flex-direction: column; align-items: center;">
                <h4 style="margin-bottom: 15px; color: #fff;">Functional channels</h4>
                {func_html}
            </div>
            <div style="display: flex; flex-direction: column; align-items: center;">
                <h4 style="margin-bottom: 15px; color: #fff;">Atom type channels</h4>
                {atom_html}
            </div>
        </div>
        """
        display(HTML(master_html))


# `pac.convert` imports `PAC`
from pac.convert import autocorrect_grid
