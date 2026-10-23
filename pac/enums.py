"""PAC channel values and clockwise lattice directions.

`TRANSFORMATIONS` maps edge directions to axial coordinate offsets.
"""

from enum import Flag, IntEnum, auto


class Framework(IntEnum):
    """
    Define the framework values encoded in channel 0.

    Zero is an empty tile. Values 1, 2, and 3 are a 6-membered ring, a 5-membered ring, and a linker, respectively.

    Attributes
    ----------
    SIX_MEMBERED : int
        Six-membered ring occupying all 6 vertices of its hexagon.
    FIVE_MEMBERED : int
        Five-membered ring occupying 5 vertices of its hexagon. The remaining vertex has atom type 0.
    LINKER : int
        Part of an open chain connecting rings that are not fused.
    """

    SIX_MEMBERED = 1
    FIVE_MEMBERED = 2
    LINKER = 3


class FunctionalGroup(IntEnum):
    """
    Define the functional groups encoded in channels 1-6.

    Values 4-14 are the supported functional groups. The channel for a vertex is ``1 + VertexDirection.value``. Zero
    indicates that a functional group is not attached at that vertex.

    Attributes
    ----------
    NITRO : int
        Nitro group (-NO2).
    CYANIDE : int
        Cyano group (-CN).
    DIMETHYLAMINO : int
        Dimethylamino group (-N(CH3)2).
    METHYL : int
        Methyl group (-CH3).
    ETHYL : int
        Ethyl group (-CH2CH3).
    METHOXY : int
        Methoxy group (-OCH3).
    ETHOXY : int
        Ethoxy group (-OCH2CH3).
    FLUORIDE : int
        Fluorine (-F).
    CHLORIDE : int
        Chlorine (-Cl).
    CARBONYL : int
        Carbonyl oxygen (=O).
    HYDROGEN : int
        Explicit hydrogen (-H).
    """

    NITRO = 4
    CYANIDE = 5
    DIMETHYLAMINO = 6
    METHYL = 7
    ETHYL = 8
    METHOXY = 9
    ETHOXY = 10
    FLUORIDE = 11
    CHLORIDE = 12
    CARBONYL = 13
    HYDROGEN = 14


class AtomType(IntEnum):
    """
    Define the element and hybridisation values encoded in channels 7-12.

    Values 15-23 are the supported framework atom types. The channel for a vertex is ``7 + VertexDirection.value``. Zero
    indicates an empty vertex.

    Attributes
    ----------
    CARBON_AROMATIC : int
        Aromatic sp2 carbon in a six-membered ring.
    CARBON_AROMATIC_5_MEMBERED : int
        Aromatic sp2 carbon in a five-membered ring.
    CARBON_SP2 : int
        Sp2 carbon that is not aromatic.
    CARBON_SP3 : int
        Sp3 carbon.
    NITROGEN_SP2_1PI : int
        Pyridine-like sp2 nitrogen. It contributes 1 electron to the pi system.
    NITROGEN_SP2_2PI : int
        Pyrrole-like sp2 nitrogen or an amine conjugated to a ring. It contributes 2 electrons to the pi system.
    NITROGEN_SP3 : int
        Sp3 nitrogen.
    OXYGEN_SP3 : int
        Sp3 oxygen.
    SULFUR_SP3 : int
        Sp3 sulfur.
    """

    CARBON_AROMATIC = 15
    CARBON_AROMATIC_5_MEMBERED = 16
    CARBON_SP2 = 17
    CARBON_SP3 = 18
    NITROGEN_SP2_1PI = 19
    NITROGEN_SP2_2PI = 20
    NITROGEN_SP3 = 21
    OXYGEN_SP3 = 22
    SULFUR_SP3 = 23


class EdgeDirection(Flag):
    """
    Index the six neighbouring tiles with bit flags.

    The order is clockwise and starts at `TOP_RIGHT`, which is at 60 degrees.

    Attributes
    ----------
    TOP_RIGHT : Flag
        Neighbour tile at 60 degrees.
    RIGHT : Flag
        Neighbour tile at 0 degrees.
    BOTTOM_RIGHT : Flag
        Neighbour tile at 300 degrees.
    BOTTOM_LEFT : Flag
        Neighbour tile at 240 degrees.
    LEFT : Flag
        Neighbour tile at 180 degrees.
    TOP_LEFT : Flag
        Neighbour tile at 120 degrees.
    """

    TOP_RIGHT = auto()
    RIGHT = auto()
    BOTTOM_RIGHT = auto()
    BOTTOM_LEFT = auto()
    LEFT = auto()
    TOP_LEFT = auto()

    @staticmethod
    def all() -> tuple:
        """
        Return the six edge directions in clockwise order.

        Returns
        -------
        tuple of EdgeDirection
            The members in the order they are declared, starting at `TOP_RIGHT`.
        """
        return (
            EdgeDirection.TOP_RIGHT,
            EdgeDirection.RIGHT,
            EdgeDirection.BOTTOM_RIGHT,
            EdgeDirection.BOTTOM_LEFT,
            EdgeDirection.LEFT,
            EdgeDirection.TOP_LEFT,
        )


class VertexDirection(Flag):
    """
    Index the six vertices of a hexagon from 0 to 5.

    The order is clockwise and starts at `TOP`, which is at 90 degrees.

    Attributes
    ----------
    TOP : Flag
        Vertex at 90 degrees.
    TOP_RIGHT : Flag
        Vertex at 30 degrees.
    BOTTOM_RIGHT : Flag
        Vertex at 330 degrees.
    BOTTOM : Flag
        Vertex at 270 degrees.
    BOTTOM_LEFT : Flag
        Vertex at 210 degrees.
    TOP_LEFT : Flag
        Vertex at 150 degrees.
    """

    TOP = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM = 3
    BOTTOM_LEFT = 4
    TOP_LEFT = 5

    @staticmethod
    def all() -> tuple:
        """
        Return the six vertex directions in clockwise order.

        Returns
        -------
        tuple of VertexDirection
            The members in the order they are declared, starting at `TOP`.
        """
        return (
            VertexDirection.TOP,
            VertexDirection.TOP_RIGHT,
            VertexDirection.BOTTOM_RIGHT,
            VertexDirection.BOTTOM,
            VertexDirection.BOTTOM_LEFT,
            VertexDirection.TOP_LEFT,
        )


vertex_direction_pair_to_edge_direction = {
    (VertexDirection.TOP, VertexDirection.TOP_RIGHT): EdgeDirection.TOP_RIGHT,
    (VertexDirection.TOP_RIGHT, VertexDirection.BOTTOM_RIGHT): EdgeDirection.RIGHT,
    (VertexDirection.BOTTOM_RIGHT, VertexDirection.BOTTOM): EdgeDirection.BOTTOM_RIGHT,
    (VertexDirection.BOTTOM, VertexDirection.BOTTOM_LEFT): EdgeDirection.BOTTOM_LEFT,
    (VertexDirection.BOTTOM_LEFT, VertexDirection.TOP_LEFT): EdgeDirection.LEFT,
    (VertexDirection.TOP_LEFT, VertexDirection.TOP): EdgeDirection.TOP_LEFT,
}
"""
Mapping from each clockwise pair of consecutive vertex directions to the `EdgeDirection` of the edge between the 2
vertices.
"""

TRANSFORMATIONS = {
    EdgeDirection.TOP_RIGHT: (0, -1),
    EdgeDirection.RIGHT: (1, 0),
    EdgeDirection.BOTTOM_RIGHT: (1, 1),
    EdgeDirection.BOTTOM_LEFT: (0, 1),
    EdgeDirection.LEFT: (-1, 0),
    EdgeDirection.TOP_LEFT: (-1, -1),
}
"""
Axial coordinate offsets `(column_step, row_step)` for stepping between neighbouring tiles.
"""
