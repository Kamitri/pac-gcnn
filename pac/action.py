"""
Discrete operations for constructing a PAC grid.

WARNING: This file is not to be used in this repository. These actions are what I was going to implement for my next
paper, not this one.

Each action specifies a grid position, target channel, and structural change. The available changes include adding
rings, attaching functional groups, and assigning atom types. `get_possible_actions` lists the changes that are legal
for a given grid, and `take_action` applies 1 of them in place.
"""

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch

import const
from pac.enums import EdgeDirection, VertexDirection

if TYPE_CHECKING:
    from .pac import PAC


class ActionType(Enum):
    """
    Define the structural modifications used by the assembly action space.
    """

    ADD_BENZENE = "[1]"
    ADD_NITRO = "[2]"
    ADD_CYANIDE = "[3]"
    ADD_DIMETHYLAMINO = "[4]"
    ADD_METHYL = "[5]"
    ADD_ETHYL = "[6]"
    ADD_METHOXY = "[7]"
    ADD_ETHOXY = "[8]"
    ADD_CARBONYL = "[9]"
    # ADD_HYDROGEN = "[10]"
    ADD_CARBON_SP2 = "[11]"
    ADD_CARBON_SP3 = "[12]"
    ADD_NITROGEN_SP2 = "[13]"
    ADD_NITROGEN_SP3 = "[14]"
    ADD_OXYGEN_SP3 = "[15]"
    ADD_SULFUR_SP3 = "[16]"

    # 5-membered ring configurations (mapped to specific missing vertices)
    ADD_FIVE_MEMBERED_TOP = "[17]"
    ADD_FIVE_MEMBERED_TOP_RIGHT = "[18]"
    ADD_FIVE_MEMBERED_BOTTOM_RIGHT = "[19]"
    ADD_FIVE_MEMBERED_BOTTOM = "[20]"
    ADD_FIVE_MEMBERED_BOTTOM_LEFT = "[21]"
    ADD_FIVE_MEMBERED_TOP_LEFT = "[22]"

    def is_add_action(self) -> bool:
        """
        Return whether the action adds a structural feature to the grid.

        Returns
        -------
        bool
            Always True because every current enum member adds a ring, functional group, or atom type.
        """
        return True

    def is_add_benzene(self) -> bool:
        """
        Return whether the action adds a 6-membered ring.

        Returns
        -------
        bool
            True only for `ADD_BENZENE`.
        """
        return self == ActionType.ADD_BENZENE

    def is_add_five_membered(self) -> bool:
        """
        Return whether the action adds a 5-membered ring.

        Returns
        -------
        bool
            True for the 6 `ADD_FIVE_MEMBERED_*` actions.
        """
        return self in {
            ActionType.ADD_FIVE_MEMBERED_TOP,
            ActionType.ADD_FIVE_MEMBERED_TOP_RIGHT,
            ActionType.ADD_FIVE_MEMBERED_BOTTOM_RIGHT,
            ActionType.ADD_FIVE_MEMBERED_BOTTOM,
            ActionType.ADD_FIVE_MEMBERED_BOTTOM_LEFT,
            ActionType.ADD_FIVE_MEMBERED_TOP_LEFT,
        }

    def get_missing_vertex(self) -> VertexDirection | None:
        """
        Return the empty vertex created by a 5-membered ring action.

        Returns
        -------
        VertexDirection or None
            Empty vertex for a 5-membered ring action. The result is None for all other action types.
        """
        mapping = {
            ActionType.ADD_FIVE_MEMBERED_TOP: VertexDirection.TOP,
            ActionType.ADD_FIVE_MEMBERED_TOP_RIGHT: VertexDirection.TOP_RIGHT,
            ActionType.ADD_FIVE_MEMBERED_BOTTOM_RIGHT: VertexDirection.BOTTOM_RIGHT,
            ActionType.ADD_FIVE_MEMBERED_BOTTOM: VertexDirection.BOTTOM,
            ActionType.ADD_FIVE_MEMBERED_BOTTOM_LEFT: VertexDirection.BOTTOM_LEFT,
            ActionType.ADD_FIVE_MEMBERED_TOP_LEFT: VertexDirection.TOP_LEFT,
        }
        return mapping.get(self, None)

    def is_add_functional_group(self) -> bool:
        """
        Return whether the action adds a functional group or assigns an atom type.

        Returns
        -------
        bool
            True for the seven non-carbonyl functional groups and six atom-type assignments.
        """
        return self.value in {
            ActionType.ADD_NITRO.value,
            ActionType.ADD_CYANIDE.value,
            ActionType.ADD_DIMETHYLAMINO.value,
            # ActionType.ADD_CARBONYL.value,
            ActionType.ADD_METHYL.value,
            ActionType.ADD_ETHYL.value,
            ActionType.ADD_METHOXY.value,
            ActionType.ADD_ETHOXY.value,
            ActionType.ADD_CARBON_SP2.value,
            ActionType.ADD_CARBON_SP3.value,
            ActionType.ADD_NITROGEN_SP2.value,
            ActionType.ADD_NITROGEN_SP3.value,
            ActionType.ADD_OXYGEN_SP3.value,
            ActionType.ADD_SULFUR_SP3.value,
        }

    @staticmethod
    def add_functional_groups_no_carbonyl() -> tuple[ActionType, ...]:
        """
        Return the functional group actions other than carbonyl addition.

        These actions are legal at any vertex that accepts a functional group. Carbonyl addition is handled separately
        because it requires an sp3 carbon or sp3 nitrogen at the target vertex.

        Returns
        -------
        tuple of ActionType
            Nitro, cyano, dimethylamino, methyl, ethyl, methoxy, and ethoxy actions, in that order.
        """
        return (
            ActionType.ADD_NITRO,
            ActionType.ADD_CYANIDE,
            ActionType.ADD_DIMETHYLAMINO,
            # ActionType.ADD_CARBONYL,
            ActionType.ADD_METHYL,
            ActionType.ADD_ETHYL,
            ActionType.ADD_METHOXY,
            ActionType.ADD_ETHOXY,
        )


# Maps EdgeDirection, seen from existing ring looking at empty tile, to 2 missing vertices that orient
# 5-membered ring correctly
FIVE_RING_CONFIGS = {
    EdgeDirection.RIGHT: (VertexDirection.TOP, VertexDirection.BOTTOM),
    EdgeDirection.TOP_RIGHT: (VertexDirection.TOP_LEFT, VertexDirection.BOTTOM_RIGHT),
    EdgeDirection.TOP_LEFT: (VertexDirection.BOTTOM_LEFT, VertexDirection.TOP_RIGHT),
    EdgeDirection.LEFT: (VertexDirection.BOTTOM, VertexDirection.TOP),
    EdgeDirection.BOTTOM_LEFT: (VertexDirection.BOTTOM_RIGHT, VertexDirection.TOP_LEFT),
    EdgeDirection.BOTTOM_RIGHT: (
        VertexDirection.TOP_RIGHT,
        VertexDirection.BOTTOM_LEFT,
    ),
}

ACTION_TYPE_FROM_VERTEX_DIR = {
    VertexDirection.TOP: ActionType.ADD_FIVE_MEMBERED_TOP,
    VertexDirection.TOP_RIGHT: ActionType.ADD_FIVE_MEMBERED_TOP_RIGHT,
    VertexDirection.BOTTOM_RIGHT: ActionType.ADD_FIVE_MEMBERED_BOTTOM_RIGHT,
    VertexDirection.BOTTOM: ActionType.ADD_FIVE_MEMBERED_BOTTOM,
    VertexDirection.BOTTOM_LEFT: ActionType.ADD_FIVE_MEMBERED_BOTTOM_LEFT,
    VertexDirection.TOP_LEFT: ActionType.ADD_FIVE_MEMBERED_TOP_LEFT,
}


class Action:
    """
    Specify a structural modification at an axial grid position.

    An action can be serialised as a tuple, a string, or a tensor for the planned reinforcement-learning workflow.

    Parameters
    ----------
    x : int
        Tile column index.
    y : int
        Tile row index.
    action_type : ActionType
        Structural modification to apply.
    channel : int, default 0
        Target grid channel.

    Attributes
    ----------
    x : int
        Tile column index.
    y : int
        Tile row index.
    type : ActionType
        Structural modification represented by this action.
    action_type : ActionType
        Alias of `type`.
    channel : int
        Grid channel modified by this action.

    Raises
    ------
    ValueError
        Raised when `action_type` is not an `ActionType` member.
    """

    def __init__(
        self, x: int, y: int, action_type: ActionType, channel: int = 0
    ) -> None:
        self.x = x
        self.y = y
        self.type = action_type
        self.action_type = action_type
        self.channel = channel
        if not isinstance(self.type, ActionType):
            raise ValueError(f"Invalid action type: {self.type}")

    def __str__(self) -> str:
        return f"Action({self.x}, {self.y}, {self.channel}, {self.type})"

    def __repr__(self) -> str:
        return str(self)

    def is_add_action(self) -> bool:
        """
        Return whether the action adds a structural feature to the grid.

        Returns
        -------
        bool
            Always True because every current enum member adds a ring, functional group, or atom type.
        """
        return self.type.is_add_action()

    def is_add_benzene(self) -> bool:
        """
        Return whether the action adds a 6-membered ring.

        Returns
        -------
        bool
            True only for `ADD_BENZENE`.
        """
        return self.type.is_add_benzene()

    @staticmethod
    def from_tuple(t: tuple[int, int, int]) -> Action:
        """
        Reconstruct an action from its packed tuple representation.

        Parameters
        ----------
        t : tuple of int
            Tuple returned by `to_tuple`.

        Returns
        -------
        Action
            Reconstructed action.

        Raises
        ------
        ValueError
            Raised when the action index does not correspond to an `ActionType` member.
        """
        position, channel, value = t
        x = position // const.GRID_SIZE
        y = position % const.GRID_SIZE
        action_type = ActionType(f"[{value + 1}]")
        return Action(x, y, action_type, channel)

    @staticmethod
    def from_tuples(tuples: list[tuple[int, int, int]]) -> list[Action]:
        """
        Reconstruct several actions from packed tuple representations.

        Parameters
        ----------
        tuples : list of tuple of int
            Tuples returned by `to_tuple`.

        Returns
        -------
        list of Action
            Reconstructed actions in input order.

        Raises
        ------
        ValueError
            Raised when an action index does not correspond to an `ActionType` member.
        """
        actions = []
        for t in tuples:
            position, channel, value = t
            x = position // const.GRID_SIZE
            y = position % const.GRID_SIZE
            action_type = ActionType(f"[{value + 1}]")
            actions.append(Action(x, y, action_type, channel))
        return actions

    @staticmethod
    def from_tensor(t: torch.Tensor) -> Action:
        """
        Reconstruct an action from a packed tensor representation.

        Parameters
        ----------
        t : torch.Tensor of shape (3,)
            Tensor returned by `to_tensor`. Any integer data type and device are accepted.

        Returns
        -------
        Action
            Reconstructed action.

        Raises
        ------
        ValueError
            Raised when the action index does not correspond to an `ActionType` member.
        """
        position, channel, value = t.tolist()
        x = position // const.GRID_SIZE
        y = position % const.GRID_SIZE
        action_type = ActionType(f"[{value + 1}]")
        return Action(x, y, action_type, channel)

    @staticmethod
    def from_tensors(tensors: torch.Tensor) -> list[Action]:
        """
        Reconstruct several actions from a batch of packed tensors.

        Parameters
        ----------
        tensors : torch.Tensor of shape (num_actions, 3)
            Rows returned by `to_tensor`, 1 row per action.

        Returns
        -------
        list of Action
            Reconstructed actions in row order.

        Raises
        ------
        ValueError
            Raised when an action index does not correspond to an `ActionType` member.
        """
        actions = []
        for t in tensors:
            position, channel, value = t.tolist()
            x = position // const.GRID_SIZE
            y = position % const.GRID_SIZE
            action_type = ActionType(f"[{value + 1}]")
            actions.append(Action(x, y, action_type, channel))
        return actions

    def to_tuple(self) -> tuple[int, int, int]:
        """
        Convert the action to a tuple of three integers.

        The position is flattened with `const.GRID_SIZE`, independently of the grid size of the PAC from which the
        action was generated.

        Returns
        -------
        tuple of int
            Flattened position ``x * const.GRID_SIZE + y``, channel, and zero-based action index.
        """
        return (
            int(self.x * const.GRID_SIZE + self.y),
            self.channel,
            int(self.type.value.strip("[]")) - 1,
        )

    def to_str(self) -> str:
        """
        Convert the action to a semicolon-separated string.

        Returns
        -------
        str
            Values returned by `to_tuple`, separated by semicolons. For example, benzene addition at column 7 and row 7
            is encoded as ``112;0;0`` on the default grid.
        """
        return ";".join(map(str, self.to_tuple()))

    @staticmethod
    def from_str(s: str) -> Action:
        """
        Reconstruct an action from a semicolon-separated string.

        Parameters
        ----------
        s : str
            String returned by `to_str`. Additional fields are ignored.

        Returns
        -------
        Action
            Reconstructed action.

        Raises
        ------
        ValueError
            Raised when a field is not an integer or the action index does not correspond to an `ActionType` member.
        """
        parts = s.split(";")
        position = int(parts[0])
        channel = int(parts[1])
        value = int(parts[2])
        x = position // const.GRID_SIZE
        y = position % const.GRID_SIZE
        action_type = ActionType(f"[{value + 1}]")
        return Action(x, y, action_type, channel)

    def to_tensor(self) -> torch.Tensor:
        """
        Convert the action to a three-element integer tensor.

        Returns
        -------
        torch.Tensor of shape (3,)
            The output of `to_tuple` as an int32 CPU tensor.
        """
        return torch.tensor(
            [
                self.x * const.GRID_SIZE + self.y,
                self.channel,
                int(self.type.value.strip("[]")) - 1,
            ],
            dtype=torch.int32,
        )

    @property
    def fill_value(self) -> int:
        """
        Return the grid value associated with the action type.

        For `ADD_BENZENE`, the returned value is `Framework.SIX_MEMBERED`.

        Returns
        -------
        int
            One-based action value encoded by the enum member.
        """
        return int(self.type.value.strip("[]"))


class Filter:
    """
    Select the types of structural actions that may be generated.

    Parameters
    ----------
    add_benzene : bool, default True
        Include 6-membered ring additions when True.
    add_five_membered : bool, default True
        Include 5-membered ring additions when True.
    add_functional_groups : bool, default True
        Include functional group additions when True.
    add_carbon_sp3 : bool, default True
        Include sp3 carbon assignments when True.
    add_nitrogen : bool, default True
        Include sp2 and sp3 nitrogen assignments when True.
    add_oxygen_sp3 : bool, default True
        Include sp3 oxygen assignments when True.
    add_sulfur_sp3 : bool, default True
        Include sp3 sulfur assignments when True.
    """

    def __init__(
        self,
        add_benzene: bool = True,
        add_five_membered: bool = True,
        add_functional_groups: bool = True,
        add_carbon_sp3: bool = True,
        add_nitrogen: bool = True,
        add_oxygen_sp3: bool = True,
        add_sulfur_sp3: bool = True,
    ) -> None:
        self.add_benzene = add_benzene
        self.add_five_membered = add_five_membered
        self.add_functional_groups = add_functional_groups
        self.add_carbon_sp3 = add_carbon_sp3
        self.add_nitrogen = add_nitrogen
        self.add_oxygen_sp3 = add_oxygen_sp3
        self.add_sulfur_sp3 = add_sulfur_sp3


def get_possible_actions(pac: PAC, filter: Filter | None = None) -> list[Action]:
    """
    Return the structural actions that are legal for a PAC grid.

    Candidate ring, functional-group, and atom-type additions are filtered by current boundary connectivity and local
    valence rules.

    Parameters
    ----------
    pac : PAC
        PAC grid for which candidate actions are generated. It is not modified.
    filter : Filter or None, default None
        Optional selection of action types to include.

    Returns
    -------
    list of Action
        Legal candidate actions for the current grid.
    """
    actions: list[Action] = []

    x_min, x_max, y_min, y_max = pac.calculate_bounding_box(negative_indexing=True)
    rows = np.array(range(y_min, y_max + 1))
    rows = np.where(
        rows < 0,
        rows + pac.grid_size,
        rows,
    )
    cols = np.array(range(x_min, x_max + 1))
    cols = np.where(
        cols < 0,
        cols + pac.grid_size,
        cols,
    )

    for y in rows:
        if (
            filter is not None
            and not filter.add_benzene
            and not filter.add_five_membered
        ):
            break

        x = cols[np.isin(cols, pac.framework[y].nonzero()[1])]
        if len(x) == 0:
            continue

        # Each entry is shifted position and direction shifted by
        directions = [
            (pac.shift(x[0], y, EdgeDirection.LEFT), EdgeDirection.LEFT),
            (pac.shift(x[-1], y, EdgeDirection.RIGHT), EdgeDirection.RIGHT),
            (pac.shift(x[0], y, EdgeDirection.TOP_LEFT), EdgeDirection.TOP_LEFT),
            (pac.shift(x[0], y, EdgeDirection.BOTTOM_LEFT), EdgeDirection.BOTTOM_LEFT),
            (pac.shift(x[-1], y, EdgeDirection.TOP_RIGHT), EdgeDirection.TOP_RIGHT),
            (
                pac.shift(x[-1], y, EdgeDirection.BOTTOM_RIGHT),
                EdgeDirection.BOTTOM_RIGHT,
            ),
        ]

        # ======[ Benzene Actions ]======
        if filter is None or filter.add_benzene:
            for pos, _ in directions:
                if pac.is_legal_benzene_position(*pos):
                    actions.append(Action(pos[0], pos[1], ActionType.ADD_BENZENE))

        # ======[ 5-Membered Ring Actions ]======
        if filter is None or filter.add_five_membered:
            for pos, edge_dir in directions:
                if pac.is_legal_five_membered_position(pos[0], pos[1]):
                    legal_missing_vertices = FIVE_RING_CONFIGS[edge_dir]
                    for missing_vertex in legal_missing_vertices:
                        action_type = ACTION_TYPE_FROM_VERTEX_DIR[missing_vertex]
                        actions.append(Action(pos[0], pos[1], action_type))

    # Functional groups
    # Extract positions of all framework rings (both 1 and 2)
    ring_positions = np.argwhere(pac.framework.toarray() > 0)
    for y, x in ring_positions:
        if filter is not None and not filter.add_functional_groups:
            break
        for vertex_direction in VertexDirection.all():
            can_add_fn_groups, can_add_carbonyl = pac.is_legal_functional_group_vertex(
                x, y, vertex_direction
            )
            if can_add_fn_groups:
                channel = (
                    vertex_direction.value + 1
                )  # Functional group channels are 1-indexed
                for action_type in ActionType.add_functional_groups_no_carbonyl():
                    actions.append(Action(x, y, action_type, channel=channel))
            if can_add_carbonyl:
                channel = vertex_direction.value + 1
                actions.append(Action(x, y, ActionType.ADD_CARBONYL, channel=channel))

    # ======[ Atom Types ]======
    for y, x in ring_positions:
        for vertex_direction in VertexDirection.all():
            can_add_nitrogen_sp2, can_add_carbon_sp2 = (
                pac.is_legal_atom_type_sp2_vertex(x, y, vertex_direction)
            )
            can_add_carbon, can_add_nitrogen, can_add_oxygen_sulfur = (
                pac.is_legal_atom_type_sp3_vertex(x, y, vertex_direction)
            )
            can_add_oxygen, can_add_sulfur = (
                (True, True) if can_add_oxygen_sulfur else (False, False)
            )

            if filter is not None:
                if not filter.add_carbon_sp3:
                    can_add_carbon = False
                if not filter.add_nitrogen:
                    can_add_nitrogen = False
                    can_add_nitrogen_sp2 = False
                if not filter.add_oxygen_sp3:
                    can_add_oxygen = False
                if not filter.add_sulfur_sp3:
                    can_add_sulfur = False

            channel = vertex_direction.value + 7
            if can_add_nitrogen_sp2:
                actions.append(
                    Action(x, y, ActionType.ADD_NITROGEN_SP2, channel=channel)
                )
            if can_add_carbon_sp2:
                actions.append(Action(x, y, ActionType.ADD_CARBON_SP2, channel=channel))
            if can_add_carbon:
                actions.append(Action(x, y, ActionType.ADD_CARBON_SP3, channel=channel))
            if can_add_nitrogen:
                actions.append(
                    Action(x, y, ActionType.ADD_NITROGEN_SP3, channel=channel)
                )
            if can_add_oxygen:
                actions.append(Action(x, y, ActionType.ADD_OXYGEN_SP3, channel=channel))
            if can_add_sulfur:
                actions.append(Action(x, y, ActionType.ADD_SULFUR_SP3, channel=channel))

    return actions


def take_action(pac: PAC, action: Action) -> None:
    """
    Apply a structural action to a PAC grid in place.

    Parameters
    ----------
    pac : PAC
        PAC grid to modify.
    action : Action
        Structural action to apply.
    """
    if action.type.is_add_five_membered():
        # A 5-membered ring takes (framework value, empty vertex)
        pac[action.channel, action.x, action.y] = (2, action.type.get_missing_vertex())
    else:
        pac[action.channel, action.x, action.y] = action.fill_value
