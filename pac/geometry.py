"""PAC lattice distances and D6 transformations.

A tile has axial coordinates (column_index, row_index). The column axis points to the right, while the row axis points
down and to the left.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def rotated_channel_index(index: int, num_rotation_steps: int = 1) -> int:
    """
    Return the source direction for a counterclockwise channel rotation.

    Parameters
    ----------
    index : int
        Vertex-direction index in the transformed PAC grid.
    num_rotation_steps : int, default 1
        Number of counterclockwise 60-degree rotations.

    Returns
    -------
    int
        Corresponding vertex-direction index in the input PAC grid.
    """
    return (index + num_rotation_steps) % 6


def reflected_channel_index(index: int) -> int:
    """
    Return the source direction for reflection across the vertical axis.

    Parameters
    ----------
    index : int
        Vertex-direction index in the reflected PAC grid.

    Returns
    -------
    int
        Corresponding vertex-direction index in the input PAC grid.
    """
    return (-index) % 6


def hexagon_centre(column_index: int, row_index: int) -> tuple[float, float]:
    """
    Return the Cartesian centre of the hexagon at a position, for hexagons with a circumradius of 1.

    The x axis points to the right and the y axis points up.

    Parameters
    ----------
    column_index : int
        Axial column coordinate.
    row_index : int
        Axial row coordinate.

    Returns
    -------
    tuple of float
        Cartesian ``(x, y)`` coordinates.
    """
    return math.sqrt(3.0) * (column_index - row_index / 2.0), -1.5 * row_index


def hexagonal_distance(column_offset: int, row_offset: int) -> int:
    """
    Return the hexagonal distance of an axial offset.

    Parameters
    ----------
    column_offset : int
        Difference between 2 axial column coordinates.
    row_offset : int
        Difference between 2 axial row coordinates.

    Returns
    -------
    int
        The largest of the absolute column offset, the absolute row offset, and the absolute difference between the two
        offsets.
    """
    return max(abs(column_offset), abs(row_offset), abs(column_offset - row_offset))


def periodic_hexagonal_distance(
    position: tuple[int, int], other_position: tuple[int, int], grid_size: int
) -> int:
    """
    Return the hexagonal distance between 2 positions on the torus.

    Parameters
    ----------
    position : tuple of int
        ``(column_index, row_index)`` of the first tile.
    other_position : tuple of int
        ``(column_index, row_index)`` of the second tile.
    grid_size : int
        Number of tiles along each grid axis.

    Returns
    -------
    int
        The minimum `hexagonal_distance` over all periodic images of the offset between the 2 positions.

    Notes
    -----
    Each offset coordinate is first reduced to the range 0 to ``grid_size - 1``. The reduced offset has a distance of at
    most ``grid_size - 1``, whereas an offset with either coordinate at least ``grid_size`` or at most ``-grid_size``
    has a distance of at least ``grid_size``. The minimum is therefore found among the 4 images with coordinates in the
    range ``-grid_size`` to ``grid_size - 1``.
    """
    column_offset = (other_position[0] - position[0]) % grid_size
    row_offset = (other_position[1] - position[1]) % grid_size
    return min(
        hexagonal_distance(column_offset - column_wrap, row_offset - row_wrap)
        for column_wrap in (0, grid_size)
        for row_wrap in (0, grid_size)
    )


def d6_orbit(
    grid_like: Any,
    rotate: Callable[[Any], Any],
    reflect: Callable[[Any], Any],
    realign: Callable[[Any], None] | None = None,
) -> list:
    """
    Generate the D6 orbit of a PAC grid.

    The first 6 images are counterclockwise rotations. The remaining 6 images are the corresponding rotations after
    reflection. `gcnn.transform` uses the same group-element order.

    Parameters
    ----------
    grid_like : object
        PAC grid to transform.
    rotate : callable
        Function that rotates the representation 60 degrees counterclockwise.
    reflect : callable
        Function that reflects the representation across its vertical axis.
    realign : callable or None, default None
        Optional function that realigns each image in place, except the first image, which is `grid_like` itself.

    Returns
    -------
    list
        Images in D6 order.
    """
    orbit = [grid_like]

    current = grid_like
    for _ in range(5):
        current = rotate(current)
        if realign is not None:
            realign(current)
        orbit.append(current)

    current = reflect(grid_like)
    if realign is not None:
        realign(current)
    orbit.append(current)
    for _ in range(5):
        current = rotate(current)
        if realign is not None:
            realign(current)
        orbit.append(current)

    return orbit
