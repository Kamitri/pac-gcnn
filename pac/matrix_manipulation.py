"""Rotate and reflect PAC grids stored as arrays or tensors.

Operations accept 2D grids and 3D batches. Circular shifts implement periodic grid boundaries.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import torch

Matrix = np.ndarray | torch.Tensor


def _spatial_axes(matrix: Matrix) -> tuple[int, int]:
    """
    Return the row and column axes of a grid or stack of grids.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid with shape ``(num_rows, num_columns)`` or stacked grids with shape ``(num_grids, num_rows,
        num_columns)``.

    Returns
    -------
    tuple of int
        Row and column axis indices. The result is ``(0, 1)`` for 1 grid and ``(1, 2)`` for stacked grids.

    Raises
    ------
    ValueError
        Raised when `matrix` is not two- or three-dimensional.
    """
    if matrix.ndim == 2:
        return 0, 1
    if matrix.ndim == 3:
        return 1, 2
    raise ValueError(
        f"Expected input to be a 2D grid or a 3D stack of grids, got {matrix.ndim} dimensions."
    )


def _copy(matrix: Matrix) -> Matrix:
    if isinstance(matrix, torch.Tensor):
        return matrix.clone()
    return np.copy(matrix)


def _roll(matrix: Matrix, shift: int, axis: int) -> Matrix:
    if isinstance(matrix, torch.Tensor):
        return torch.roll(matrix, shifts=shift, dims=axis)
    return np.roll(matrix, shift=shift, axis=axis)


def _flip(matrix: Matrix, axis: int) -> Matrix:
    if isinstance(matrix, torch.Tensor):
        return torch.flip(matrix, dims=[axis])
    return np.flip(matrix, axis=axis)


def rotate_matrix_counterclockwise(matrix: Matrix) -> Matrix:
    """
    Rotate the underlying square array 90 degrees counterclockwise.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid or stack of grids. The input is not modified.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Rotated array or tensor. The spatial dimensions are exchanged. A NumPy result shares memory with the input,
        whereas a PyTorch result is a new tensor.

    Notes
    -----
    After the rotation, the element of the upper-left corner is in the lower-left corner.
    """
    row_axis, column_axis = _spatial_axes(matrix)
    if isinstance(matrix, torch.Tensor):
        return torch.rot90(matrix, k=1, dims=(row_axis, column_axis))
    return np.rot90(matrix, k=1, axes=(row_axis, column_axis))


def shift_columns_from_center(matrix: Matrix, center_column: int) -> Matrix:
    """
    Apply the column shear used for hexagonal lattice transformations.

    Each column is shifted along the periodic row axis by its signed distance from `center_column`. The central column
    is unchanged, columns to its right are shifted down, and columns to its left are shifted up.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid or stack of grids. The input is not modified.
    center_column : int
        Column about which the shear is applied.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Sheared grid with the same type, shape, and data type as `matrix`.
    """
    _, column_axis = _spatial_axes(matrix)
    num_columns = matrix.shape[column_axis]
    shifted_matrix = _copy(matrix)

    for column_index in range(num_columns):
        distance_from_center = column_index - center_column
        if distance_from_center == 0:
            continue
        # pyrefly does not relate array type of `shifted_matrix` to array type that `_roll` returns
        if matrix.ndim == 2:
            shifted_matrix[:, column_index] = _roll(  # pyrefly: ignore[unsupported-operation]
                matrix[:, column_index], distance_from_center, axis=0
            )
        else:
            shifted_matrix[:, :, column_index] = _roll(  # pyrefly: ignore[unsupported-operation]
                matrix[:, :, column_index], distance_from_center, axis=1
            )

    return shifted_matrix


def shift_rows_from_center(matrix: Matrix, center_row: int) -> Matrix:
    """
    Apply the row shear used for hexagonal lattice transformations.

    Each row is shifted along the periodic column axis by its signed distance from `center_row`. The central row is
    unchanged, rows below it are shifted right, and rows above it are shifted left.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid or stack of grids. The input is not modified.
    center_row : int
        Row about which the shear is applied.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Sheared grid with the same type, shape, and data type as `matrix`.
    """
    row_axis, _ = _spatial_axes(matrix)
    num_rows = matrix.shape[row_axis]
    shifted_matrix = _copy(matrix)

    for row_index in range(num_rows):
        distance_from_center = row_index - center_row
        if distance_from_center == 0:
            continue
        if matrix.ndim == 2:
            shifted_matrix[row_index, :] = _roll(
                matrix[row_index, :], distance_from_center, axis=0
            )
        else:
            shifted_matrix[:, row_index, :] = _roll(
                matrix[:, row_index, :], distance_from_center, axis=1
            )

    return shifted_matrix


def flip_up_down(matrix: Matrix) -> Matrix:
    """
    Reverse a grid along its row axis.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid or stack of grids. The input is not modified.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Row-reversed grid with the same type, shape, and data type as `matrix`. A NumPy result shares memory with the
        input, whereas a PyTorch result is a new tensor.
    """
    row_axis, _ = _spatial_axes(matrix)
    return _flip(matrix, row_axis)


def flip_left_right(matrix: Matrix) -> Matrix:
    """
    Reverse a grid along its column axis.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid or stack of grids. The input is not modified.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Column-reversed grid with the same type, shape, and data type as `matrix`. A NumPy result shares memory with the
        input, whereas a PyTorch result is a new tensor.
    """
    _, column_axis = _spatial_axes(matrix)
    return _flip(matrix, column_axis)


def _resolve_center(matrix: Matrix, center: tuple[int, int] | None) -> tuple[int, int]:
    if center is not None:
        return center
    row_axis, column_axis = _spatial_axes(matrix)
    return matrix.shape[row_axis] // 2, matrix.shape[column_axis] // 2


@overload
def rotate(
    matrix: torch.Tensor, center: tuple[int, int] | None = None
) -> torch.Tensor: ...


@overload
def rotate(matrix: np.ndarray, center: tuple[int, int] | None = None) -> np.ndarray: ...


def rotate(matrix: Matrix, center: tuple[int, int] | None = None) -> Matrix:
    """
    Rotate a grid 60 degrees counterclockwise on the hexagonal lattice.

    The transformation consists of a 90-degree array rotation followed by a column shear about the selected centre.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid or stack of grids.
    center : tuple of int or None, default None
        Spatial index ``(row_index, column_index)`` about which the grid is rotated. The grid centre is used when this
        is None.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Rotated grid. The input is not modified.
    """
    _, center_column = _resolve_center(matrix, center)
    rotated_matrix = rotate_matrix_counterclockwise(_copy(matrix))
    return shift_columns_from_center(rotated_matrix, center_column)


@overload
def reflect(
    matrix: torch.Tensor, center: tuple[int, int] | None = None
) -> torch.Tensor: ...


@overload
def reflect(
    matrix: np.ndarray, center: tuple[int, int] | None = None
) -> np.ndarray: ...


def reflect(matrix: Matrix, center: tuple[int, int] | None = None) -> Matrix:
    """
    Reflect a grid across the vertical axis of the hexagonal lattice.

    The transformation consists of reversing the row order and then applying a row shear about the selected centre.

    Parameters
    ----------
    matrix : numpy.ndarray or torch.Tensor
        Individual grid or stack of grids.
    center : tuple of int or None, default None
        Spatial index ``(row_index, column_index)`` on the reflection axis. The grid centre is used when this is None.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Reflected grid. The input is not modified.
    """
    center_row, _ = _resolve_center(matrix, center)
    reflected_matrix = flip_left_right(_copy(matrix))
    return shift_rows_from_center(reflected_matrix, center_row)
