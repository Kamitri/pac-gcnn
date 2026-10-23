"""
Target standardisation.
"""

from typing import overload

import numpy as np
import torch


class MultiTargetScaler:
    """
    Standardise each regression target independently.

    Attributes
    ----------
    mean : numpy.ndarray of shape (num_targets,) or None
        Mean of each target, or None before the first call of `fit`.
    std : numpy.ndarray of shape (num_targets,) or None
        Population standard deviation of each target, with 1.0 in place of 0, or None before the first call of `fit`.
    """

    def __init__(self) -> None:
        self.mean = None
        self.std = None
        self._tensor_cache = {}

    def fit(self, targets: np.ndarray) -> None:
        """
        Compute the mean and the population standard deviation of each target.

        Parameters
        ----------
        targets : numpy.ndarray of shape (num_samples, num_targets)
            Target values in the original units.
        """
        self.mean = np.mean(targets, axis=0)
        self.std = np.std(targets, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        self._tensor_cache = {}

    def _get_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the fitted statistics as tensors on a device.

        The pair of tensors of a combination of device and dtype is created at the first call and returned again by
        later calls until the next call of `fit`.

        Parameters
        ----------
        device : torch.device
            Device of the returned tensors.
        dtype : torch.dtype
            Floating-point type of the returned tensors.

        Returns
        -------
        mean : torch.Tensor of shape (num_targets,)
            Mean of each target.
        std : torch.Tensor of shape (num_targets,)
            Standard deviation of each target.
        """
        key = (device, dtype)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = (
                torch.tensor(self.mean, device=device, dtype=dtype),
                torch.tensor(self.std, device=device, dtype=dtype),
            )
        return self._tensor_cache[key]

    @overload
    def transform(self, data: torch.Tensor) -> torch.Tensor: ...

    @overload
    def transform(self, data: np.ndarray) -> np.ndarray: ...

    def transform(self, data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """
        Standardise values with the fitted statistics.

        The method returns ``(data - mean) / std`` for each target column.

        Parameters
        ----------
        data : torch.Tensor or numpy.ndarray of shape (num_samples, num_targets)
            Values in the original units.

        Returns
        -------
        torch.Tensor or numpy.ndarray of shape (num_samples, num_targets)
            Standardised values, of the array type of `data`. A tensor result has the device and the dtype of `data`.
        """
        if isinstance(data, torch.Tensor):
            mean, std = self._get_tensors(data.device, data.dtype)
            return (data - mean) / std
        return (data - self.mean) / self.std

    @overload
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor: ...

    @overload
    def inverse_transform(self, data: np.ndarray) -> np.ndarray: ...

    def inverse_transform(
        self, data: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """
        Convert standardised values back to the original units.

        The method returns ``data * std + mean`` for each target column.

        Parameters
        ----------
        data : torch.Tensor or numpy.ndarray of shape (num_samples, num_targets)
            Standardised targets or predictions.

        Returns
        -------
        torch.Tensor or numpy.ndarray of shape (num_samples, num_targets)
            Values in the original units, of the array type of `data`. A tensor result has the device and the dtype of
            `data`.
        """
        if isinstance(data, torch.Tensor):
            mean, std = self._get_tensors(data.device, data.dtype)
            return (data * std) + mean
        return (data * self.std) + self.mean
