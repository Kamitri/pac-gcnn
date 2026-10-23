"""
PyTorch Lightning modules and callbacks for training and evaluating the regression models.
"""

from collections.abc import Sequence
from typing import Any

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import nn
from torchmetrics import (
    MaxMetric,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MinMetric,
    R2Score,
)

from benchmarks.scaler import MultiTargetScaler


class MetricDict(nn.ModuleDict):
    """
    Store a `torchmetrics` metric for each property name.

    Indexing returns the stored module with the type `torchmetrics.Metric`, the type that `LightningModule.log` accepts
    as a logged value. For a module of another type, indexing raises TypeError.
    """

    def __getitem__(self, key: str) -> Metric:
        metric = super().__getitem__(key)
        if not isinstance(metric, Metric):
            raise TypeError(f"The module of {key!r} is not a torchmetrics metric.")
        return metric


class LightningModule(pl.LightningModule):
    """
    Train and evaluate a multi-target regression model with PyTorch Lightning.

    The loss is the mean squared error between the model output and the standardised targets, and the metrics are
    computed in the original units after the conversion with `scaler`. The optimiser is AdamW with a OneCycleLR
    schedule.

    Parameters
    ----------
    model : torch.nn.Module
        Regression model with 1 output column for each property.
    property_names : sequence of str
        Property names in the order of the output columns. The names are part of the keys of the logged metrics.
    scaler : MultiTargetScaler
        Scaler fitted to the training targets.
    lr : float, default 1e-3
        Maximum learning rate of the OneCycleLR schedule.
    weight_decay : float, default 1e-4
        Weight decay of AdamW.

    Attributes
    ----------
    criterion : torch.nn.MSELoss
        Loss function in the standardised space.
    train_rmsd, val_rmsd, test_rmsd : MetricDict
        RMSD metric of each property in the original units, by property name.
    test_mae, test_r2, test_max, test_min : MetricDict
        Mean absolute error, R2, maximum absolute error, and minimum absolute error of each property on the test set, by
        property name.
    show_val_loss_in_progress_bar : bool
        Whether the progress bar shows ``val_loss``. The value is False in this class.

    Notes
    -----
    The logged keys are ``learning_rate``, ``train_loss``, ``val_loss``, and ``test_loss`` in the standardised space,
    and ``train_rmsd_<name>``, ``val_rmsd_<name>``, ``test_rmsd_<name>``, ``test_mae_<name>``, ``test_r2_<name>``,
    ``test_max_<name>``, and ``test_min_<name>`` for each property name in the original units.
    """

    show_val_loss_in_progress_bar: bool = False

    def __init__(
        self,
        model: nn.Module,
        property_names: Sequence[str],
        scaler: MultiTargetScaler,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.model = model
        self.property_names = property_names
        self.scaler = scaler
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss()

        self.train_rmsd = MetricDict(
            {prop: MeanSquaredError(squared=False) for prop in property_names}
        )
        self.val_rmsd = MetricDict(
            {prop: MeanSquaredError(squared=False) for prop in property_names}
        )

        self.test_rmsd = MetricDict(
            {prop: MeanSquaredError(squared=False) for prop in property_names}
        )
        self.test_mae = MetricDict(
            {prop: MeanAbsoluteError() for prop in property_names}
        )
        self.test_r2 = MetricDict({prop: R2Score() for prop in property_names})
        self.test_max = MetricDict({prop: MaxMetric() for prop in property_names})
        self.test_min = MetricDict({prop: MinMetric() for prop in property_names})

    def forward(self, x: Any) -> torch.Tensor:
        """
        Compute the output of the wrapped model.

        Parameters
        ----------
        x : object
            Inputs of the batch in the format of the wrapped model.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_properties)
            Predictions in the standardised space.
        """
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        """
        Log the learning rate at the start of each training epoch.

        The logged value ``learning_rate`` is the learning rate of the first parameter group of the optimiser.
        """
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=False,
        )

    def _unscale_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch from the standardised space to the original units.

        Parameters
        ----------
        tensor : torch.Tensor of shape (batch_size, num_properties)
            Standardised predictions or targets.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_properties)
            Values in the original units.
        """
        return self.scaler.inverse_transform(tensor)

    def _split_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """
        Return the inputs and the standardised targets of a batch.

        Parameters
        ----------
        batch : sequence
            ``(inputs, targets)`` with standardised targets of batch_size * num_properties elements.

        Returns
        -------
        tuple of (object, torch.Tensor)
            The inputs in the format of the wrapped model and the standardised targets.
        """
        inputs, targets = batch
        return inputs, targets

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Compute the loss of 1 training batch and add the batch to the RMSD of each property.

        The loss is logged as ``train_loss``, the mean over the epoch weighted by the batch size.

        Parameters
        ----------
        batch : object
            Batch that `_split_batch` separates into the inputs and the standardised targets.
        batch_idx : int
            Index of the batch in the epoch.

        Returns
        -------
        torch.Tensor
            Scalar loss for the optimiser step.
        """
        inputs, targets = self._split_batch(batch)
        outputs = self(inputs)
        targets = targets.view(outputs.shape)
        loss = self.criterion(outputs, targets)
        self.log(
            "train_loss",
            loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=targets.size(0),
        )

        outputs_unscaled = self._unscale_batch(outputs)
        targets_unscaled = self._unscale_batch(targets)

        for i, prop in enumerate(self.property_names):
            self.train_rmsd[prop](outputs_unscaled[:, i], targets_unscaled[:, i])
            self.log(
                f"train_rmsd_{prop}",
                self.train_rmsd[prop],
                on_step=False,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Compute the loss of 1 validation batch and add the batch to the RMSD of each property.

        The loss is logged as ``val_loss``, the mean over the epoch weighted by the batch size.

        Parameters
        ----------
        batch : object
            Batch that `_split_batch` separates into the inputs and the standardised targets.
        batch_idx : int
            Index of the batch in the epoch.

        Returns
        -------
        torch.Tensor
            Scalar loss in the standardised space.
        """
        inputs, targets = self._split_batch(batch)
        outputs = self(inputs)
        targets = targets.view(outputs.shape)
        loss = self.criterion(outputs, targets)
        self.log(
            "val_loss",
            loss,
            prog_bar=self.show_val_loss_in_progress_bar,
            on_step=False,
            on_epoch=True,
            batch_size=targets.size(0),
        )

        outputs_unscaled = self._unscale_batch(outputs)
        targets_unscaled = self._unscale_batch(targets)

        for i, prop in enumerate(self.property_names):
            self.val_rmsd[prop](outputs_unscaled[:, i], targets_unscaled[:, i])
            self.log(
                f"val_rmsd_{prop}", self.val_rmsd[prop], on_step=False, on_epoch=True
            )

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Compute the loss of 1 test batch and add the batch to the test metrics of each property.

        The loss is logged as ``test_loss``, the mean over the epoch weighted by the batch size, and the test metrics
        are logged at the end of the test epoch under the keys listed in the Notes section of the class.

        Parameters
        ----------
        batch : object
            Batch that `_split_batch` separates into the inputs and the standardised targets.
        batch_idx : int
            Index of the batch in the test loader.

        Returns
        -------
        torch.Tensor
            Scalar loss in the standardised space.
        """
        inputs, targets = self._split_batch(batch)
        outputs = self(inputs)
        targets = targets.view(outputs.shape)
        loss = self.criterion(outputs, targets)
        self.log("test_loss", loss, batch_size=targets.size(0))

        outputs_unscaled = self._unscale_batch(outputs)
        targets_unscaled = self._unscale_batch(targets)

        for i, prop in enumerate(self.property_names):
            pred = outputs_unscaled[:, i]
            true = targets_unscaled[:, i]
            error = torch.abs(pred - true)

            self.test_rmsd[prop](pred, true)
            self.test_mae[prop](pred, true)
            self.test_r2[prop].update(pred, true)
            self.test_max[prop](torch.max(error))
            self.test_min[prop](torch.min(error))

            self.log(
                f"test_rmsd_{prop}", self.test_rmsd[prop], on_step=False, on_epoch=True
            )
            self.log(
                f"test_mae_{prop}", self.test_mae[prop], on_step=False, on_epoch=True
            )
            self.log(
                f"test_r2_{prop}", self.test_r2[prop], on_step=False, on_epoch=True
            )
            self.log(
                f"test_max_{prop}", self.test_max[prop], on_step=False, on_epoch=True
            )
            self.log(
                f"test_min_{prop}", self.test_min[prop], on_step=False, on_epoch=True
            )

        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """
        Create the AdamW optimiser and the OneCycleLR learning-rate schedule.

        The schedule runs for ``trainer.estimated_stepping_batches`` optimiser steps using default
        `torch.optim.lr_scheduler.OneCycleLR` settings and updates after each optimiser step.

        Returns
        -------
        OptimizerLRSchedulerConfig
            Lightning configuration with the keys ``optimizer`` and ``lr_scheduler``.

        Raises
        ------
        ValueError
            Raised when the trainer runs an infinite number of training steps.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches
        if not isinstance(total_steps, int):
            raise ValueError("OneCycleLR requires a finite number of training steps.")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class ChemPropLightningModule(LightningModule):
    """
    Train and evaluate a Chemprop message-passing model with PyTorch Lightning.

    The class has the loss, the metrics, the logged keys, and the optimiser of `LightningModule`, for batches of the
    type `chemprop.data.TrainingBatch`. The validation loss is also shown in the progress bar.

    Parameters
    ----------
    model : torch.nn.Module
        Chemprop regression model with 1 output column for each property.
    property_names : sequence of str
        Property names in the order of the output columns. The names are part of the keys of the logged metrics.
    scaler : MultiTargetScaler
        Scaler fitted to the training targets.
    lr : float, default 1e-3
        Maximum learning rate of the OneCycleLR schedule.
    weight_decay : float, default 1e-4
        Weight decay of AdamW.
    """

    show_val_loss_in_progress_bar = True

    def _split_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        """
        Return the molecular graphs and the standardised targets of a Chemprop batch.

        Parameters
        ----------
        batch : chemprop.data.TrainingBatch
            Batch with the molecular graphs ``bmg`` and the standardised targets ``Y``.

        Returns
        -------
        tuple of (chemprop.data.BatchMolGraph, torch.Tensor)
            ``batch.bmg`` and ``batch.Y``.
        """
        return batch.bmg, batch.Y


class PrintEpochMetricsCallback(Callback):
    """
    Print the losses and the validation RMSDs at the end of each validation epoch.

    After every validation epoch except the sanity check, the callback prints 1 line with the epoch number, the training
    and the validation loss in the standardised space, and the validation RMSD of each property in the original units. A
    missing loss is printed as ``N/A``, and a missing RMSD is excluded from the line. Because Lightning computes the
    epoch value of ``train_loss`` after the validation epoch, the training loss on the line of epoch N is the loss of
    epoch N - 1, and the line of epoch 1 shows ``N/A``.

    Parameters
    ----------
    property_names : sequence of str
        Properties of the RMSD values on the printed line.
    """

    def __init__(self, property_names: Sequence[str]) -> None:
        self.property_names = property_names

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Print the metric line of the finished validation epoch.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            Trainer with the logged metrics.
        pl_module : lightning.pytorch.LightningModule
            Module in training.
        """
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch + 1

        train_loss = metrics.get("train_loss")
        train_str = f"{train_loss:.5f}" if train_loss is not None else "N/A"

        val_loss = metrics.get("val_loss")
        val_str = f"{val_loss:.5f}" if val_loss is not None else "N/A"

        rmsd_strs = []
        for prop in self.property_names:
            val = metrics.get(f"val_rmsd_{prop}")
            if val is not None:
                rmsd_strs.append(f"{prop} Val RMSD: {val:.5f}")

        rmsd_str = " | ".join(rmsd_strs)

        print(
            f"Epoch {epoch:03d} | Scaled Train Loss: {train_str} | Scaled Val Loss: {val_str} | {rmsd_str}"
        )


class BestEpochTracker(Callback):
    """
    Record the epoch with the lowest validation loss.

    The sanity check and an epoch without ``val_loss`` in ``trainer.callback_metrics`` do not change the attributes.
    Epochs are counted from 1, and of 2 epochs with the same lowest loss, `best_epoch` is the earlier one.

    Attributes
    ----------
    best_loss : float or None
        Lowest validation loss, or None before the first epoch with a validation loss.
    best_epoch : int or None
        Epoch of `best_loss`, or None before the first epoch with a validation loss.
    last_epoch : int
        Last epoch with a validation loss, or 0 before the first one.
    """

    def __init__(self) -> None:
        self.best_loss: float | None = None
        self.best_epoch: int | None = None
        self.last_epoch = 0

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Record the validation loss of the finished validation epoch.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            Trainer with the logged metrics.
        pl_module : lightning.pytorch.LightningModule
            Module in training.
        """
        if trainer.sanity_checking:
            return
        logged_loss = trainer.callback_metrics.get("val_loss")
        if logged_loss is None:
            return
        validation_loss = float(logged_loss)
        self.last_epoch = trainer.current_epoch + 1
        if self.best_loss is None or validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_epoch = self.last_epoch
