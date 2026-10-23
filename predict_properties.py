"""
Predict electronic and thermodynamic properties from PAC structures.

Command line, run from the repository root::

    python predict_properties.py "c1ccc2ccccc2c1"
    python predict_properties.py "c1ccccc1" --theory pm7 --property homo

Python interface: `predict_properties`.

Available predictions:
    GFN2-xTB has HOMO and LUMO energies and the HOMO-LUMO gap (eV). PM7 has the same electronic properties and the heat
    of formation (kJ/mol). CAM-B3LYP predictions are available through the DIRECT, DELTA, FINETUNE, TRANSFER,
    FINETUNE-DELTA, and TRANSFER-DELTA methods.

Required files:
    Model checkpoints are obtained from `MODELS_DIRECTORY`. Dataset partitions and target standardisation statistics are
    obtained from `SPLIT_DIRECTORY`.
"""

import json
import os
import sys
from collections.abc import Sequence
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import typer
from rdkit import Chem
from torch import nn

from benchmarks.scaler import MultiTargetScaler
from dataset.pac5_database import MAX_EMBEDDING_ATTEMPTS
from gcnn.base_model import GroupEquivariantCNN
from pac import PAC, smiles_to_pac

REPOSITORY_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIRECTORY = os.path.join(REPOSITORY_ROOT, "models")
SPLIT_DIRECTORY = os.path.join(REPOSITORY_ROOT, "dataset", "PAC5-1")

THEORY_NAMES = ("PM7", "GFN2-xTB", "CAM-B3LYP")
THEORY_ALIASES = {
    "pm7": "PM7",
    "gfn2-xtb": "GFN2-xTB",
    "gfn2xtb": "GFN2-xTB",
    "xtb": "GFN2-xTB",
    "cam-b3lyp": "CAM-B3LYP",
    "camb3lyp": "CAM-B3LYP",
}

PROPERTY_NAMES = ("HOMO", "LUMO", "HL-gap", "heat-of-formation")
PROPERTY_ALIASES = {
    "homo": "HOMO",
    "lumo": "LUMO",
    "hl-gap": "HL-gap",
    "hlgap": "HL-gap",
    "hl": "HL-gap",
    "hl_gap": "HL-gap",
    "band_gap": "HL-gap",
    "band-gap": "HL-gap",
    "heat-of-formation": "heat-of-formation",
    "heat_of_formation": "heat-of-formation",
    "hof": "heat-of-formation",
}

HIGH_FIDELITY_METHOD_NAMES = (
    "DIRECT",
    "DELTA",
    "FINETUNE",
    "TRANSFER",
    "FINETUNE-DELTA",
    "TRANSFER-DELTA",
)
HIGH_FIDELITY_METHOD_ALIASES = {
    "direct": "DIRECT",
    "delta": "DELTA",
    "finetune": "FINETUNE",
    "fine-tune": "FINETUNE",
    "transfer": "TRANSFER",
    "finetune-delta": "FINETUNE-DELTA",
    "finetune_delta": "FINETUNE-DELTA",
    "transfer-delta": "TRANSFER-DELTA",
    "transfer_delta": "TRANSFER-DELTA",
}
# Correction methods
DELTA_METHODS = frozenset({"DELTA", "FINETUNE-DELTA", "TRANSFER-DELTA"})

PROPERTY_UNITS = {
    "HOMO": "eV",
    "LUMO": "eV",
    "HL-gap": "eV",
    "heat-of-formation": "kJ/mol",
}
# Format output
PROPERTY_DISPLAY_NAMES = {
    "HOMO": "HOMO",
    "LUMO": "LUMO",
    "HL-gap": "HL-gap",
    "heat-of-formation": "heat of formation",
}

# Target columns
LOW_LEVEL_PROPERTIES = (
    "xtb_homo",
    "xtb_lumo",
    "xtb_hl_gap",
    "pm7_homo",
    "pm7_lumo",
    "pm7_hl_gap",
    "pm7_hof",
)
LOW_LEVEL_PROPERTY_TYPES = (
    "intensive",
    "intensive",
    "intensive",
    "intensive",
    "intensive",
    "intensive",
    "extensive",
)
LOW_LEVEL_COLUMNS = {
    ("GFN2-xTB", "HOMO"): "xtb_homo",
    ("GFN2-xTB", "LUMO"): "xtb_lumo",
    ("GFN2-xTB", "HL-gap"): "xtb_hl_gap",
    ("PM7", "HOMO"): "pm7_homo",
    ("PM7", "LUMO"): "pm7_lumo",
    ("PM7", "HL-gap"): "pm7_hl_gap",
    ("PM7", "heat-of-formation"): "pm7_hof",
}
LOW_LEVEL_KERNEL_SIZE = 5
LOW_LEVEL_NUM_HIDDEN = 3
LOW_LEVEL_HIDDEN_CHANNELS = 32
LOW_LEVEL_FOLD = 1
LOW_LEVEL_DIRECTORY_TEMPLATE = "gcnn_1.1M-pac_grid-[PAC5-1]-fold_{fold}"

HIGH_FIDELITY_KERNEL_SIZE = 5
HIGH_FIDELITY_NUM_HIDDEN = 5
HIGH_FIDELITY_HIDDEN_CHANNELS = 48
HIGH_FIDELITY_DIRECTORY_STEMS = {
    "HL-gap": "gcnn_3.7M-pac_grid-[PAC5-hl_gap-to_camb3lyp_hl_gap]",
    "heat-of-formation": "gcnn_3.7M-pac_grid-[PAC5-hof-to_nist_hof]",
}
HIGH_FIDELITY_DIRECTORY_SUFFIXES = {
    "DIRECT": "-direct",
    "DELTA": "-delta",
    "FINETUNE": "-finetune",
    "TRANSFER": "-transfer",
    "FINETUNE-DELTA": "-finetune_delta",
    "TRANSFER-DELTA": "-transfer_delta",
}
HIGH_FIDELITY_PROPERTIES = ("HL-gap", "heat-of-formation")

DEFAULT_BATCH_SIZE = 256


class GCNNDirectPrediction(nn.Module):
    """
    Predict a CAM-B3LYP property without semiempirical pretraining.

    For the DIRECT method, a `GroupEquivariantCNN` feature extractor and an MLP regression head are trained from random
    initialisation on the high-fidelity subset.

    Parameters
    ----------
    kernel_size : int, default 5
        Side length of the square array containing each hexagonal kernel.
    num_hidden : int, default 5
        Number of group convolution layers after the lifting layer.
    hidden_channels : int, default 48
        Number of hidden channels per group element.
    extensive : bool, default False
        Use sum pooling when True and max pooling otherwise.

    Attributes
    ----------
    feature_extractor : gcnn.base_model.GroupEquivariantCNN
        D6-equivariant feature extractor with identity prediction heads.
    prediction_head : torch.nn.Sequential
        MLP regression head with batch normalisation and 2 128-unit SiLU layers.
    """

    def __init__(
        self,
        kernel_size: int = HIGH_FIDELITY_KERNEL_SIZE,
        num_hidden: int = HIGH_FIDELITY_NUM_HIDDEN,
        hidden_channels: int = HIGH_FIDELITY_HIDDEN_CHANNELS,
        extensive: bool = False,
    ) -> None:
        super().__init__()
        self.extensive = extensive
        self.feature_extractor = GroupEquivariantCNN(
            property_types=["intensive"],
            kernel_size=kernel_size,
            num_hidden=num_hidden,
            hidden_channels=hidden_channels,
        )
        self.feature_extractor.thermodynamic_head = nn.Identity()
        self.feature_extractor.electronic_head = nn.Identity()
        self.prediction_head = nn.Sequential(
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, pac_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict a property for a batch of PAC grids.

        Parameters
        ----------
        pac_tensor : torch.Tensor
            Floating-point PAC grids with shape ``(batch_size, 13, grid_size, grid_size)``.

        Returns
        -------
        torch.Tensor of shape (batch_size, 1)
            Predicted property values.
        """
        intensive_features, extensive_features = (
            self.feature_extractor.extract_features(pac_tensor)
        )
        features = extensive_features if self.extensive else intensive_features
        return self.prediction_head(features)


class GCNNDeltaCorrection(nn.Module):
    """
    Predict a CAM-B3LYP property from a PM7-pretrained feature extractor.

    DELTA, FINETUNE, TRANSFER, FINETUNE-DELTA, and TRANSFER-DELTA all begin with a `GroupEquivariantCNN` trained on PM7
    targets.

    With `delta=True`, the PM7 value is concatenated with the invariant molecular features. The regression head predicts
    a correction that is added to the PM7 value. This configuration is used for DELTA, FINETUNE-DELTA, and
    TRANSFER-DELTA.

    With `delta=False`, the regression head predicts the CAM-B3LYP value from the invariant features alone. This
    configuration is used for FINETUNE and TRANSFER.

    Parameters
    ----------
    kernel_size : int, default 5
        Side length of the square array containing each hexagonal kernel.
    num_hidden : int, default 5
        Number of group convolution layers after the lifting layer.
    hidden_channels : int, default 48
        Number of hidden channels per group element.
    delta : bool, default False
        Predict a correction to an input PM7 value when True.
    extensive : bool, default False
        Use spatial sum pooling when True and spatial max pooling otherwise.

    Attributes
    ----------
    feature_extractor : gcnn.base_model.GroupEquivariantCNN
        PM7-pretrained D6-equivariant feature extractor.
    correction_head : torch.nn.Sequential
        Regression head acting on the invariant features and, for delta methods, the PM7 value.
    """

    def __init__(
        self,
        kernel_size: int = HIGH_FIDELITY_KERNEL_SIZE,
        num_hidden: int = HIGH_FIDELITY_NUM_HIDDEN,
        hidden_channels: int = HIGH_FIDELITY_HIDDEN_CHANNELS,
        delta: bool = False,
        extensive: bool = False,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.extensive = extensive
        self.feature_extractor = GroupEquivariantCNN(
            property_types=["intensive"],
            kernel_size=kernel_size,
            num_hidden=num_hidden,
            hidden_channels=hidden_channels,
        )
        self.feature_extractor.thermodynamic_head = nn.Identity()
        self.feature_extractor.electronic_head = nn.Identity()
        head_inputs = hidden_channels + 1 if delta else hidden_channels
        self.correction_head = nn.Sequential(
            nn.BatchNorm1d(head_inputs),
            nn.Linear(head_inputs, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, pac_tensor: torch.Tensor, low_level_value: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the high-fidelity property for a batch of PAC grids.

        Parameters
        ----------
        pac_tensor : torch.Tensor
            Floating-point PAC grids with shape ``(batch_size, 13, grid_size, grid_size)``.
        low_level_value : torch.Tensor or None, default None
            PM7 property values with shape ``(batch_size,)`` or ``(batch_size, 1)``. Required when `delta` is True.

        Returns
        -------
        predicted : torch.Tensor of shape (batch_size, 1)
            High-fidelity prediction. For a delta method, this equals `low_level_value` plus `correction`.
        correction : torch.Tensor of shape (batch_size, 1)
            Predicted correction. This tensor is 0 when `delta` is False.

        Raises
        ------
        ValueError
            Raised when `delta` is True and `low_level_value` is None.
        """
        intensive_features, extensive_features = (
            self.feature_extractor.extract_features(pac_tensor)
        )
        features = extensive_features if self.extensive else intensive_features
        if not self.delta:
            predicted = self.correction_head(features)
            return predicted, torch.zeros_like(predicted)
        if low_level_value is None:
            raise ValueError("A delta model requires `low_level_value`.")
        if low_level_value.dim() == 1:
            low_level_value = low_level_value.unsqueeze(1)
        combined = torch.cat([features, low_level_value], dim=1)
        correction = self.correction_head(combined)
        return low_level_value + correction, correction


def resolve_theory(text: str) -> str:
    """
    Resolve a level-of-theory option to its canonical form.

    Matching is case-insensitive and accepts the aliases in `THEORY_ALIASES`. For example, `"xtb"` is resolved to
    `"GFN2-xTB"`.

    Parameters
    ----------
    text : str
        Level of theory or accepted alias.

    Returns
    -------
    str
        Canonical entry from `THEORY_NAMES`.

    Raises
    ------
    ValueError
        Raised when `text` is not a recognised theory or alias.
    """
    resolved = THEORY_ALIASES.get(str(text).strip().lower())
    if resolved is None:
        raise ValueError(
            f"Unknown level of theory {text!r}. Use 1 of "
            + ", ".join(THEORY_NAMES)
            + " (xtb and camb3lyp are accepted too)."
        )
    return resolved


def resolve_property(text: str) -> str:
    """
    Resolve a property option to its canonical form.

    Matching is case-insensitive and accepts the aliases in `PROPERTY_ALIASES`. For example, `"hl_gap"` is resolved to
    `"HL-gap"`.

    Parameters
    ----------
    text : str
        Property or accepted alias.

    Returns
    -------
    str
        Canonical entry from `PROPERTY_NAMES`.

    Raises
    ------
    ValueError
        Raised when `text` is not a recognised property or alias.
    """
    resolved = PROPERTY_ALIASES.get(str(text).strip().lower())
    if resolved is None:
        raise ValueError(
            f"Unknown property {text!r}. Use 1 of "
            + ", ".join(PROPERTY_NAMES)
            + " (hl_gap, hlgap, hl, band-gap, band_gap, hof and heat_of_formation are also accepted)."
        )
    return resolved


def resolve_high_fidelity_method(text: str) -> str:
    """
    Resolve a high-fidelity method option to its canonical form.

    Matching is case-insensitive and treats hyphens and underscores as equivalent.

    Parameters
    ----------
    text : str
        High-fidelity method or accepted alias.

    Returns
    -------
    str
        Canonical entry from `HIGH_FIDELITY_METHOD_NAMES`.

    Raises
    ------
    ValueError
        Raised when `text` is not a recognised method or alias.
    """
    resolved = HIGH_FIDELITY_METHOD_ALIASES.get(str(text).strip().lower())
    if resolved is None:
        raise ValueError(
            f"Unknown high fidelity method {text!r}. Use 1 of "
            + ", ".join(HIGH_FIDELITY_METHOD_NAMES)
            + "."
        )
    return resolved


def check_combination(
    theory: str,
    property_name: str,
    high_fidelity_method: str | None,
    high_fidelity_fold: int,
) -> None:
    """
    Validate a requested combination of theory, property, method, and fold.

    Parameters
    ----------
    theory : str
        Canonical level of theory.
    property_name : str
        Canonical target property.
    high_fidelity_method : str or None
        Canonical correction method for CAM-B3LYP. This must be None for a semiempirical theory.
    high_fidelity_fold : int
        Cross-validation fold from 1 to 5.

    Raises
    ------
    ValueError
        Raised for an unavailable property, an incompatible correction method, or a fold outside 1-5.
    """
    if theory == "CAM-B3LYP":
        if property_name not in HIGH_FIDELITY_PROPERTIES:
            raise ValueError(
                f"CAM-B3LYP models exist only for {' and '.join(HIGH_FIDELITY_PROPERTIES)}, "
                f"not {property_name}. Request PM7 or GFN2-xTB instead."
            )
        if not 1 <= int(high_fidelity_fold) <= 5:
            raise ValueError(
                f"--high-fidelity-fold is {high_fidelity_fold}. Expected folds to be 1 to 5."
            )
        return
    if high_fidelity_method is not None:
        raise ValueError(
            f"Requested level of theory requested is {theory}. --high-fidelity-method applies to CAM-B3LYP only."
        )
    if (theory, property_name) not in LOW_LEVEL_COLUMNS:
        raise ValueError(
            f"There is no {property_name} at the {theory} level. GFN2-xTB and PM7 have HOMO, LUMO and HL-gap. PM7 also "
            f"have the heat of formation."
        )


def to_grid_tensor(
    molecules: str | PAC | torch.Tensor | Sequence[str | PAC | torch.Tensor],
) -> torch.Tensor:
    """
    Convert molecular inputs into a batch of PAC grid tensors.

    A SMILES string is first converted to the RDKit canonical SMILES string without stereochemistry, the form of the
    SMILES strings in PAC5, and the PAC grid of a molecule therefore does not depend on how the input SMILES string is
    written.

    Parameters
    ----------
    molecules : str, pac.PAC, torch.Tensor, or sequence of these types
        One SMILES string, PAC object, grid tensor, batch tensor, or a sequence containing any mixture of these forms.

    Returns
    -------
    torch.Tensor of shape (batch_size, 13, grid_size, grid_size)
        Float32 PAC grids on the CPU in input order.

    Raises
    ------
    TypeError
        Raised when an input item has an unsupported type.
    ValueError
        Raised for empty input, failed SMILES conversion, or an incompatible tensor shape.
    """
    if isinstance(molecules, (str, PAC, torch.Tensor)):
        items = [molecules]
    elif isinstance(molecules, Sequence):
        items = list(molecules)
    else:
        raise TypeError(
            f"Cannot read {type(molecules).__name__} as a molecule. Pass a SMILES string, a pac.PAC, a grid tensor, "
            f"or a sequence of these types."
        )
    if not items:
        raise ValueError("No molecules were passed.")

    grids = []
    for position, item in enumerate(items):
        if isinstance(item, str):
            molecule = Chem.MolFromSmiles(item)
            if molecule is None:
                raise ValueError(
                    f"Item {position} ({item!r}) is not a valid SMILES string."
                )
            Chem.RemoveStereochemistry(molecule)
            pac = smiles_to_pac(
                Chem.MolToSmiles(molecule), max_attempts=MAX_EMBEDDING_ATTEMPTS
            )
            if pac is None:
                raise ValueError(
                    f"Item {position} ({item!r}) is outside the scope of the grid representation and has no encoding."
                )
            grid = pac.to_tensor()
        elif isinstance(item, PAC):
            grid = item.to_tensor()
        elif isinstance(item, torch.Tensor):
            grid = item
        else:
            raise TypeError(
                f"Item {position} is a {type(item).__name__}. Pass a SMILES string, a pac.PAC or a grid tensor."
            )
        grid = grid.float()
        if grid.dim() == 4:
            grids.extend(grid[index] for index in range(grid.shape[0]))
            continue
        if grid.dim() != 3:
            raise ValueError(
                f"Item {position} has shape {tuple(grid.shape)}. Expected (13, grid_size, grid_size)."
            )
        grids.append(grid)
    return torch.stack(grids)


def low_level_checkpoint_path(fold: int = LOW_LEVEL_FOLD) -> str:
    """
    Return the semiempirical model checkpoint path for 1 fold.

    Parameters
    ----------
    fold : int, default 1
        Cross-validation fold from 1 to 5.

    Returns
    -------
    str
        Expected checkpoint path. File existence is not checked.
    """
    directory = LOW_LEVEL_DIRECTORY_TEMPLATE.format(fold=fold)
    return os.path.join(MODELS_DIRECTORY, directory, "best.ckpt")


def high_fidelity_checkpoint_path(
    property_name: str, high_fidelity_method: str, fold: int
) -> str:
    """
    Return the CAM-B3LYP model checkpoint path for 1 method and fold.

    Parameters
    ----------
    property_name : str
        CAM-B3LYP target property.
    high_fidelity_method : str
        High-fidelity training method.
    fold : int
        Cross-validation fold from 1 to 5.

    Returns
    -------
    str
        Expected checkpoint path. File existence is not checked.
    """
    directory = (
        HIGH_FIDELITY_DIRECTORY_STEMS[property_name]
        + HIGH_FIDELITY_DIRECTORY_SUFFIXES[high_fidelity_method]
    )
    return os.path.join(MODELS_DIRECTORY, directory, f"fold_{fold}.pth")


def load_target_scaler(fold: int = LOW_LEVEL_FOLD) -> MultiTargetScaler:
    """
    Load or reconstruct the target scaler for 1 cross-validation fold.

    The scaler parameters are loaded from `target_scaler.json` when available. Otherwise, the mean and standard
    deviation of each target are recalculated from the training partition.

    Parameters
    ----------
    fold : int, default 1
        Cross-validation fold from 1 to 5.

    Returns
    -------
    benchmarks.scaler.MultiTargetScaler
        Fitted scaler for the properties in `LOW_LEVEL_PROPERTIES`.

    Raises
    ------
    FileNotFoundError
        Raised when neither the saved scaler nor the training partition is available.
    ValueError
        Raised when the saved scaler does not contain every required property.
    """
    directory = os.path.join(
        MODELS_DIRECTORY, LOW_LEVEL_DIRECTORY_TEMPLATE.format(fold=fold)
    )
    cache_path = os.path.join(directory, "target_scaler.json")
    scaler = MultiTargetScaler()

    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        if tuple(cached["properties"]) != LOW_LEVEL_PROPERTIES:
            raise ValueError(
                f"{cache_path} was generated for a different set of properties. Manually delete it for this script to"
                f"automatically regenerate it."
            )
        scaler.mean = np.asarray(cached["mean"], dtype=np.float64)
        scaler.std = np.asarray(cached["std"], dtype=np.float64)
        return scaler

    split_path = os.path.join(SPLIT_DIRECTORY, f"fold_{fold}_train.csv")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{split_path} is missing and {cache_path} has not been generated yet. The semiempirical model predicts "
            f"standardized values so 1 of the two is needed. Run `python -m dataset.split PAC5-1` from the repository "
            f"root to write the splits."
        )
    frame = pd.read_csv(split_path).dropna(subset=list(LOW_LEVEL_PROPERTIES))
    scaler.fit(frame[list(LOW_LEVEL_PROPERTIES)].to_numpy(dtype=np.float64))
    if scaler.mean is None or scaler.std is None:
        raise RuntimeError(
            "The target scaler does not have a mean and a standard deviation after the fit."
        )

    os.makedirs(directory, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "properties": list(LOW_LEVEL_PROPERTIES),
                "mean": scaler.mean.tolist(),
                "std": scaler.std.tolist(),
            },
            handle,
            indent=2,
        )
    return scaler


def load_low_level_model(
    device: torch.device, fold: int = LOW_LEVEL_FOLD
) -> GroupEquivariantCNN:
    """
    Initialise the semiempirical PAC G-CNN and load a trained checkpoint.

    Parameters
    ----------
    device : torch.device
        Device used for inference.
    fold : int, default 1
        Cross-validation fold from 1 to 5.

    Returns
    -------
    gcnn.base_model.GroupEquivariantCNN
        Model in evaluation mode, configured for `LOW_LEVEL_PROPERTIES`.

    Raises
    ------
    FileNotFoundError
        Raised when the requested checkpoint is unavailable.
    """
    path = low_level_checkpoint_path(fold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No semiempirical checkpoint at {path}.")
    model = GroupEquivariantCNN(
        property_types=list(LOW_LEVEL_PROPERTY_TYPES),
        kernel_size=LOW_LEVEL_KERNEL_SIZE,
        num_hidden=LOW_LEVEL_NUM_HIDDEN,
        hidden_channels=LOW_LEVEL_HIDDEN_CHANNELS,
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def load_high_fidelity_model(
    property_name: str, high_fidelity_method: str, fold: int, device: torch.device
) -> GCNNDirectPrediction | GCNNDeltaCorrection:
    """
    Initialise a CAM-B3LYP model and load its trained checkpoint.

    Parameters
    ----------
    property_name : str
        Target property, either `"HL-gap"` or `"heat-of-formation"`.
    high_fidelity_method : str
        High-fidelity training method.
    fold : int
        Cross-validation fold from 1 to 5.
    device : torch.device
        Device used for inference.

    Returns
    -------
    GCNNDirectPrediction or GCNNDeltaCorrection
        Model in evaluation mode.

    Raises
    ------
    FileNotFoundError
        Raised when the requested checkpoint is unavailable.
    """
    path = high_fidelity_checkpoint_path(property_name, high_fidelity_method, fold)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No CAM-B3LYP checkpoint at {path}.")
    extensive = property_name == "heat-of-formation"
    if high_fidelity_method == "DIRECT":
        model = GCNNDirectPrediction(extensive=extensive)
    else:
        model = GCNNDeltaCorrection(
            delta=high_fidelity_method in DELTA_METHODS, extensive=extensive
        )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def predict_low_level(
    grids: torch.Tensor,
    theory: str,
    property_name: str,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fold: int = LOW_LEVEL_FOLD,
    model: GroupEquivariantCNN | None = None,
    scaler: MultiTargetScaler | None = None,
) -> npt.NDArray[np.float64]:
    """
    Predict a PM7 or GFN2-xTB property for a batch of PAC grids.

    Parameters
    ----------
    grids : torch.Tensor of shape (batch_size, 13, grid_size, grid_size)
        Input PAC grids.
    theory : str
        Semiempirical theory, either `"PM7"` or `"GFN2-xTB"`.
    property_name : str
        Target property available at the selected level of theory.
    device : torch.device
        Device used for inference.
    batch_size : int, default 256
        Number of molecules processed in each forward pass.
    fold : int, default 1
        Cross-validation fold used when loading a model or scaler.
    model : gcnn.base_model.GroupEquivariantCNN or None, default None
        Optional preloaded low-fidelity model.
    scaler : benchmarks.scaler.MultiTargetScaler or None, default None
        Optional fitted target scaler.

    Returns
    -------
    numpy.ndarray of shape (batch_size,)
        Predicted values in the physical unit of `property_name`.
    """
    column = LOW_LEVEL_COLUMNS[(theory, property_name)]
    column_index = LOW_LEVEL_PROPERTIES.index(column)
    if model is None:
        model = load_low_level_model(device, fold)
    if scaler is None:
        scaler = load_target_scaler(fold)

    standardized = []
    with torch.no_grad():
        for start in range(0, grids.shape[0], batch_size):
            batch = grids[start : start + batch_size].to(device)
            standardized.append(model(batch).cpu().numpy())
    standardized = np.concatenate(standardized, axis=0)
    return scaler.inverse_transform(standardized)[:, column_index]


def predict_high_fidelity(
    grids: torch.Tensor,
    property_name: str,
    high_fidelity_method: str,
    fold: int,
    device: torch.device,
    low_level_values: Sequence[float] | npt.ArrayLike | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> npt.NDArray[np.float64]:
    """
    Predict a CAM-B3LYP property for a batch of PAC grids.

    Parameters
    ----------
    grids : torch.Tensor of shape (batch_size, 13, grid_size, grid_size)
        Input PAC grids.
    property_name : str
        Target property, either `"HL-gap"` or `"heat-of-formation"`.
    high_fidelity_method : str
        High-fidelity training method.
    fold : int
        Cross-validation fold from 1 to 5.
    device : torch.device
        Device used for inference.
    low_level_values : sequence of float, array_like, or None, default None
        PM7 property value for each molecule, used by the delta methods. The semiempirical PAC G-CNN predicts the values
        when this is None.
    batch_size : int, default 256
        Number of molecules processed in each forward pass.

    Returns
    -------
    numpy.ndarray of shape (batch_size,)
        CAM-B3LYP predictions in the physical unit of `property_name`.

    Raises
    ------
    ValueError
        Raised when the number of PM7 values differs from the batch size or PM7 values are supplied to a non-delta
        method.
    FileNotFoundError
        Raised when the requested checkpoint is unavailable.
    """
    model = load_high_fidelity_model(property_name, high_fidelity_method, fold, device)
    needs_low_level = high_fidelity_method in DELTA_METHODS

    if needs_low_level:
        if low_level_values is None:
            low_level_values = predict_low_level(
                grids, "PM7", property_name, device, batch_size=batch_size
            )
        low_level_values = np.asarray(low_level_values, dtype=np.float32).reshape(-1)
        if low_level_values.shape[0] != grids.shape[0]:
            raise ValueError(
                f"{low_level_values.shape[0]} PM7 values were given for {grids.shape[0]} molecules."
            )
    elif low_level_values is not None:
        raise ValueError(
            f"{high_fidelity_method} method does not require a PM7 value. Only " + ", ".join(sorted(DELTA_METHODS))
            + " require a PM7 value."
        )

    predictions = []
    with torch.no_grad():
        for start in range(0, grids.shape[0], batch_size):
            batch = grids[start : start + batch_size].to(device)
            if high_fidelity_method == "DIRECT":
                output = model(batch)
            elif low_level_values is not None:
                low_level_batch = torch.tensor(
                    low_level_values[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(1)
                output, _ = model(batch, low_level_batch)
            else:
                output, _ = model(batch, None)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions, axis=0).reshape(-1).astype(np.float64)


def predict_properties(
    molecules: str | PAC | torch.Tensor | Sequence[str | PAC | torch.Tensor],
    theory: str = "CAM-B3LYP",
    property_name: str = "HL-gap",
    high_fidelity_method: str | None = None,
    high_fidelity_fold: int = 1,
    low_level_values: Sequence[float] | npt.ArrayLike | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | str | None = None,
) -> npt.NDArray[np.float64]:
    """
    Predict a molecular property for 1 or more PACs.

    Parameters
    ----------
    molecules : str, pac.PAC, torch.Tensor, or sequence of these types
        One SMILES string, PAC object, grid tensor, or a sequence containing any mixture of these forms.
    theory : str, default 'CAM-B3LYP'
        Level of theory: `"PM7"`, `"GFN2-xTB"`, or `"CAM-B3LYP"`.
    property_name : str, default 'HL-gap'
        Target property: `"HOMO"`, `"LUMO"`, `"HL-gap"`, or `"heat-of-formation"`.
    high_fidelity_method : str or None, default None
        High-fidelity training method. `"FINETUNE"` is used when this is None.
    high_fidelity_fold : int, default 1
        Cross-validation fold from 1 to 5 for CAM-B3LYP models.
    low_level_values : sequence of float, array_like, or None, default None
        PM7 property value for each molecule, used by the delta methods. The semiempirical PAC G-CNN predicts the values
        when this is None.
    batch_size : int, default 256
        Number of molecules processed in each forward pass.
    device : torch.device, str, or None, default None
        Inference device. CUDA is selected when available if this is None.

    Returns
    -------
    numpy.ndarray of shape (num_molecules,)
        Predicted values in the physical unit of `property_name`.

    Raises
    ------
    ValueError
        Raised for an invalid option or a SMILES string that cannot be converted to a PAC grid.
    TypeError
        Raised when an input item has an unsupported type.
    FileNotFoundError
        Raised when a required checkpoint or target-scaler file is unavailable.
    """
    theory = resolve_theory(theory)
    property_name = resolve_property(property_name)
    if theory == "CAM-B3LYP" and high_fidelity_method is None:
        high_fidelity_method = "FINETUNE"
    if high_fidelity_method is not None:
        high_fidelity_method = resolve_high_fidelity_method(high_fidelity_method)
    check_combination(theory, property_name, high_fidelity_method, high_fidelity_fold)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    grids = to_grid_tensor(molecules)
    if theory == "CAM-B3LYP":
        if high_fidelity_method is None:
            raise ValueError("A CAM-B3LYP prediction requires a high-fidelity method.")
        return predict_high_fidelity(
            grids,
            property_name,
            high_fidelity_method,
            int(high_fidelity_fold),
            device,
            low_level_values=low_level_values,
            batch_size=batch_size,
        )
    if low_level_values is not None:
        raise ValueError(
            "low_level_values applies to the CAM-B3LYP delta methods only."
        )
    return predict_low_level(
        grids, theory, property_name, device, batch_size=batch_size
    )


app = typer.Typer(
    name="predict_properties",
    help="Predict a property of a polycyclic aromatic compound (PAC) from its SMILES.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command(no_args_is_help=True)
def predict_cli(
    smiles: Annotated[
        list[str],
        typer.Argument(
            metavar="SMILES...",
            help="One or more SMILES strings. Several are predicted in a batch.",
        ),
    ],
    theory: Annotated[
        str,
        typer.Option(
            "--theory",
            help="Level of theory: [bold cyan]PM7[/], [bold cyan]GFN2-xTB[/] (or [bold cyan]xtb[/]), [bold cyan]CAM-B3LYP[/] (or [bold cyan]camb3lyp[/]). Case-insensitive.",
        ),
    ] = "CAM-B3LYP",
    property_name: Annotated[
        str,
        typer.Option(
            "--property",
            help="HOMO, LUMO, HL-gap (or hl_gap, hlgap, hl, band-gap, band_gap), or heat-of-formation (or hof, heat_of_formation). Case-insensitive.",
        ),
    ] = "HL-gap",
    high_fidelity_method: Annotated[
        str,
        typer.Option(
            "--high-fidelity-method",
            help="CAM-B3LYP only: [bold cyan]DIRECT[/], [bold cyan]DELTA[/], [bold cyan]FINETUNE[/], [bold cyan]TRANSFER[/], [bold cyan]FINETUNE-DELTA[/] or [bold cyan]TRANSFER-DELTA[/]. Case-insensitive.",
        ),
    ] = "FINETUNE",
    high_fidelity_fold: Annotated[
        int,
        typer.Option(
            "--high-fidelity-fold",
            min=1,
            max=5,
            help="CAM-B3LYP only: cross-validation fold index to load.",
        ),
    ] = 1,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Number of molecules evaluated per forward pass.",
        ),
    ] = DEFAULT_BATCH_SIZE,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            show_default=False,
            help="Torch device (cuda or cpu). Defaults to CUDA when available.",
        ),
    ] = None,
) -> None:
    """
    Predict a property of a polycyclic aromatic compound from its SMILES.
    """
    resolved_theory = resolve_theory(theory)
    method_for_predict = (
        high_fidelity_method
        if resolved_theory == "CAM-B3LYP"
        else (high_fidelity_method if high_fidelity_method != "FINETUNE" else None)
    )
    try:
        values = predict_properties(
            smiles,
            theory=theory,
            property_name=property_name,
            high_fidelity_method=method_for_predict,
            high_fidelity_fold=high_fidelity_fold,
            batch_size=batch_size,
            device=device,
        )
    except (ValueError, TypeError, FileNotFoundError) as error:
        print(f"predict_properties: {error}", file=sys.stderr)
        raise typer.Exit(code=1)

    resolved_property = resolve_property(property_name)
    display_name = PROPERTY_DISPLAY_NAMES[resolved_property]
    unit = PROPERTY_UNITS[resolved_property]

    heading = f"Predicting {resolved_theory} {display_name} ({unit})"
    if resolved_theory == "CAM-B3LYP":
        method = resolve_high_fidelity_method(high_fidelity_method or "FINETUNE")
        heading += f" using {method}. Weights loaded from fold {high_fidelity_fold}"
    else:
        method = None
        heading += f". Weights loaded from fold {LOW_LEVEL_FOLD}"
    print(heading)
    if method in DELTA_METHODS:
        print(
            f"The PM7 {display_name} ({unit}) used by {method} is estimated by the PAC G-CNN at the PM7 level. Weights "
            f"loaded from fold {LOW_LEVEL_FOLD}"
        )

    for molecule_smiles, value in zip(smiles, values):
        print(f" + {molecule_smiles}: {value:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        app(args=argv, standalone_mode=False)
        return 0
    except typer.Exit as exc:
        return exc.exit_code
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0


if __name__ == "__main__":
    app()
