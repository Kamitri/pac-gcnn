"""
Provide shared functions for model training and molecular analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cclib
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Mol, rdDetermineBonds
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from pac import PAC


class EarlyStopping:
    """
    Stop training after the validation loss ceases to improve.

    Model weights are saved whenever the validation loss improves by at least `min_delta`. Training stops after
    `patience` consecutive epochs without such an improvement.

    Parameters
    ----------
    save_path : str or None, default None
        Destination for the best model weights. Checkpoints are not created when this is None.
    patience : int, default 50
        Number of consecutive unimproved epochs permitted.
    min_delta : float, default 1e-5
        Minimum decrease in validation loss required to reset the counter.
    """

    def __init__(
        self,
        save_path: str | None = None,
        patience: int = 50,
        min_delta: float = 0.00001,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    @property
    def current_patience(self) -> int:
        return self.counter

    def reset(self) -> None:
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """
        Save a model state dictionary to `save_path`.

        Parameters
        ----------
        model : torch.nn.Module
            Model to checkpoint.
        """
        if self.save_path is None:
            return
        torch.save(model.state_dict(), self.save_path)


def get_fingerprint(
    mol: Chem.Mol, radius: int = 3, fp_size: int = 2048
) -> torch.Tensor:
    """
    Compute the Morgan count fingerprint of a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule to fingerprint.
    radius : int, default 3
        Morgan fingerprint radius.
    fp_size : int, default 2048
        Length of the count fingerprint.

    Returns
    -------
    torch.Tensor
        One-dimensional Morgan count fingerprint stored as float32.
    """
    fp_gen = GetMorganGenerator(radius=radius, fpSize=fp_size)
    fp = fp_gen.GetCountFingerprintAsNumPy(mol)
    return torch.from_numpy(fp).float()


def log_molecule_to_tensorboard(
    writer: SummaryWriter, tag: str, pac: PAC | None, step: int
) -> None:
    """
    Add a 2D picture of a PAC to TensorBoard.

    Parameters
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
        TensorBoard summary writer.
    tag : str
        Tag assigned to the image summary.
    pac : PAC or None
        PAC to depict. The function returns without writing when this is None.
    step : int
        Training step associated with the image.
    """
    if writer is None or pac is None:
        return
    try:
        img = MolToImage(pac.mol, size=(300, 300))
        arr = np.array(img)  # HWC uint8
        writer.add_image(tag, arr, step, dataformats="HWC")
    except Exception:
        pass


def add_indexes(mol: Mol) -> Mol:
    """
    Annotate each RDKit atom with molecular index.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to annotate, modified in place.

    Returns
    -------
    Chem.Mol
        Input molecule with atom-index annotations.
    """
    for i in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(i).SetProp("atomNote", str(i))
    return mol


def meets_condition(pac: PAC, condition: Any) -> float:
    """
    Calculate the fraction of structural constraints satisfied by a PAC.

    Seven possible property ranges are tested independently.

    Parameters
    ----------
    pac : PAC
        PAC grid to evaluate.
    condition : Any
        Object defining the accepted range for each structural property.

    Returns
    -------
    float
        Fraction of the seven constraints that are satisfied.
    """
    score = 7
    if condition.count_benzene is not None and not (
        condition.count_benzene[0]
        <= pac.count_6_membered_rings
        <= condition.count_benzene[1]
    ):
        score -= 1
    if condition.count_functional_groups is not None and not (
        condition.count_functional_groups[0]
        <= pac.sum_functional_groups
        <= condition.count_functional_groups[1]
    ):
        score -= 1
    if condition.count_sp3_carbon is not None and not (
        condition.count_sp3_carbon[0]
        <= pac.count_sp3_carbon
        <= condition.count_sp3_carbon[1]
    ):
        score -= 1
    if condition.count_nitrogen is not None and not (
        condition.count_nitrogen[0] <= pac.count_nitrogen <= condition.count_nitrogen[1]
    ):
        score -= 1
    if condition.count_oxygen is not None and not (
        condition.count_oxygen[0] <= pac.count_oxygen <= condition.count_oxygen[1]
    ):
        score -= 1
    if condition.count_sulfur is not None and not (
        condition.count_sulfur[0] <= pac.count_sulfur <= condition.count_sulfur[1]
    ):
        score -= 1
    normalized_score = score / 7.0
    return normalized_score


def calc_num_rings(smiles: str) -> int:
    """
    Count molecular rings from a SMILES string.

    Parameters
    ----------
    smiles : str
        Molecular structure to analyse.

    Returns
    -------
    int
        Number of rings.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetRingInfo().NumRings()


def calc_num_rings_from_mol(mol: Chem.Mol) -> int:
    """
    Count the rings in an RDKit molecule.

    Parameters
    ----------
    mol : Chem.Mol
        Molecule to analyse.

    Returns
    -------
    int
        Number of rings.
    """
    return mol.GetRingInfo().NumRings()


def test_equivariance(model: torch.nn.Module, pac: PAC) -> None:
    """
    Test numerical invariance under all 12 D6 transformations.

    The original PAC and its 11 non-identity rotations and reflections are evaluated separately. The maximum absolute
    output difference is printed for each orientation.

    Parameters
    ----------
    model : torch.nn.Module
        Molecular property model to evaluate.
    pac : PAC
        PAC grid used for the test.

    Raises
    ------
    AssertionError
        Raised when any transformed prediction differs by at least 1e-4.
    """
    model.eval()
    tensors = []
    tensors.append(pac.to_tensor().cuda().unsqueeze(0))
    # Rotations 1..5
    r = pac
    for i in range(1, 6):
        r = r.rotated()
        tensors.append(r.to_tensor().cuda().unsqueeze(0))
    # Rotation 6 (full 6) then reflections
    r = r.rotated()
    tensors.append(r.to_tensor().cuda().unsqueeze(0))  # 6th rotation
    s = pac.reflected()
    tensors.append(s.to_tensor().cuda().unsqueeze(0))
    for i in range(1, 6):
        s = s.rotated()
        tensors.append(s.to_tensor().cuda().unsqueeze(0))

    outs = [model(t).detach().cpu() for t in tensors]
    for i in range(1, len(outs)):
        diff = (outs[0] - outs[i]).abs().max().item()
        print(f"Group element {i}, max abs diff {diff:.3e}")
        assert diff < 1e-4, f"Equivariance violated at group element {i} diff {diff}"


def cclib_extract_3d_mol(path: str) -> Chem.Mol:
    """
    Reconstruct a 3D RDKit molecule from a Gaussian log file.

    Atomic numbers and coordinates are obtained through cclib. Molecular bonds are then inferred from the 3D geometry.

    Parameters
    ----------
    path : str
        Gaussian output file to parse.

    Returns
    -------
    Chem.Mol
        Reconstructed molecule with 1 3D conformer.
    """
    data = cclib.io.ccread(path)
    if data is None:
        raise ValueError(f"cclib failed to parse {path}")

    atom_coords = data.atomcoords[-1]
    atom_nos = data.atomnos
    xyz_lines = [f"{len(atom_nos)}", ""]
    pt = Chem.GetPeriodicTable()
    for i, atom_no in enumerate(atom_nos):
        symbol = pt.GetElementSymbol(int(atom_no))
        x, y, z = atom_coords[i]
        xyz_lines.append(f"{symbol} {x:>10.5f} {y:>10.5f} {z:>10.5f}")
    xyz_block = "\n".join(xyz_lines)
    raw_mol = Chem.MolFromXYZBlock(xyz_block)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineConnectivity(mol)
    return mol
