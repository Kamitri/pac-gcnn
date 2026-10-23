"""
Morgan fingerprints for the MLP baseline.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import Dataset

from const import DEFAULT_NUM_WORKERS
from dataset.common import create_executor


def process_fingerprint(smiles: str) -> npt.NDArray[np.float32] | None:
    """
    Compute the 2048-bit Morgan fingerprint of radius 2 of a SMILES string.

    The bits are computed with the RDKit function `GetMorganFingerprintAsBitVect`.

    Parameters
    ----------
    smiles : str
        Molecular structure to fingerprint.

    Returns
    -------
    numpy.ndarray of shape (2048,) or None
        Fingerprint bits as float32, or None when RDKit cannot parse the SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.zeros((2048,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_array)
    return fp_array


class PAC5FingerprintDataset(Dataset):
    """
    Store the Morgan fingerprints and the target values of a partition in memory.

    Rows with a missing value in `properties` and SMILES strings that RDKit cannot parse are excluded. Each item is the
    fingerprint and the target values as float32. Indexing with a list of positions returns the fingerprints and the
    target values of these molecules as 2-dimensional tensors.

    Parameters
    ----------
    csv_path : str
        Partition CSV file with the column ``smiles`` and the target columns.
    properties : list of str
        Columns of the target values, in the order of the targets of each item.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the fingerprint calculation.

    Attributes
    ----------
    x : torch.Tensor of shape (num_molecules, 2048)
        Fingerprint bits as float32.
    y : torch.Tensor of shape (num_molecules, num_targets)
        Target values as float32, in the original units.
    """

    def __init__(
        self,
        csv_path: str,
        properties: list[str],
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        df = pd.read_csv(csv_path).dropna(subset=properties)
        smiles_list = df["smiles"].tolist()
        targets_array = df[properties].values.astype(np.float32)

        with create_executor(num_workers) as executor:
            results = list(
                executor.map(process_fingerprint, smiles_list, chunksize=100)
            )

        valid_x = []
        valid_y = []
        for fp, target in zip(results, targets_array):
            if fp is not None:
                valid_x.append(fp)
                valid_y.append(target)

        self.x = torch.tensor(np.array(valid_x), dtype=torch.float32)
        self.y = torch.tensor(np.array(valid_y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
