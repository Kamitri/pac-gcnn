"""
Load the COMPAS-1D benchmark as 13-channel PAC grids.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from const import DEFAULT_NUM_WORKERS
from dataset.common import create_executor
from pac import smiles_to_pac


def process_compas_smiles(smiles: str) -> torch.Tensor:
    """
    Convert a COMPAS-1D SMILES string into a realigned PAC grid tensor.

    The SMILES string is converted with `pac.smiles_to_pac` and the default number of embedding attempts.

    Parameters
    ----------
    smiles : str
        Molecular structure to convert.

    Returns
    -------
    torch.Tensor of shape (13, grid_size, grid_size)
        Realigned PAC grid of type torch.uint8.

    Raises
    ------
    ValueError
        Raised by `pac.smiles_to_pac` when the SMILES string is invalid or the molecule cannot be embedded on the grid.
    """
    pac = smiles_to_pac(smiles)
    return pac.to_tensor().to(torch.uint8)


class COMPAS1DDataset(Dataset):
    """
    Store the PAC grids and the target values of the COMPAS-1D molecules in memory.

    Each item is the PAC grid as float32 and the target values in the original units, in the order of `df`.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path of a CSV file, or a data frame, with the column ``smiles`` and the target columns. A data frame is copied
        before the rows are removed.
    target_properties : list of str
        Columns of the target values, in the order of the targets of each item.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the conversion of the SMILES strings.

    Attributes
    ----------
    df : pandas.DataFrame
        Data without the rows with a missing target value, with a new index from 0.
    smiles : list of str
        SMILES strings in the order of `df`.
    targets : torch.Tensor of shape (num_molecules, num_targets)
        Target values as float32.
    grids : torch.Tensor of shape (num_molecules, 13, grid_size, grid_size)
        PAC grids of type torch.uint8.

    Raises
    ------
    ValueError
        Raised when a SMILES string cannot be converted into a PAC grid.
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        target_properties: list[str],
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()

        self.df = self.df.dropna(subset=target_properties).reset_index(drop=True)

        self.smiles = self.df["smiles"].tolist()
        self.targets = torch.tensor(
            self.df[target_properties].values, dtype=torch.float32
        )

        print(f"Generating PAC representations for {len(self.smiles)} molecules...")

        with create_executor(num_workers) as executor:
            grids = list(
                executor.map(process_compas_smiles, self.smiles, chunksize=100)
            )

        self.grids = torch.stack(grids)

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.grids[index].float(), self.targets[index]
