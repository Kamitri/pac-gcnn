"""
Load the PubChemQC dataset.

The dataset is the table ``pubchemqc`` of ``data/pubchemqc.db``. Each molecule is a closed-shell PAC5-4 molecule, with
the PubChemQC B3LYP/6-31G*//PM6 labels and the regenerated B3LYP/6-31G*//PM6 labels from the PAC5 conformer.
"""

import sqlite3
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from const import DEFAULT_NUM_WORKERS
from dataset.common import DATA_PUBCHEMQC_SQLITE, create_executor
from dataset.pac_matrix import process_pac_row
from pac import PAC

QUALITY_COLUMNS = (
    "pubchemqc_connectivity_match",
    "pubchemqc_formula_match",
    "pm6_connectivity_match",
)


class PubChemQCDataset(Dataset):
    """
    Load the PAC grids and the target values of the PubChemQC database.

    The dataset contains the molecules of the table ``pubchemqc`` with a serialised PAC grid and a value in
    ``pubchemqc_hl_gap`` and in ``b3lyppm6_hl_gap``, in the order of ``pubchem_cid``. Each item is the PAC grid and the
    target values as float32.

    Parameters
    ----------
    properties : list of str
        Database columns of the target values, in the order of the targets of each item, for example
        ``["pubchemqc_homo", "pubchemqc_lumo", "pubchemqc_hl_gap"]``.
    realign : bool, default True
        Realign each PAC grid when True.
    require_quality_flags : bool, default False
        Keep only the molecules with the value 1 in every column of `QUALITY_COLUMNS` when True: the PubChemQC PM6
        geometry keeps the PubChem connectivity, the PubChemQC record has the atoms of the regenerated calculation, and
        the regenerated PM6 geometry keeps the bonds of the PAC5 conformer.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids.

    Attributes
    ----------
    pubchem_cids : list of int
        PubChem CID of each molecule.
    pacs : list of PAC
        Reconstructed PAC grids in the order of ``pubchem_cid``.
    smiles : list of str
        SMILES string of each molecule.
    targets : list of tuple, numpy.ndarray, or torch.Tensor
        Target values of each molecule. The constructor creates a list of tuples. A caller can replace the list with an
        array or a tensor of shape (num_molecules, num_properties), for example with standardised values.
    grids : list of torch.Tensor of shape (13, grid_size, grid_size)
        PAC grids as float32.
    """

    def __init__(
        self,
        properties: list[str],
        realign: bool = True,
        require_quality_flags: bool = False,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        conditions = [
            "serialization IS NOT NULL",
            "pubchemqc_hl_gap IS NOT NULL",
            "b3lyppm6_hl_gap IS NOT NULL",
        ]
        if require_quality_flags:
            conditions.extend(f"{column} = 1" for column in QUALITY_COLUMNS)

        connection = sqlite3.connect(DATA_PUBCHEMQC_SQLITE)
        rows = connection.execute(
            f"SELECT pubchem_cid, serialization, smiles, {', '.join(properties)} FROM pubchemqc "
            f"WHERE {' AND '.join(conditions)} ORDER BY pubchem_cid"
        ).fetchall()
        connection.close()

        worker = partial(process_pac_row, subset=None, realign=realign)
        with create_executor(num_workers) as executor:
            results = list(
                executor.map(worker, [row[1:] for row in rows], chunksize=100)
            )

        self.pubchem_cids: list[int] = [row[0] for row in rows]
        self.pacs: list[PAC] = [result[0] for result in results]
        self.smiles: list[str] = [result[1] for result in results]
        self.targets: list[tuple[Any, ...]] | np.ndarray | torch.Tensor = [
            result[2] for result in results
        ]
        self.grids = [pac.to_tensor().to(dtype=torch.float32) for pac in self.pacs]

    def __len__(self) -> int:
        return len(self.pacs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.as_tensor(self.targets[index], dtype=torch.float32)
        return self.grids[index], target
