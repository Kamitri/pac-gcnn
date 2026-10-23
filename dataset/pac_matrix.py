"""
Load 13-channel PAC grids and their target values from the PAC5 database.
"""

import sqlite3
from collections.abc import Callable, Collection
from functools import partial
from typing import Any, overload

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from const import DEFAULT_NUM_WORKERS
from dataset.common import DATA_PAC5_SQLITE, create_executor
from dataset.pac5_database import create_pac5_database_if_missing
from pac import PAC


@overload
def process_pac_row(
    row: tuple[Any, ...], subset: None, realign: bool
) -> tuple[PAC, str, tuple[Any, ...]]: ...


@overload
def process_pac_row(
    row: tuple[Any, ...], subset: Collection[str], realign: bool
) -> tuple[PAC, str, tuple[Any, ...]] | None: ...


def process_pac_row(
    row: tuple[Any, ...], subset: Collection[str] | None, realign: bool
) -> tuple[PAC, str, tuple[Any, ...]] | None:
    """
    Reconstruct the PAC grid, the SMILES string, and the target values of 1 database row.

    Parameters
    ----------
    row : tuple
        ``(serialization, smiles, *targets)`` from a database query.
    subset : collection of str or None
        Categories of the returned molecules, for example ``{'10000', '11000'}``, or None for every category.
    realign : bool
        Realign the PAC grid when True.

    Returns
    -------
    tuple of (PAC, str, tuple) or None
        ``(pac, smiles, targets)``, or None when the category is outside `subset`.
    """
    serialization = row[0]
    smiles = row[1]
    targets = row[2:]

    pac = PAC.deserialize(serialization)
    if subset is not None and pac.category not in subset:
        return None

    if realign:
        pac.realign()

    return pac, smiles, targets


class PAC5Dataset(Dataset):
    """
    Load the PAC grids and the target values of PAC5 molecules from the database.

    The dataset contains the molecules of the PAC5 database with ``is_open_shell = 0`` and a value in every column of
    `properties`, in the order of ``pubchem_cid``. Each item is the PAC grid as float32 and the target values in the
    original units. The PAC5 database is created first when it does not exist.

    Parameters
    ----------
    properties : list of str
        Database columns of the target values, in the order of the targets of each item.
    fully_aromatic : bool, default False
        Include only the molecules with ``is_fully_aromatic = 1`` when True.
    polycyclic : bool, default False
        Include only the molecules with ``is_polycyclic = 1`` when True.
    subset : collection of str or None, default None
        Categories of the included molecules, or None for every category.
    realign : bool, default True
        Realign each PAC grid when True.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids.

    Attributes
    ----------
    pacs : list of PAC
        Reconstructed PAC grids in the order of ``pubchem_cid``.
    smiles : list of str
        SMILES string of each molecule.
    targets : list of tuple
        Target values of each molecule.
    grids : list of torch.Tensor of shape (13, grid_size, grid_size)
        PAC grids of type torch.uint8.
    """

    def __init__(
        self,
        properties: list[str],
        fully_aromatic: bool = False,
        polycyclic: bool = False,
        subset: Collection[str] | None = None,
        realign: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        self.pacs = []
        self.smiles = []
        self.targets = []

        create_pac5_database_if_missing()
        conn = sqlite3.connect(DATA_PAC5_SQLITE)
        cursor = conn.cursor()

        query_is_fully_aromatic = (
            " AND is_fully_aromatic = 1 " if fully_aromatic else ""
        )
        query_polycyclic = " AND is_polycyclic = 1" if polycyclic else ""

        props_str = ", ".join(properties)
        null_checks = " AND ".join([f"{p} IS NOT NULL" for p in properties])

        cursor.execute(
            f"SELECT serialization, smiles, {props_str} FROM pac5 "
            f"WHERE {null_checks} AND is_open_shell = 0"
            f"{query_is_fully_aromatic} {query_polycyclic} "
            f"ORDER BY pubchem_cid"
        )
        rows = cursor.fetchall()
        conn.close()

        worker: Callable[[tuple[Any, ...]], tuple[PAC, str, tuple[Any, ...]] | None] = (
            partial(process_pac_row, subset=subset, realign=realign)
        )
        with create_executor(num_workers) as executor:
            results = executor.map(worker, rows, chunksize=100)

        valid_results = [res for res in results if res is not None]

        self.pacs = [res[0] for res in valid_results]
        self.smiles = [res[1] for res in valid_results]
        self.targets = [res[2] for res in valid_results]

        self.grids = [pac.to_tensor().to(dtype=torch.uint8) for pac in self.pacs]

    def __len__(self) -> int:
        return len(self.pacs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.tensor(self.targets[index], dtype=torch.float32)
        return self.grids[index].float(), target

    @staticmethod
    def from_pacs(pacs: list[tuple[PAC, Any]]) -> PAC5Dataset:
        """
        Create a dataset from existing PAC grids without a database query.

        Indexing the returned dataset raises AttributeError because the dataset does not have `targets`.

        Parameters
        ----------
        pacs : list of tuple
            ``(PAC, value)`` pairs. Only the PAC grid of each pair is converted into a tensor.

        Returns
        -------
        PAC5Dataset
            Dataset with the input list as `pacs` and the PAC grids as float32 tensors in `grids`.
        """
        dataset = PAC5Dataset.__new__(PAC5Dataset)
        dataset.pacs = pacs
        dataset.grids = [pac.to_tensor().to(dtype=torch.float32) for pac, _ in pacs]
        return dataset


class PAC5MatrixDataset(Dataset):
    """
    Load the PAC grids and the target values of the molecules of 1 PAC5 partition.

    Each item is the PAC grid as float32 and the target values in the original units, in the order of the CSV file. The
    target values are loaded from the PAC5 database, not from the CSV file. Rows with a missing value in `properties`
    and SMILES strings without a database row are excluded. When several database rows have the same SMILES string, the
    row with the highest ``pubchem_cid`` is used. The PAC5 database is created first when it does not exist.

    Parameters
    ----------
    csv_path : str
        Partition CSV file with the column ``smiles`` and the target columns.
    properties : list of str
        Database columns of the target values, in the order of the targets of each item.
    realign : bool, default True
        Realign each PAC grid when True.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids.

    Attributes
    ----------
    smiles : list of str
        SMILES strings in the order of the CSV file.
    targets : torch.Tensor of shape (num_molecules, num_properties)
        Target values as float32.
    grids : torch.Tensor of shape (num_molecules, 13, grid_size, grid_size)
        PAC grids of type torch.uint8.
    """

    def __init__(
        self,
        csv_path: str,
        properties: list[str],
        realign: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        df = pd.read_csv(csv_path).dropna(subset=properties)
        smiles_list = df["smiles"].tolist()

        create_pac5_database_if_missing()
        conn = sqlite3.connect(DATA_PAC5_SQLITE)
        cursor = conn.cursor()

        props_str = ", ".join(properties)
        cursor.execute(
            f"SELECT serialization, smiles, {props_str} FROM pac5 ORDER BY pubchem_cid"
        )

        db_dict = {row[1]: row for row in cursor.fetchall()}
        conn.close()

        rows = [db_dict[s] for s in smiles_list if s in db_dict]

        worker = partial(process_pac_row, subset=None, realign=realign)
        with create_executor(num_workers) as executor:
            results = list(executor.map(worker, rows, chunksize=100))

        self.pacs = [(res[0], res[2]) for res in results]
        self.smiles = [res[1] for res in results]
        self.targets = torch.tensor([res[2] for res in results], dtype=torch.float32)

        self.pacs.reverse()
        grids = []
        while self.pacs:
            pac, _ = self.pacs.pop()
            grids.append(pac.to_tensor().to(torch.uint8))
        self.grids = torch.stack(grids)
        del self.pacs, grids

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.grids[index].float(), self.targets[index]


class PAC5BackToBasicsMatrixDataset(Dataset):
    """
    Load the PAC grids and the target values of the molecules of 1 PAC5-4 partition.

    Each item is the PAC grid and the target values as float32 tensors. The dataset contains the molecules of the
    partition CSV file with a database row in 1 of the 5 PAC5-4 categories, ordered by category, ``10001``, ``10010``,
    ``10100``, ``11000``, and ``10000``, and by the order of the CSV file within each category. The target values are
    loaded from the PAC5 database, not from the CSV file, and rows with a missing value in `properties` are excluded.
    The PAC5 database is created first when it does not exist.

    Parameters
    ----------
    csv_path : str
        Partition CSV file with the column ``smiles`` and the target columns.
    properties : list of str
        Database columns of the target values, in the order of the targets of each item.
    realign : bool, default True
        Realign each PAC grid when True.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids.

    Attributes
    ----------
    pacs : list of tuple
        ``(PAC, targets)`` pair of each molecule, in the order of the items.
    smiles : list of str
        SMILES strings in the order of the items.
    targets : torch.Tensor of shape (num_molecules, num_properties)
        Target values as float32.
    grids : list of torch.Tensor of shape (13, grid_size, grid_size)
        PAC grids as float32.
    """

    def __init__(
        self,
        csv_path: str,
        properties: list[str],
        realign: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        df = pd.read_csv(csv_path).dropna(subset=properties)
        smiles_list = df["smiles"].tolist()

        create_pac5_database_if_missing()
        conn = sqlite3.connect(DATA_PAC5_SQLITE)
        cursor = conn.cursor()

        props_str = ", ".join(properties)
        cursor.execute(
            f"SELECT serialization, smiles, {props_str} FROM pac5 "
            f"WHERE characteristics_1 = 1 AND characteristics_2 = 0 AND characteristics_3 = 0 AND "
            f"characteristics_4 = 0 AND characteristics_5 = 0 ORDER BY pubchem_cid"
        )

        db_dict = {row[1]: row for row in cursor.fetchall()}
        rows = [db_dict[s] for s in smiles_list if s in db_dict]

        worker = partial(process_pac_row, subset=None, realign=realign)
        with create_executor(num_workers) as executor:
            results = list(executor.map(worker, rows, chunksize=100))

        self.pacs = [(res[0], res[2]) for res in results]
        self.smiles = [res[1] for res in results]
        target_values = [res[2] for res in results]

        cursor.execute(
            f"SELECT serialization, smiles, {props_str} FROM pac5 "
            f"WHERE characteristics_1 = 1 AND characteristics_2 = 1 AND characteristics_3 = 0 AND "
            f"characteristics_4 = 0 AND characteristics_5 = 0 ORDER BY pubchem_cid"
        )

        db_dict = {row[1]: row for row in cursor.fetchall()}

        rows = [db_dict[s] for s in smiles_list if s in db_dict]

        worker = partial(process_pac_row, subset=None, realign=realign)
        with create_executor(num_workers) as executor:
            results = list(executor.map(worker, rows, chunksize=100))

        self.pacs = [(res[0], res[2]) for res in results] + self.pacs
        self.smiles = [res[1] for res in results] + self.smiles
        target_values = [res[2] for res in results] + target_values

        cursor.execute(
            f"SELECT serialization, smiles, {props_str} FROM pac5 "
            f"WHERE characteristics_1 = 1 AND characteristics_2 = 0 AND characteristics_3 = 1 AND "
            f"characteristics_4 = 0 AND characteristics_5 = 0 ORDER BY pubchem_cid"
        )

        db_dict = {row[1]: row for row in cursor.fetchall()}

        rows = [db_dict[s] for s in smiles_list if s in db_dict]

        worker = partial(process_pac_row, subset=None, realign=realign)
        with create_executor(num_workers) as executor:
            results = list(executor.map(worker, rows, chunksize=100))

        self.pacs = [(res[0], res[2]) for res in results] + self.pacs
        self.smiles = [res[1] for res in results] + self.smiles
        target_values = [res[2] for res in results] + target_values

        cursor.execute(
            f"SELECT serialization, smiles, {props_str} FROM pac5 "
            f"WHERE characteristics_1 = 1 AND characteristics_2 = 0 AND characteristics_3 = 0 AND "
            f"characteristics_4 = 1 AND characteristics_5 = 0 ORDER BY pubchem_cid"
        )

        db_dict = {row[1]: row for row in cursor.fetchall()}

        rows = [db_dict[s] for s in smiles_list if s in db_dict]

        worker = partial(process_pac_row, subset=None, realign=realign)
        with create_executor(num_workers) as executor:
            results = list(executor.map(worker, rows, chunksize=100))

        self.pacs = [(res[0], res[2]) for res in results] + self.pacs
        self.smiles = [res[1] for res in results] + self.smiles
        target_values = [res[2] for res in results] + target_values

        cursor.execute(
            f"SELECT serialization, smiles, {props_str} FROM pac5 "
            f"WHERE characteristics_1 = 1 AND characteristics_2 = 0 AND characteristics_3 = 0 AND "
            f"characteristics_4 = 0 AND characteristics_5 = 1 ORDER BY pubchem_cid"
        )

        db_dict = {row[1]: row for row in cursor.fetchall()}
        conn.close()

        rows = [db_dict[s] for s in smiles_list if s in db_dict]

        worker = partial(process_pac_row, subset=None, realign=realign)
        with create_executor(num_workers) as executor:
            results = list(executor.map(worker, rows, chunksize=100))

        self.pacs = [(res[0], res[2]) for res in results] + self.pacs
        self.smiles = [res[1] for res in results] + self.smiles
        self.targets = torch.tensor(
            [res[2] for res in results] + target_values, dtype=torch.float32
        )

        self.grids = [pac.to_tensor().to(dtype=torch.float32) for pac, _ in self.pacs]

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.grids[index], self.targets[index]


def parse_hof_row(
    row: tuple[Any, ...],
) -> tuple[torch.Tensor, float, float, str, str, int] | None:
    """
    Reconstruct the PAC grid of 1 database row with a pair of heats of formation.

    Parameters
    ----------
    row : tuple
        ``(serialization, pm7_hof, high_fidelity_hof, smiles, num_total_rings)`` from a database query.

    Returns
    -------
    tuple of (torch.Tensor, float, float, str, str, int) or None
        ``(grid, pm7_hof, high_fidelity_hof, smiles, category, num_total_rings)``, with the grid as a float32 tensor of
        shape (13, grid_size, grid_size), or None when the reconstruction raises an exception.
    """
    serialization, low_hof, high_hof, smiles, ring_count = row
    try:
        pac = PAC.deserialize(serialization)

        grid = pac.to_tensor().to(dtype=torch.float32)
        level = pac.category

        return grid, low_hof, high_hof, smiles, level, ring_count
    except Exception:
        return None


class PAC5MatrixHoFDataset(Dataset):
    """
    Pair the PM7 heat of formation with a high-fidelity heat of formation for each PAC5 molecule.

    The dataset contains the molecules of the PAC5 database with ``is_open_shell = 0``, more than 1 ring, and a value in
    ``pm7_hof`` and in `target_hof_col`, in the order of ``pubchem_cid``. A molecule with a PAC grid that cannot be
    reconstructed is excluded. Each item is the PAC grid as float32, the PM7 and the high-fidelity heat of formation as
    float32 tensors of shape (1,), the SMILES string, and the category. The PAC5 database is created first when it does
    not exist.

    Parameters
    ----------
    target_hof_col : str, default 'nist_hof'
        Database column of the high-fidelity heat of formation.

    Attributes
    ----------
    paired_data : list of tuple
        Results of `parse_hof_row` for the rows of the query.
    """

    def __init__(self, target_hof_col: str = "nist_hof") -> None:
        print("Fetching paired HOF data...")

        create_pac5_database_if_missing()
        conn = sqlite3.connect(DATA_PAC5_SQLITE)
        cursor = conn.cursor()

        query = f"""
            SELECT serialization, pm7_hof, {target_hof_col}, smiles,
                   num_total_rings
            FROM pac5
            WHERE pm7_hof IS NOT NULL AND num_total_rings > 1 AND is_open_shell = 0
              AND {target_hof_col} IS NOT NULL
            ORDER BY pubchem_cid
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        self.paired_data = []
        for row in rows:
            res = parse_hof_row(row)
            if res is not None:
                self.paired_data.append(res)

        print(f"Matched {len(self.paired_data)} paired samples.")

    def __len__(self) -> int:
        return len(self.paired_data)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        grid, low_hof, high_hof, smiles, level, _ = self.paired_data[index]
        low_tensor = torch.tensor([low_hof], dtype=torch.float32)
        high_tensor = torch.tensor([high_hof], dtype=torch.float32)
        return grid, low_tensor, high_tensor, smiles, level


class PAC5MatrixHLGapTransferLearningDataset(Dataset):
    """
    Pair the PM7 HOMO-LUMO gap with the CAM-B3LYP HOMO-LUMO gap for each PAC5 molecule.

    The molecules are the molecules with a CAM-B3LYP and a PM7 HOMO-LUMO gap, loaded with `PAC5Dataset`, realigned and
    without the aromaticity and polycyclicity filters. Each item is the PAC grid as unsigned 8-bit integers, the PM7 and
    the CAM-B3LYP gap as float32 tensors of shape (1,), the SMILES string, and the stratification label
    ``f"{category}_{count_total_rings}"``.

    Attributes
    ----------
    dataset_high : PAC5Dataset
        Molecules with a CAM-B3LYP HOMO-LUMO gap.
    paired_data : list of tuple
        ``(grid, pm7_gap, camb3lyp_gap, smiles, label)`` of each paired molecule.
    """

    def __init__(self) -> None:
        print("Loading high-level theory (CAM-B3LYP) dataset...")
        self.dataset_high = PAC5Dataset(
            properties=["camb3lyp_hl_gap"],
            polycyclic=False,
            fully_aromatic=False,
        )

        print("Fetching PM7 values...")
        create_pac5_database_if_missing()
        conn = sqlite3.connect(DATA_PAC5_SQLITE)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT smiles, pm7_hl_gap FROM pac5 "
            "WHERE camb3lyp_hl_gap IS NOT NULL AND pm7_hl_gap IS NOT NULL AND is_open_shell = 0"
        )
        low_dict = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        self.paired_data = []

        for i in range(len(self.dataset_high.smiles)):
            smiles = self.dataset_high.smiles[i]

            if smiles in low_dict:
                grid = self.dataset_high.grids[i]
                pac = self.dataset_high.pacs[i]

                dataset_item = self.dataset_high[i]
                high_bg_raw = dataset_item[1]

                if hasattr(high_bg_raw, "item"):
                    high_bg = high_bg_raw.item()
                elif isinstance(high_bg_raw, (list, tuple, np.ndarray)):
                    high_bg = float(high_bg_raw[0])
                else:
                    high_bg = float(high_bg_raw)

                low_bg = low_dict[smiles]
                label = pac.category + f"_{pac.count_total_rings}"

                self.paired_data.append((grid, low_bg, high_bg, smiles, label))

        print(f"Matched {len(self.paired_data)} paired samples.")

    def __len__(self) -> int:
        return len(self.paired_data)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        grid, low_bg, high_bg, smiles, level = self.paired_data[index]

        low_tensor = torch.tensor([low_bg], dtype=torch.float32)
        high_tensor = torch.tensor([high_bg], dtype=torch.float32)

        return grid, low_tensor, high_tensor, smiles, level
