"""
Build the geometry cache of the 3D baselines on PAC5-4.

The cache contains the atomic numbers and the Cartesian coordinates of the PAC5 conformer of every molecule of the
5-fold split of PAC5-4, including the hydrogen atoms. The conformers are the ETKDG/OpenFF conformers in
``data/rdkit_openff_sdf``, which are also the initial geometries of the GFN2-xTB and PM7 calculations. The script writes
2 files:

``dataset/PAC5-4/geom_openff.npz``
    ``Z``, int8 of shape (num_atoms_total,): the atomic numbers of all molecules, concatenated. ``R``, float32 of shape
    (num_atoms_total, 3): the Cartesian coordinates in Å, in the same atom order. ``ptr``, int64 of shape (num_molecules
    + 1,): molecule ``m`` contains the atoms ``ptr[m]`` to ``ptr[m + 1] - 1``.
``dataset/PAC5-4/geom_openff_index.csv``
    The columns ``smiles``, ``pubchem_cid``, and ``row``, where ``row`` is the index ``m`` of the molecule in ``ptr``.

Usage:

    python prep_3d_geometry_cache.py
    python prep_3d_geometry_cache.py --num-workers 16
"""

import os
import sqlite3
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
from rdkit import Chem, rdBase

from const import DEFAULT_NUM_WORKERS, PAC5_4_CATEGORIES
from dataset.common import create_executor

rdBase.DisableLog("rdApp.*")

REPOSITORY_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SDF_DIRECTORY = os.path.join(REPOSITORY_DIRECTORY, "data", "rdkit_openff_sdf")
PAC5_DATABASE_PATH = os.path.join(REPOSITORY_DIRECTORY, "data", "pac5.db")
SPLIT_DIRECTORY = os.path.join(REPOSITORY_DIRECTORY, "dataset", "PAC5-4")

TARGETS = [
    "xtb_homo",
    "xtb_lumo",
    "xtb_hl_gap",
    "pm7_homo",
    "pm7_lumo",
    "pm7_hl_gap",
    "pm7_hof",
]
PARTITIONS = ("train", "val", "test")

# Atomic numbers and Cartesian coordinates of 1 molecule
Geometry = tuple[npt.NDArray[np.int8], npt.NDArray[np.float32]]


def collect_split_smiles(split_directory: str, num_folds: int = 5) -> dict[str, int]:
    """
    Collect the unique SMILES strings of the k-fold split in the order of their first appearance.

    The partition files are visited fold by fold, each fold in the order train, validation, and test, and a SMILES
    string receives the next row index at its first appearance. A row with a missing value in a column of `TARGETS` is
    skipped.

    Parameters
    ----------
    split_directory : str
        Folder with the partition files ``fold_{fold}_{partition}.csv``.
    num_folds : int, default 5
        Number of cross-validation folds.

    Returns
    -------
    dict of str to int
        The row index of each SMILES string, counted from 0.

    Raises
    ------
    FileNotFoundError
        If a partition file of the split does not exist.
    """
    row_by_smiles: dict[str, int] = {}
    for fold in range(1, num_folds + 1):
        for partition in PARTITIONS:
            path = os.path.join(split_directory, f"fold_{fold}_{partition}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            molecules = pd.read_csv(path, usecols=["smiles"] + TARGETS).dropna(
                subset=TARGETS
            )
            for smiles in molecules["smiles"].values:
                if smiles not in row_by_smiles:
                    row_by_smiles[smiles] = len(row_by_smiles)
    return row_by_smiles


def load_cid_by_smiles(pac5_database_path: str) -> dict[str, int]:
    """
    Load the PubChem CID of every closed-shell PAC5-4 molecule of the PAC5 database.

    The database and the partition files contain the same SMILES strings, and the strings are compared without
    canonicalization. For a SMILES string with more than 1 row, the first row in the scan order is used, and the number
    of such rows is printed.

    Parameters
    ----------
    pac5_database_path : str
        Path of ``pac5.db``.

    Returns
    -------
    dict of str to int
        The PubChem CID of each SMILES string with ``is_open_shell = 0`` and a category of `PAC5_4_CATEGORIES`.
    """
    connection = sqlite3.connect(pac5_database_path)
    cursor = connection.cursor()
    placeholders = ", ".join("?" * len(PAC5_4_CATEGORIES))
    cursor.execute(
        f"SELECT smiles, pubchem_cid FROM pac5 WHERE is_open_shell = 0 AND category IN ({placeholders})",
        PAC5_4_CATEGORIES,
    )
    cid_by_smiles: dict[str, int] = {}
    num_duplicates = 0
    num_rows = 0
    while True:
        chunk = cursor.fetchmany(50000)
        if not chunk:
            break
        for smiles, pubchem_cid in chunk:
            num_rows += 1
            if smiles in cid_by_smiles:
                num_duplicates += 1
                continue
            cid_by_smiles[smiles] = pubchem_cid
        print(f"Scanned rows {num_rows:,}", end="\r", flush=True)
    print(f"Scanned rows {num_rows:,}")
    connection.close()
    if num_duplicates:
        print(
            f"Duplicate SMILES strings in the database {num_duplicates:,} (the first PubChem CID is used)"
        )
    return cid_by_smiles


def load_sdf_geometry(pubchem_cid: int) -> Geometry | None:
    """
    Load the atomic numbers and the Cartesian coordinates of the PAC5 conformer of 1 molecule.

    The conformer is the first molecule of ``{pubchem_cid}.sdf`` in `SDF_DIRECTORY`, loaded with the hydrogen atoms and
    without sanitization. The function returns None when the file does not exist, when the first molecule cannot be
    parsed or does not have a conformer, and when the molecule does not have any atom.

    Parameters
    ----------
    pubchem_cid : int
        PubChem CID of the molecule.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray) or None
        The atomic numbers, int8 of shape (num_atoms,), and the coordinates in Å, float32 of shape (num_atoms, 3), in
        the atom order of the SDF file, or None.
    """
    path = os.path.join(SDF_DIRECTORY, f"{pubchem_cid}.sdf")
    if not os.path.exists(path):
        return None
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    molecule = next(iter(supplier), None)
    if molecule is None or molecule.GetNumConformers() == 0:
        return None
    atomic_numbers = np.array(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.int8
    )
    coordinates = molecule.GetConformer().GetPositions().astype(np.float32)
    if len(atomic_numbers) != len(coordinates) or len(atomic_numbers) == 0:
        return None
    return atomic_numbers, coordinates


def load_geometry_of_row(task: tuple[int, int]) -> tuple[int, Geometry | None]:
    """
    Load the PAC5 conformer of 1 molecule and return the conformer with the row index of the molecule.

    Parameters
    ----------
    task : tuple of (int, int)
        The row index and the PubChem CID of the molecule.

    Returns
    -------
    row : int
        The row index of `task`.
    geometry : tuple of (numpy.ndarray, numpy.ndarray) or None
        The return value of `load_sdf_geometry`.
    """
    row, pubchem_cid = task
    return row, load_sdf_geometry(pubchem_cid)


app = typer.Typer(add_completion=False)


@app.command(help=__doc__.strip().split("\n")[0])
def main(
    split_directory: Annotated[
        str,
        typer.Option(
            "--split-directory",
            "--split-dir",
            help="Folder with the partition files of the k-fold split. The default is dataset/PAC5-4 in the "
            "repository folder.",
            show_default=False,
        ),
    ] = SPLIT_DIRECTORY,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "--out",
            help="Path of the .npz cache. The default is geom_openff.npz in the folder of --split-directory.",
            show_default=False,
        ),
    ] = None,
    num_workers: Annotated[
        int,
        typer.Option(
            "--num-workers",
            "--workers",
            min=0,
            help="Number of worker processes.",
        ),
    ] = DEFAULT_NUM_WORKERS,
    num_folds: Annotated[
        int,
        typer.Option(
            "--num-folds", "--folds", help="Number of cross-validation folds."
        ),
    ] = 5,
) -> None:
    """
    Build the geometry cache and the index file, and print a summary.

    The molecules are the SMILES of the split with a PubChem CID in the PAC5 database, in the order of
    `collect_split_smiles`. Molecules without PAC5 conformer are excluded.

    Parameters
    ----------
    split_directory : str, default SPLIT_DIRECTORY
        Folder with the partition files ``fold_{fold}_{partition}.csv``.
    output : str or None, default None
        Path of the ``.npz`` cache. None means ``geom_openff.npz`` in `split_directory`. The index file has the path of
        the cache with ``_index.csv`` in place of ``.npz``.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the loading of the SDF files.
    num_folds : int, default 5
        Number of cross-validation folds.
    """
    cache_path = output or os.path.join(split_directory, "geom_openff.npz")
    index_path = os.path.splitext(cache_path)[0] + "_index.csv"

    print("Collecting SMILES of the k-fold split")
    row_by_smiles = collect_split_smiles(split_directory, num_folds)
    print(f"Unique molecules {len(row_by_smiles):,}")

    print("Mapping SMILES to PubChem CIDs")
    cid_by_smiles = load_cid_by_smiles(PAC5_DATABASE_PATH)
    rows: list[int] = []
    cids: list[int] = []
    ordered_smiles: list[str] = []
    num_without_cid = 0
    for smiles, row in sorted(row_by_smiles.items(), key=lambda item: item[1]):
        pubchem_cid = cid_by_smiles.get(smiles)
        if pubchem_cid is None:
            num_without_cid += 1
            continue
        rows.append(row)
        cids.append(pubchem_cid)
        ordered_smiles.append(smiles)
    if num_without_cid:
        print(
            f"SMILES strings without PubChem CID in database {num_without_cid:,}"
        )

    print(f"Loading {len(cids):,} SDF files (processes: {num_workers})")
    geometries: dict[int, Geometry] = {}
    with create_executor(num_workers) as executor:
        results = executor.map(
            load_geometry_of_row, list(zip(rows, cids)), chunksize=256
        )
        for index, (row, geometry) in enumerate(results):
            if geometry is not None:
                geometries[row] = geometry
            if (index + 1) % 25000 == 0:
                print(f"{index + 1:,}/{len(cids):,}")

    kept_rows = [row for row in rows if row in geometries]
    print(
        f"Geometries loaded {len(kept_rows):,}. Missing {len(rows) - len(kept_rows):,}"
    )

    atomic_numbers = np.concatenate([geometries[row][0] for row in kept_rows])
    coordinates = np.concatenate([geometries[row][1] for row in kept_rows])
    pointers = np.zeros(len(kept_rows) + 1, dtype=np.int64)
    np.cumsum([len(geometries[row][0]) for row in kept_rows], out=pointers[1:])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, Z=atomic_numbers, R=coordinates, ptr=pointers)

    smiles_by_row = dict(zip(rows, ordered_smiles))
    cid_by_row = dict(zip(rows, cids))
    pd.DataFrame(
        {
            "smiles": [smiles_by_row[row] for row in kept_rows],
            "pubchem_cid": [cid_by_row[row] for row in kept_rows],
            "row": list(range(len(kept_rows))),
        }
    ).to_csv(index_path, index=False)

    num_atoms = np.diff(pointers)
    print(f"Saved {cache_path} ({os.path.getsize(cache_path) / 1e6:.0f} MB)")
    print(f"Saved {index_path}")
    print(f"Molecules {len(pointers) - 1:,}")
    print(f"Atoms {len(atomic_numbers):,}")
    print(
        f"Atoms per molecule: min {num_atoms.min()}, mean {num_atoms.mean():.1f}, max {num_atoms.max()}"
    )
    print(
        f"Atomic numbers of elements {sorted(set(np.unique(atomic_numbers).tolist()))}"
    )


if __name__ == "__main__":
    app()
