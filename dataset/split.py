"""
Create the partition files of the PAC5 datasets.

`generate_kfold_splits` writes the stratified cross-validation partitions of a selection of PAC5 molecules, and
`generate_single_split` writes 1 stratified train, validation, and test partition. `create_partition_set` creates 1 of
the partition sets below. Each set contains the closed-shell molecules with a value in every column of `PROPERTIES`,
with a further selection for some sets:

- ``PAC5-1``: 5 folds of these molecules.
- ``PAC5-2``: 5 folds of the fully aromatic molecules.
- ``PAC5-3``: 5 folds of the fully aromatic polycyclic molecules.
- ``PAC5-4``: 5 folds of the molecules of a category in `const.PAC5_4_CATEGORIES`.
- ``PAC5-1-no_high_fidelity``: 1 partition of the molecules without a value in ``camb3lyp_hl_gap`` or ``nist_hof``,
  which excludes the molecules of `dataset.pac_matrix.PAC5MatrixHLGapTransferLearningDataset` and
  `dataset.pac_matrix.PAC5MatrixHoFDataset`.
- ``PAC5-1-no_high_fidelity_hof``: 1 partition of the molecules without a value in ``nist_hof``, which excludes the
  molecules of `dataset.pac_matrix.PAC5MatrixHoFDataset` with the default column ``nist_hof``.
- ``PAC5-1-no_high_fidelity_hl_gap``: 1 partition of the molecules without a value in ``camb3lyp_hl_gap``.

Examples
--------
Create PAC5-4 and PAC5-1-no_high_fidelity in the directories of the same names next to this module::

    python -m dataset.split PAC5-4 PAC5-1-no_high_fidelity
"""

import os
import sqlite3
from collections.abc import Sequence
from functools import partial
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
from sklearn.model_selection import StratifiedKFold, train_test_split

from const import DEFAULT_NUM_WORKERS, PAC5_4_CATEGORIES
from dataset.common import DATA_PAC5_SQLITE, create_executor
from dataset.pac5_database import create_pac5_database_if_missing
from dataset.pac_matrix import process_pac_row

DATASET_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# Seed of published partitions
RANDOM_STATE = 42
PROPERTIES = [
    "xtb_homo",
    "xtb_lumo",
    "xtb_hl_gap",
    "pm7_homo",
    "pm7_lumo",
    "pm7_hl_gap",
    "pm7_hof",
]
KFOLD_SETS: dict[str, dict[str, Any]] = {
    "PAC5-1": {},
    "PAC5-2": {"fully_aromatic": True},
    "PAC5-3": {"fully_aromatic": True, "polycyclic": True},
    "PAC5-4": {"categories": PAC5_4_CATEGORIES, "merge_rare_labels": True},
}
SINGLE_SPLIT_SETS: dict[str, dict[str, Any]] = {
    "PAC5-1-no_high_fidelity": {"excluded_columns": ("camb3lyp_hl_gap", "nist_hof")},
    "PAC5-1-no_high_fidelity_hof": {"excluded_columns": ("nist_hof",)},
    "PAC5-1-no_high_fidelity_hl_gap": {"excluded_columns": ("camb3lyp_hl_gap",)},
}


def load_partition_molecules(
    properties: Sequence[str],
    fully_aromatic: bool = False,
    polycyclic: bool = False,
    categories: Sequence[str] | None = None,
    excluded_columns: Sequence[str] = (),
    num_workers: int = DEFAULT_NUM_WORKERS,
    batch_size: int = 20000,
) -> tuple[pd.DataFrame, npt.NDArray[np.str_]]:
    """
    Load the molecules of a partition set and the stratification label of each molecule.

    The molecules are the rows of the PAC5 database with ``source = 'pubchem'``, ``is_open_shell = 0``, and a value in
    every column of `properties`, in the order of ``pubchem_cid``, and the other parameters restrict the selection
    further. The stratification label of a molecule is ``f"{category}_{num_total_rings}"``, where the category is
    `pac.pac.PAC.category` of the PAC grid reconstructed from the column ``serialization``. The PAC5 database is created
    first when it does not exist.

    Parameters
    ----------
    properties : sequence of str
        Property columns of the database, in the order of the columns of the partition files.
    fully_aromatic : bool, default False
        Include only the molecules with ``is_fully_aromatic = 1`` when True.
    polycyclic : bool, default False
        Include only the molecules with ``is_polycyclic = 1`` when True.
    categories : sequence of str or None, default None
        Categories of the included molecules in the column ``category``, or None for every category.
    excluded_columns : sequence of str, default ()
        Columns of high-fidelity reference values. A molecule with a value in any of these columns is excluded.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids.
    batch_size : int, default 20000
        Number of database rows fetched and reconstructed at a time.

    Returns
    -------
    molecules : pandas.DataFrame
        The column ``smiles`` and 1 column for each property, with 1 row for each molecule.
    labels : numpy.ndarray of str
        Stratification label of each molecule, in the order of the rows of `molecules`.
    """
    conditions = [f"{column} IS NOT NULL" for column in properties]
    conditions.append("source = 'pubchem'")
    conditions.append("is_open_shell = 0")
    if fully_aromatic:
        conditions.append("is_fully_aromatic = 1")
    if polycyclic:
        conditions.append("is_polycyclic = 1")
    if categories is not None:
        category_list = ", ".join(f"'{category}'" for category in categories)
        conditions.append(f"category IN ({category_list})")
    conditions.extend(f"{column} IS NULL" for column in excluded_columns)
    query = (
        f"SELECT serialization, smiles, num_total_rings, {', '.join(properties)} FROM pac5 "
        f"WHERE {' AND '.join(conditions)} ORDER BY pubchem_cid"
    )

    create_pac5_database_if_missing()
    connection = sqlite3.connect(DATA_PAC5_SQLITE)
    cursor = connection.execute(query)
    worker = partial(process_pac_row, subset=None, realign=False)
    smiles_column: list[str] = []
    property_columns: dict[str, list[Any]] = {column: [] for column in properties}
    labels: list[str] = []
    with create_executor(num_workers) as executor:
        while rows := cursor.fetchmany(batch_size):
            # Worker input of process_pac_row: (serialization, smiles, *targets), without ring count
            worker_inputs = [(row[0], row[1], *row[3:]) for row in rows]
            ring_counts = [row[2] for row in rows]
            results = executor.map(worker, worker_inputs, chunksize=100)
            for (pac, smiles, targets), ring_count in zip(results, ring_counts):
                smiles_column.append(smiles)
                for column, value in zip(properties, targets):
                    property_columns[column].append(value)
                labels.append(f"{pac.category}_{ring_count}")
            print(f"Molecules loaded {len(smiles_column):,}")
    connection.close()

    molecules = pd.DataFrame({"smiles": smiles_column, **property_columns})
    return molecules, np.array(labels)


def merge_rare_stratification_labels(
    labels: npt.NDArray[np.str_], num_folds: int
) -> npt.NDArray[np.str_]:
    """
    Merge the stratification labels with fewer molecules than folds.

    Every label with fewer than `num_folds` molecules is replaced by ``rare``. When fewer than `num_folds` molecules
    have the label ``rare`` after the replacement, these molecules receive the most frequent original label instead,
    because `sklearn.model_selection.StratifiedKFold` requires at least `num_folds` molecules for each label.

    Parameters
    ----------
    labels : numpy.ndarray of str
        Stratification label of each molecule.
    num_folds : int
        Number of cross-validation folds.

    Returns
    -------
    numpy.ndarray of str
        Labels after the merge. The input array is returned when every label has at least `num_folds` molecules.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    rare_labels = list(unique_labels[counts < num_folds])
    if not rare_labels:
        return labels
    is_rare = np.isin(labels, rare_labels)
    print(
        f"Labels merged into rare {len(rare_labels)} ({int(is_rare.sum())} molecules)"
    )
    merged_labels = np.where(is_rare, "rare", labels)
    if 0 < int((merged_labels == "rare").sum()) < num_folds:
        most_frequent_label = unique_labels[counts.argmax()]
        merged_labels = np.where(
            merged_labels == "rare", most_frequent_label, merged_labels
        )
    return merged_labels


def write_partition(
    molecules: pd.DataFrame, indices: npt.NDArray[np.intp], path: str
) -> None:
    """
    Write the molecules at the given row positions to a CSV file without the index column.

    Parameters
    ----------
    molecules : pandas.DataFrame
        Molecules returned by `load_partition_molecules`.
    indices : numpy.ndarray of int
        Row positions of the molecules of the partition, in the order of the rows of the file.
    path : str
        Path of the CSV file.
    """
    molecules.iloc[indices].reset_index(drop=True).to_csv(path, index=False)


def generate_kfold_splits(
    properties: Sequence[str],
    output_directory: str,
    fully_aromatic: bool = False,
    polycyclic: bool = False,
    categories: Sequence[str] | None = None,
    merge_rare_labels: bool = False,
    num_folds: int = 5,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Write stratified train, validation, and test partitions for each cross-validation fold.

    `sklearn.model_selection.StratifiedKFold` with `num_folds` folds, shuffling, and ``random_state=42`` selects the
    test set of each fold from the molecules of `load_partition_molecules`, and a stratified split of the other
    molecules with ``test_size=0.125`` and ``random_state=42`` selects the validation set. With 5 folds, the train,
    validation, and test partitions have 70%, 10%, and 20% of the molecules. The files ``fold_<k>_train.csv``,
    ``fold_<k>_val.csv``, and ``fold_<k>_test.csv`` for k from 1 to `num_folds` are written to `output_directory`, with
    the column ``smiles`` and 1 column for each property. When a label has too few molecules for the stratified
    validation split of a fold, the validation set of that fold is selected at random without stratification.

    Parameters
    ----------
    properties : sequence of str
        Property columns of the database, in the order of the columns of the partition files.
    output_directory : str
        Directory of the partition files. The directory is created when it does not exist.
    fully_aromatic : bool, default False
        Include only the molecules with ``is_fully_aromatic = 1`` when True.
    polycyclic : bool, default False
        Include only the molecules with ``is_polycyclic = 1`` when True.
    categories : sequence of str or None, default None
        Categories of the included molecules in the column ``category``, or None for every category.
    merge_rare_labels : bool, default False
        Apply `merge_rare_stratification_labels` to the labels before the split when True.
    num_folds : int, default 5
        Number of cross-validation folds.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids, as in `load_partition_molecules`.
    """
    molecules, labels = load_partition_molecules(
        properties,
        fully_aromatic=fully_aromatic,
        polycyclic=polycyclic,
        categories=categories,
        num_workers=num_workers,
    )
    if merge_rare_labels:
        labels = merge_rare_stratification_labels(labels, num_folds)

    os.makedirs(output_directory, exist_ok=True)
    splitter = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE
    )
    for fold_index, (train_validation_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(molecules)), labels)
    ):
        try:
            train_indices, validation_indices = train_test_split(
                train_validation_indices,
                test_size=0.125,
                random_state=RANDOM_STATE,
                stratify=labels[train_validation_indices],
            )
        except ValueError:
            train_indices, validation_indices = train_test_split(
                train_validation_indices, test_size=0.125, random_state=RANDOM_STATE
            )

        fold = fold_index + 1
        write_partition(
            molecules,
            train_indices,
            os.path.join(output_directory, f"fold_{fold}_train.csv"),
        )
        write_partition(
            molecules,
            validation_indices,
            os.path.join(output_directory, f"fold_{fold}_val.csv"),
        )
        write_partition(
            molecules,
            test_indices,
            os.path.join(output_directory, f"fold_{fold}_test.csv"),
        )
        print(
            f"Fold {fold} molecules: train {len(train_indices)}, validation {len(validation_indices)}, "
            f"test {len(test_indices)}"
        )


def generate_single_split(
    properties: Sequence[str],
    output_directory: str,
    excluded_columns: Sequence[str] = (),
    fully_aromatic: bool = False,
    polycyclic: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Write 1 stratified train, validation, and test partition in the proportions 70:10:20.

    The test set is 20% of the molecules of `load_partition_molecules`, and the validation set is 12.5% of the rest,
    both from a stratified `sklearn.model_selection.train_test_split` with ``random_state=42``. The files ``train.csv``,
    ``val.csv``, and ``test.csv`` are written to `output_directory`, with the column ``smiles`` and 1 column for each
    property. When a label has too few molecules for a stratified split, the molecules of that split are selected at
    random without stratification.

    Parameters
    ----------
    properties : sequence of str
        Property columns of the database, in the order of the columns of the partition files.
    output_directory : str
        Directory of the partition files. The directory is created when it does not exist.
    excluded_columns : sequence of str, default ()
        Columns of high-fidelity reference values. A molecule with a value in any of these columns is excluded.
    fully_aromatic : bool, default False
        Include only the molecules with ``is_fully_aromatic = 1`` when True.
    polycyclic : bool, default False
        Include only the molecules with ``is_polycyclic = 1`` when True.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids, as in `load_partition_molecules`.
    """
    molecules, labels = load_partition_molecules(
        properties,
        fully_aromatic=fully_aromatic,
        polycyclic=polycyclic,
        excluded_columns=excluded_columns,
        num_workers=num_workers,
    )
    all_indices = np.arange(len(molecules))
    try:
        train_validation_indices, test_indices = train_test_split(
            all_indices,
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        train_validation_indices, test_indices = train_test_split(
            all_indices, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
        )
    try:
        train_indices, validation_indices = train_test_split(
            train_validation_indices,
            test_size=0.125,
            random_state=RANDOM_STATE,
            stratify=labels[train_validation_indices],
        )
    except ValueError:
        train_indices, validation_indices = train_test_split(
            train_validation_indices, test_size=0.125, random_state=RANDOM_STATE
        )

    os.makedirs(output_directory, exist_ok=True)
    write_partition(
        molecules, train_indices, os.path.join(output_directory, "train.csv")
    )
    write_partition(
        molecules, validation_indices, os.path.join(output_directory, "val.csv")
    )
    write_partition(molecules, test_indices, os.path.join(output_directory, "test.csv"))
    print(
        f"Partition molecules: train {len(train_indices)}, validation {len(validation_indices)}, "
        f"test {len(test_indices)}"
    )


def create_partition_set(
    name: str,
    output_directory: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Create 1 partition set of `KFOLD_SETS` or `SINGLE_SPLIT_SETS` with the properties in `PROPERTIES`.

    Parameters
    ----------
    name : str
        Key of `KFOLD_SETS` or `SINGLE_SPLIT_SETS`.
    output_directory : str or None, default None
        Directory of the partition files, or None for the directory `name` next to this module.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids, as in `load_partition_molecules`.

    Raises
    ------
    KeyError
        Raised when `name` is not a key of `KFOLD_SETS` or `SINGLE_SPLIT_SETS`.
    """
    if output_directory is None:
        output_directory = os.path.join(DATASET_DIRECTORY, name)
    if name in KFOLD_SETS:
        generate_kfold_splits(
            PROPERTIES, output_directory, num_workers=num_workers, **KFOLD_SETS[name]
        )
    elif name in SINGLE_SPLIT_SETS:
        generate_single_split(
            PROPERTIES,
            output_directory,
            num_workers=num_workers,
            **SINGLE_SPLIT_SETS[name],
        )
    else:
        raise KeyError(f"Unknown partition set {name!r}.")


app = typer.Typer(add_completion=False)


@app.command(help=__doc__.strip().split("\n")[0])
def main(
    names: Annotated[
        list[str],
        typer.Argument(
            help="Partition sets, keys of KFOLD_SETS or SINGLE_SPLIT_SETS.",
            show_default=False,
        ),
    ],
    num_workers: Annotated[
        int,
        typer.Option(
            min=0,
            help="Number of worker processes for the reconstruction of the PAC grids.",
        ),
    ] = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Create the partition sets given on the command line.

    Parameters
    ----------
    names : list of str
        Keys of `KFOLD_SETS` or `SINGLE_SPLIT_SETS`.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the reconstruction of the PAC grids, as in `load_partition_molecules`.

    Raises
    ------
    typer.BadParameter
        Raised when a name is not a key of `KFOLD_SETS` or `SINGLE_SPLIT_SETS`.
    """
    unknown_names = [
        name
        for name in names
        if name not in KFOLD_SETS and name not in SINGLE_SPLIT_SETS
    ]
    if unknown_names:
        raise typer.BadParameter(
            f"Unknown partition sets {unknown_names}. The partition sets are {[*KFOLD_SETS, *SINGLE_SPLIT_SETS]}."
        )
    for name in names:
        print(f"Partition set {name}")
        create_partition_set(name, num_workers=num_workers)


if __name__ == "__main__":
    app()
