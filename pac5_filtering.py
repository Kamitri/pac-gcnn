"""
Apply stages 2 and 3 of the PAC5 filter cascade in Supplementary Table 2.

Stage 2 keeps the molecules representable by the PAC grid. Each SMILES string is converted with `pac.smiles_to_pac`,
and a molecule is out of scope when the conversion fails or when the SMILES string of the realigned PAC grid differs
from the canonical SMILES string of the input. The canonical SMILES string keeps stereochemistry and isotope labels,
which the PAC grid does not encode, and a molecule with either label is therefore out of scope. For every molecule in
scope, the serialized PAC grid, the atom, ring, and functional-group counts, and the 5 characteristics are written to
the database.

Stage 3 sets the `excluded` flag of every molecule in scope with `is_excluded`.

The script updates `data/pac5.db` in the form it has during the PubChem scoping stage, with the `in_scope`, `excluded`,
and `has_characteristics_1` to `has_characteristics_5` columns.
"""

import os
import queue
import sqlite3
import threading

from loguru import logger
from rdkit import Chem, rdBase
from tqdm import tqdm

from const import DEFAULT_NUM_WORKERS
from dataset.common import create_executor
from dataset.pac5_database import MAX_EMBEDDING_ATTEMPTS
from pac import smiles_to_pac
from pac.enums import FunctionalGroup

rdBase.DisableLog("rdApp.*")

DATABASE_PATH = "./data/pac5.db"
BATCH_SIZE = 50_000

MAX_TOTAL_RINGS = 12
MAX_ATOMS = 75
HETEROATOM_FRACTION_LIMIT = 0.15

BOND_PATTERNS = [
    # Single bond between 2 saturated atoms: aliphatic atom with 4 connections, N atom with 3 connections, or O/S atom
    # with 2 connections
    Chem.MolFromSmarts("[A&X4,#7&X3,#8&X2,#16&X2]-[A&X4,#7&X3,#8&X2,#16&X2]"),
    # N/O/S atom bonded to O/S atom
    Chem.MolFromSmarts("[#7,#8,#16][#8,#16]"),
]

STRUCTURE_PATTERNS = [
    # Cyclic N-oxide
    Chem.MolFromSmarts("[#7;+1;R1][#8;-1]"),
    # Ring N/O/S atom bonded to a ring O/S atom
    Chem.MolFromSmarts("[#7,#8,#16;R1][#8,#16;R1]"),
    # Fully aliphatic 5-membered ring
    Chem.MolFromSmarts("[A;r5]1~[A;r5]~[A;r5]~[A;r5]~[A;r5]1"),
    # Fully aliphatic 6-membered ring
    Chem.MolFromSmarts("[A;r6]1~[A;r6]~[A;r6]~[A;r6]~[A;r6]~[A;r6]1"),
    # Sulfur with a valence of 4 or 6
    Chem.MolFromSmarts("[v4,v6;#16]"),
    # Chain of 3 saturated ring atoms
    Chem.MolFromSmarts(
        "[A&X4,#7&X3,#8&X2,#16&X2;R1]-[A&X4,#7&X3,#8&X2,#16&X2;R1]-[A&X4,#7&X3,#8&X2,#16&X2;R1]"
    ),
    # 6-membered ring with a substituent on every ring atom
    Chem.MolFromSmarts(
        "[$([r6;R1]-[*])]1[$([r6;R1]-[*])][$([r6;R1]-[*])][$([r6;R1]-[*])][$([r6;R1]-[*])]"
        "[$([r6;R1]-[*])]1"
    ),
    # 5-membered ring with a substituent on every ring atom
    Chem.MolFromSmarts(
        "[$([r5;R1]-[*])]1[$([r5;R1]-[*])][$([r5;R1]-[*])][$([r5;R1]-[*])][$([r5;R1]-[*])]1"
    ),
]

FUNCTIONAL_GROUP_COLUMNS = {
    "num_fg_nitro": FunctionalGroup.NITRO,
    "num_fg_cyanide": FunctionalGroup.CYANIDE,
    "num_fg_dimethylamino": FunctionalGroup.DIMETHYLAMINO,
    "num_fg_methyl": FunctionalGroup.METHYL,
    "num_fg_ethyl": FunctionalGroup.ETHYL,
    "num_fg_fluoride": FunctionalGroup.FLUORIDE,
    "num_fg_chloride": FunctionalGroup.CHLORIDE,
    "num_fg_carbonyl": FunctionalGroup.CARBONYL,
    "num_fg_hydrogen": FunctionalGroup.HYDROGEN,
}

UPDATE_COLUMNS = [
    "in_scope",
    "excluded",
    "serialization",
    "num_atoms",
    "num_total_rings",
    "num_6_membered_rings",
    "num_5_membered_rings",
    "num_heterocyclic_nitrogen",
    "num_heterocyclic_oxygen",
    "num_heterocyclic_sulfur",
    "num_sp3_carbon",
    *FUNCTIONAL_GROUP_COLUMNS,
    "has_characteristics_1",
    "has_characteristics_2",
    "has_characteristics_3",
    "has_characteristics_4",
    "has_characteristics_5",
]


def is_excluded(
    molecule: Chem.Mol, num_atoms: int, num_total_rings: int, num_heteroatoms: int
) -> bool:
    """
    Apply the stage 3 filters to 1 molecule in scope.

    A molecule is excluded when it has more than `MAX_TOTAL_RINGS` rings or more than `MAX_ATOMS` heavy atoms, when the
    fraction of N, O, and S atoms in the PAC grid framework is `HETEROATOM_FRACTION_LIMIT` or higher, or when it matches
    1 of the patterns in `BOND_PATTERNS` or `STRUCTURE_PATTERNS`.

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
        Molecule parsed from the SMILES string in the database.
    num_atoms : int
        Number of heavy atoms.
    num_total_rings : int
        Number of rings in the PAC grid.
    num_heteroatoms : int
        Number of N, O, and S atoms in the PAC grid framework. Atoms of functional groups are not counted.

    Returns
    -------
    bool
        True when the molecule is excluded.
    """
    if num_total_rings > MAX_TOTAL_RINGS or num_atoms > MAX_ATOMS:
        return True
    if num_atoms > 0 and num_heteroatoms / num_atoms >= HETEROATOM_FRACTION_LIMIT:
        return True
    if any(molecule.HasSubstructMatch(pattern) for pattern in BOND_PATTERNS):
        return True
    return any(molecule.HasSubstructMatch(pattern) for pattern in STRUCTURE_PATTERNS)


def process_molecule(row: tuple[int, str]) -> tuple:
    """
    Calculate the scope, the PAC grid, the counts, and the exclusion flag of 1 molecule.

    Parameters
    ----------
    row : tuple
        PubChem CID and SMILES string.

    Returns
    -------
    tuple
        Values in the order of `UPDATE_COLUMNS`, followed by the PubChem CID. A molecule out of scope has `in_scope` 0,
        `excluded` 1, and None in the other columns.
    """
    pubchem_cid, smiles = row
    out_of_scope = (0, 1) + (None,) * (len(UPDATE_COLUMNS) - 2) + (pubchem_cid,)
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return out_of_scope

        pac = smiles_to_pac(smiles, max_attempts=MAX_EMBEDDING_ATTEMPTS)
        if pac.smiles != Chem.MolToSmiles(molecule):
            return out_of_scope

        column_values = {
            "in_scope": 1,
            "num_atoms": molecule.GetNumAtoms(),
            "num_total_rings": pac.count_total_rings,
            "num_6_membered_rings": pac.count_6_membered_rings,
            "num_5_membered_rings": pac.count_5_membered_rings,
            "num_heterocyclic_nitrogen": pac.count_nitrogen,
            "num_heterocyclic_oxygen": pac.count_oxygen,
            "num_heterocyclic_sulfur": pac.count_sulfur,
            "num_sp3_carbon": pac.count_sp3_carbon,
        }
        functional_group_counts = pac.count_functional_groups
        for column, functional_group in FUNCTIONAL_GROUP_COLUMNS.items():
            column_values[column] = functional_group_counts[functional_group]
        characteristics = pac.has_category_x_characteristics
        for position in range(1, 6):
            column_values[f"has_characteristics_{position}"] = int(
                characteristics.get(position, False)
            )

        num_heteroatoms = (
            column_values["num_heterocyclic_nitrogen"]
            + column_values["num_heterocyclic_oxygen"]
            + column_values["num_heterocyclic_sulfur"]
        )
        column_values["excluded"] = int(
            is_excluded(
                molecule,
                column_values["num_atoms"],
                column_values["num_total_rings"],
                num_heteroatoms,
            )
        )
        update_values: dict[str, int | str] = {
            **column_values,
            "serialization": pac.serialize(),
        }
        return tuple(update_values[column] for column in UPDATE_COLUMNS) + (
            pubchem_cid,
        )
    except Exception:
        return out_of_scope


def write_batches(write_queue: queue.Queue) -> None:
    """
    Write result batches to the database from a background thread.

    All database writes are made from this thread so the worker processes do not compete for the SQLite lock.

    Parameters
    ----------
    write_queue : queue.Queue
        Lists of results from `process_molecule`, followed by None.
    """
    assignments = ", ".join(f"{column} = ?" for column in UPDATE_COLUMNS)
    statement = f"UPDATE pac5 SET {assignments} WHERE pubchem_cid = ?"

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA journal_mode = WAL;")
        connection.execute("PRAGMA synchronous = NORMAL;")
        connection.execute("PRAGMA temp_store = MEMORY;")
        cursor = connection.cursor()

        while True:
            batch = write_queue.get()
            if batch is None:
                break
            cursor.executemany(statement, batch)
            connection.commit()
            write_queue.task_done()


def main() -> None:
    """
    Apply stages 2 and 3 to every molecule in scope and write the results.
    """
    if not os.path.exists(DATABASE_PATH):
        logger.error(f"The database does not exist: {DATABASE_PATH}")
        return

    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            "SELECT pubchem_cid, smiles FROM pac5 WHERE in_scope = 1"
        ).fetchall()

    if not rows:
        logger.warning("The database does not contain a molecule in scope.")
        return

    logger.info(
        f"Processing {len(rows)} molecules (worker processes: {DEFAULT_NUM_WORKERS})."
    )

    write_queue = queue.Queue(maxsize=5)
    writer = threading.Thread(target=write_batches, args=(write_queue,), daemon=True)
    writer.start()

    results = []
    with create_executor(DEFAULT_NUM_WORKERS) as executor:
        for result in tqdm(
            executor.map(process_molecule, rows, chunksize=2500),
            total=len(rows),
            desc="Filtering molecules",
        ):
            results.append(result)
            if len(results) >= BATCH_SIZE:
                write_queue.put(results)
                results = []

    if results:
        write_queue.put(results)
    write_queue.put(None)
    writer.join()
    logger.success("The database update is complete.")


if __name__ == "__main__":
    main()
