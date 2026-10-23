"""
Create the PAC5 SQLite database from the PAC5 spreadsheet.

The spreadsheet ``data/pac5.xlsx`` has the molecular structures and the property values, and the database adds 3 kinds
of columns: the serialised PAC grid of each molecule, the number of rings in the RDKit ring information of the molecule,
``num_total_rings``, and the 5 digits of the spreadsheet column ``category`` as ``characteristics_1`` to
``characteristics_5``.
"""

import os
import sqlite3

import polars
from rdkit import Chem, rdBase
from tqdm import tqdm

from const import DEFAULT_NUM_WORKERS
from dataset.common import DATA_PAC5_EXCEL, DATA_PAC5_SQLITE, create_executor
from pac import smiles_to_pac

MAX_EMBEDDING_ATTEMPTS = 10_000  # Lower for more speed, but expect more failed attempts

SPREADSHEET_COLUMNS = {
    "pubchem_cid": polars.Int64,
    "smiles": polars.String,
    "category": polars.String,
    "is_polycyclic": polars.Int64,
    "is_fully_aromatic": polars.Int64,
    "is_open_shell": polars.Int64,
    "source": polars.String,
    "xtb_homo": polars.Float64,
    "xtb_lumo": polars.Float64,
    "xtb_hl_gap": polars.Float64,
    "pm7_homo": polars.Float64,
    "pm7_lumo": polars.Float64,
    "pm7_hl_gap": polars.Float64,
    "pm7_hof": polars.Float64,
    "camb3lyp_homo": polars.Float64,
    "camb3lyp_lumo": polars.Float64,
    "camb3lyp_hl_gap": polars.Float64,
    "nist_hof": polars.Float64,
}

PAC5_TABLE_COLUMNS = {
    "pubchem_cid": "INT",
    "smiles": "TEXT",
    "characteristics_1": "INT",
    "characteristics_2": "INT",
    "characteristics_3": "INT",
    "characteristics_4": "INT",
    "characteristics_5": "INT",
    "category": "TEXT",
    "serialization": "TEXT",
    "num_total_rings": "INT",
    "is_polycyclic": "INT",
    "is_fully_aromatic": "INT",
    "is_open_shell": "INT",
    "xtb_homo": "REAL",
    "xtb_lumo": "REAL",
    "xtb_hl_gap": "REAL",
    "pm7_homo": "REAL",
    "pm7_lumo": "REAL",
    "pm7_hl_gap": "REAL",
    "pm7_hof": "REAL",
    "nist_hof": "REAL",
    "camb3lyp_homo": "REAL",
    "camb3lyp_lumo": "REAL",
    "camb3lyp_hl_gap": "REAL",
    "source": "TEXT",
}


def smiles_to_serialized_pac(smiles: str) -> str | None:
    """
    Convert a SMILES string into a serialised, realigned PAC grid.

    The SMILES string is converted with `pac.smiles_to_pac` and `MAX_EMBEDDING_ATTEMPTS` embedding attempts in each pass
    of `pac.convert.nx_to_pac`. The PAC grid is accepted only when the molecule decoded from the grid has the InChIKey
    of the input molecule. The RDKit log messages are blocked during the conversion.

    Parameters
    ----------
    smiles : str
        Molecular structure to convert.

    Returns
    -------
    str or None
        Serialised PAC grid, or None when the conversion raises an exception or the InChIKeys differ.
    """
    with rdBase.BlockLogs():
        try:
            pac = smiles_to_pac(smiles, max_attempts=MAX_EMBEDDING_ATTEMPTS)
            if Chem.MolToInchiKey(pac.mol) != Chem.MolToInchiKey(
                Chem.MolFromSmiles(smiles)
            ):
                return None
            return pac.serialize()
        except Exception:
            return None


def create_pac5_database_if_missing(num_workers: int = DEFAULT_NUM_WORKERS) -> None:
    """
    Create the PAC5 SQLite database from the PAC5 spreadsheet when the database does not exist.

    The function returns immediately when `DATA_PAC5_SQLITE` is a nonempty file. Otherwise, each molecule of the sheet
    ``PAC5`` of `DATA_PAC5_EXCEL` is converted with `smiles_to_serialized_pac` and inserted into the table ``pac5`` with
    the spreadsheet columns and the added columns, and a molecule that cannot be converted is excluded. The
    database is written to ``<database path>.partial`` and renamed to the database path only after the table is
    complete.

    Parameters
    ----------
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes for the conversion of the SMILES strings.

    Raises
    ------
    FileNotFoundError
        Raised when neither the database nor the spreadsheet exists.
    """
    # sqlite3.connect creates file of 0 bytes when path does not exist so size is checked too
    if os.path.isfile(DATA_PAC5_SQLITE) and os.path.getsize(DATA_PAC5_SQLITE) > 0:
        return

    database_path = os.path.normpath(DATA_PAC5_SQLITE)
    spreadsheet_path = os.path.normpath(DATA_PAC5_EXCEL)
    if not os.path.isfile(spreadsheet_path):
        raise FileNotFoundError(
            f"The PAC5 spreadsheet does not exist: {spreadsheet_path}"
        )
    print(
        f"{database_path} does not exist so it is created from {spreadsheet_path}. This step runs only once."
    )

    spreadsheet = polars.read_excel(
        spreadsheet_path, sheet_name="PAC5", schema_overrides=SPREADSHEET_COLUMNS
    )
    smiles_list = spreadsheet["smiles"].to_list()

    column_definitions = ", ".join(
        f"{column_name} {column_type}"
        for column_name, column_type in PAC5_TABLE_COLUMNS.items()
    )
    column_names = ", ".join(PAC5_TABLE_COLUMNS)
    placeholders = ", ".join(f":{column_name}" for column_name in PAC5_TABLE_COLUMNS)
    insert_statement = f"INSERT INTO pac5 ({column_names}) VALUES ({placeholders})"

    partial_path = database_path + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    num_stored = 0
    num_skipped = 0
    connection = sqlite3.connect(partial_path)
    try:
        connection.execute(f"CREATE TABLE pac5 ({column_definitions})")
        with create_executor(num_workers) as executor:
            serializations = executor.map(
                smiles_to_serialized_pac, smiles_list, chunksize=100
            )
            for row, serialization in tqdm(
                zip(spreadsheet.iter_rows(named=True), serializations),
                total=len(smiles_list),
                desc="Converting SMILES strings to PAC grids",
            ):
                if serialization is None:
                    num_skipped += 1
                    continue
                characteristics = {
                    f"characteristics_{position}": int(digit)
                    for position, digit in enumerate(row["category"], start=1)
                }
                molecule = Chem.MolFromSmiles(row["smiles"])
                connection.execute(
                    insert_statement,
                    {
                        **row,
                        **characteristics,
                        "serialization": serialization,
                        "num_total_rings": molecule.GetRingInfo().NumRings(),
                    },
                )
                num_stored += 1

        connection.execute("CREATE INDEX pac5_pubchem_cid_index ON pac5 (pubchem_cid)")
        connection.execute("CREATE INDEX pac5_category_index ON pac5 (category)")
        connection.commit()
    finally:
        connection.close()
    os.replace(partial_path, database_path)

    print(f"Molecules stored in the PAC5 database {num_stored:,}")
    print(f"Molecules skipped (conversion to a PAC grid failed) {num_skipped:,}")
