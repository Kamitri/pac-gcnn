"""OpenFF is only available in Linux. Since its installation is very complex, refer to OpenFF's installation to install
the correct dependencies if this script is to be run. `pebble` is an additional dependency."""

import os
import sqlite3
from concurrent.futures import TimeoutError
from typing import TYPE_CHECKING, Annotated

import pandas as pd
import typer
from loguru import logger
from pebble import ProcessPool
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolDescriptors
from tqdm import tqdm

from const import DEFAULT_NUM_WORKERS

if TYPE_CHECKING:
    from openff.toolkit import ForceField

GLOBAL_FF: ForceField | None = None


def init_worker():
    """
    Load the OpenFF Sage force field once in each worker process.
    """
    global GLOBAL_FF
    from openff.toolkit import ForceField

    GLOBAL_FF = ForceField("openff-2.3.1.offxml")


def process_molecule(args):
    """
    Embed a molecule with ETKDGv3 and optimise it with OpenFF Sage.

    OpenMM performs the force-field optimisation and the final geometry is saved as an SDF file.

    Parameters
    ----------
    args : tuple
        SMILES string, PubChem CID, and output directory.

    Returns
    -------
    tuple
        PubChem CID and processing status.
    """
    global GLOBAL_FF
    smiles, pubchem_cid, directory = args
    output_path = os.path.join(directory, f"{pubchem_cid}.sdf")

    if os.path.exists(output_path):
        return pubchem_cid, "exists"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pubchem_cid, "invalid_smiles"

    mol = Chem.AddHs(mol)

    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if num_rotatable_bonds == 0:
        num_conformers = 20
    elif 1 <= num_rotatable_bonds < 4:
        num_conformers = 50
    elif 4 <= num_rotatable_bonds < 8:
        num_conformers = 100
    else:
        num_conformers = 200

    strat = rdDistGeom.ETKDGv3()
    strat.randomSeed = 42
    strat.numThreads = 1
    strat.useRandomCoords = True

    try:
        cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_conformers, params=strat)
    except Exception as e:
        return pubchem_cid, f"error: embed exception ({e!s})"

    if not cids:
        return pubchem_cid, "embed_failed"

    try:
        import openmm
        from openff.toolkit import Molecule
        from openff.units import unit as off_unit
        from openmm import unit as omm_unit

        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)

        topology = off_mol.to_topology()
        if GLOBAL_FF is None:
            raise RuntimeError(
                "The OpenFF force field is not loaded"
            )
        interchange = GLOBAL_FF.create_interchange(topology)
        openmm_sys = interchange.to_openmm()

        integrator = openmm.VerletIntegrator(1.0 * omm_unit.femtoseconds)

        # Reference platform is safest under heavy multi-process use
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(openmm_sys, integrator, platform)

        best_energy = float("inf")
        best_coords = None

        for conf in off_mol.conformers:
            xyz_nm = conf.m_as(off_unit.nanometer)
            context.setPositions(xyz_nm * omm_unit.nanometer)

            openmm.LocalEnergyMinimizer.minimize(context, tolerance=10.0)

            state = context.getState(getEnergy=True, getPositions=True)
            energy = state.getPotentialEnergy().value_in_unit(
                omm_unit.kilocalories_per_mole
            )

            if energy < best_energy:
                best_energy = energy
                best_coords = state.getPositions(asNumpy=True).value_in_unit(
                    omm_unit.angstrom
                )

        del context

        if best_coords is not None:
            off_mol._conformers = [best_coords * off_unit.angstrom]

            off_mol.properties["pubchem_cid"] = str(pubchem_cid)
            off_mol.properties["smiles"] = str(smiles)
            off_mol.properties["openff_energy_kcal_mol"] = f"{best_energy:.4f}"
            off_mol.properties["generation_strategy"] = (
                f"High_OpenFF_Sage_ETKDGv3_{num_conformers}"
            )

            off_mol.to_file(output_path, file_format="SDF")
            return pubchem_cid, "success (OpenFF)"
        else:
            return pubchem_cid, "error: OpenMM minimization failed to yield coordinates"

    except Exception as e:
        return pubchem_cid, f"error: Openff pipeline failed ({e!s})"


app = typer.Typer(add_completion=False)


@app.command(
    help="Run the conformer search for every PAC5 molecule without an SDF file in data/rdkit_openff_sdf."
)
def main(
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            "-t",
            help="Timeout (s) for processing a single molecule.",
        ),
    ] = 60,
    num_workers: Annotated[
        int,
        typer.Option(
            min=0,
            help="Number of worker processes.",
        ),
    ] = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Run the conformer search for every PAC5 molecule without an SDF file in ``data/rdkit_openff_sdf``.

    The molecules are the rows of the table ``pac5`` in ``data/pac5.db`` with ``excluded = 0``, and both paths are
    relative to the current directory. A molecule with an existing file ``{pubchem_cid}.sdf`` is skipped. Every other
    molecule is processed by `process_molecule` in a pool of `num_workers` worker processes, or of 1 worker process when
    `num_workers` is 0. The processing of a molecule that runs longer than `timeout` seconds is terminated, and a
    warning is logged.

    Parameters
    ----------
    timeout : int, default 60
        Timeout in seconds for processing a single molecule.
    num_workers : int, default const.DEFAULT_NUM_WORKERS
        Number of worker processes. A value of 0 starts 1 worker process, because the timeout terminates the worker
        process of a molecule.

    Raises
    ------
    typer.Exit
        Raised with the exit code 1 when the table cannot be loaded from the database.
    """
    pac5_dataset_sql_path = r"./data/pac5.db"

    directory = r"./data/rdkit_openff_sdf"
    os.makedirs(directory, exist_ok=True)

    try:
        with sqlite3.connect(pac5_dataset_sql_path) as conn:
            query = "SELECT * FROM pac5 WHERE excluded = 0"
            pac5_dataset_df = pd.read_sql(query, conn)
    except Exception as e:
        logger.error(f"Failed to read database: {e}")
        raise typer.Exit(code=1)

    existing_files = os.listdir(directory)
    existing_cids = {f.split(".")[0] for f in existing_files if f.endswith(".sdf")}

    tasks = []
    for _, row in pac5_dataset_df.iterrows():
        cid_str = str(row["pubchem_cid"])
        if cid_str not in existing_cids:
            tasks.append((row["smiles"], row["pubchem_cid"], directory))

    total_mols = len(pac5_dataset_df)
    skipped_mols = total_mols - len(tasks)

    logger.info(f"Total molecules in DB: {total_mols}")
    logger.info(f"Found {skipped_mols} existing SDF files; skipping existing files.")

    if not tasks:
        logger.info("All molecules already processed")
        return

    num_processes = max(num_workers, 1)

    logger.info(
        f"Running {len(tasks)} pending tasks (processes: {num_processes})"
    )

    with ProcessPool(max_workers=num_processes, initializer=init_worker) as pool:
        future = pool.map(process_molecule, tasks, timeout=timeout)
        iterator = future.result()

        with tqdm(total=len(tasks), desc="Processing Molecules") as pbar:
            while True:
                try:
                    pubchem_cid, status = next(iterator)

                    if "error" in status or status == "embed_failed":
                        logger.warning(
                            f"Skipping {pubchem_cid}: Geometry generation failed. ({status})"
                        )

                except StopIteration:
                    break
                except TimeoutError:
                    logger.warning(
                        f"A molecule processing task exceeded the timeout limit of {timeout}s and was terminated."
                    )
                except Exception as error:
                    logger.error(f"Worker raised an exception: {error}")
                finally:
                    pbar.update(1)


if __name__ == "__main__":
    app()
