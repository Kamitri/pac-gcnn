# PAC G-CNN

PAC G-CNN predicts electronic and thermodynamic properties of polycyclic aromatic compounds (PACs) from a PAC grid.

```{image} static/banner_light.svg
:class: only-light
:align: center
:width: 100%
:alt: PAC G-CNN: an input molecule, its PAC grid, and the predicted properties with the test RMSDs
```

```{image} static/banner_dark.svg
:class: only-dark
:align: center
:width: 100%
:alt: PAC G-CNN: an input molecule, its PAC grid, and the predicted properties with the test RMSDs
```

Each molecular ring is assigned to 1 tile of the lattice. Fused rings are assigned to neighbouring tiles, which preserves the connectivity of the ring system.

The PAC grid has 13 channels: channel 0 is the framework, channels 1 to 6 are the functional groups at the 6 vertices of a hexagon, and channels 7 to 12 are the atom types at the same vertices.

The network is equivariant under D<sub>6</sub>, the 12 rotations and reflections of a regular hexagon. The final prediction is therefore invariant to the orientation of the PAC grid.

```{toctree}
:maxdepth: 2
:caption: Guide

installation
representation
glossary
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

## Packages

`pac`
: The PAC grid, including conversion between molecular graphs and PAC grids, lattice geometry, and D<sub>6</sub> transformations.

`gcnn`
: The D<sub>6</sub>-equivariant lifting layer, group convolutions, and transformation tables.

`dataset`
: PyTorch datasets for PAC grids, fingerprints, and SMILES, together with the partitioning module.

`benchmarks`
: The CNN, BiLSTM, and MLP baselines, with their training and evaluation routines.

## Getting Started

- [Grid Representation](representation.md) describes the lattice encoding and D<sub>6</sub> equivariance.
- `pac.smiles_to_pac` converts a SMILES string into a PAC grid. The command-line prediction interface is provided in [`predict_properties.py`](../predict_properties.py).
- [Installation](installation.md) describes the software and data required to run the project.
