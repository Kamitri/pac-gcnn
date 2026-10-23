# `dataset`

The `dataset` package contains the PyTorch datasets for each molecular representation and the scripts that create the data partitions.

The PAC5 dataset classes obtain molecular structures and calculated properties from the same SQLite database.

## Common Utilities

```{eval-rst}
.. automodule:: dataset.common
   :members:
   :show-inheritance:
```

## PAC5 Database

```{eval-rst}
.. automodule:: dataset.pac5_database
   :members:
   :show-inheritance:
```

## PAC Grid Datasets

```{eval-rst}
.. automodule:: dataset.pac_matrix
   :members:
   :show-inheritance:
```

## Baseline Input Representations

The baseline datasets represent each molecule as a tokenised SMILES sequence or a Morgan fingerprint.

```{eval-rst}
.. automodule:: dataset.smiles
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: dataset.fingerprint
   :members:
   :show-inheritance:
```

## External Benchmark Datasets

These modules provide the COMPAS-1D and PubChemQC benchmark datasets.

```{eval-rst}
.. automodule:: dataset.compas
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: dataset.pubchemqc
   :members:
   :show-inheritance:
```

## Dataset Splitting Protocols

```{eval-rst}
.. automodule:: dataset.split
   :members:
   :show-inheritance:
```
