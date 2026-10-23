# `benchmarks`

```{eval-rst}
.. automodule:: benchmarks
   :members:
   :show-inheritance:
```

## PyTorch Lightning Wrappers

```{eval-rst}
.. automodule:: benchmarks.lightning_integration
   :members:
   :show-inheritance:
```

## Target Scaling

```{eval-rst}
.. automodule:: benchmarks.scaler
   :members:
   :show-inheritance:
```

## Baseline Models

The PAC G-CNN is compared with three baseline architectures:

- A plain 2D CNN over the PAC grid (`benchmarks.cnn`).
- A bidirectional LSTM over SMILES sequences (`benchmarks.bilstm`).
- A multi-layer perceptron over Morgan fingerprints (`benchmarks.mlp`).

```{eval-rst}
.. automodule:: benchmarks.cnn
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: benchmarks.bilstm
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: benchmarks.mlp
   :members:
   :show-inheritance:
```
