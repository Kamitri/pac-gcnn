# PAC Grid Representation

The PAC grid maps a molecular graph onto a 2D hexagonal lattice. This page describes the lattice, its 13 input channels, and the D<sub>6</sub>-equivariant network that operates on the resulting grid.

## Grid Geometry

A polycyclic aromatic compound (PAC) contains a network of carbocyclic or heterocyclic rings. Each ring occupies 1 tile in a `grid_size` $\times$ `grid_size` array. The default `grid_size` is 15, as specified by `const.GRID_SIZE`.

Two fused rings occupy neighbouring tiles and share the two atoms along their common edge. Each shared atom is encoded at a coinciding vertex of both hexagons. Periodic boundary conditions are applied along both grid axes so the grid has the topology of a torus.

## Input Channels

A PAC grid has 13 channels. Each 1 has shape `(grid_size, grid_size)`.

| Channel | Feature | Values |
|---|---|---|
| 0 | Framework | `Framework`: 0 (empty), 1 (6-membered ring), 2 (5-membered ring), 3 (linker) |
| 1 to 6 | Functional group at each vertex | `FunctionalGroup`, 0 when the vertex does not have a functional group |
| 7 to 12 | Atom type and hybridisation at each vertex | `AtomType`, 0 when the vertex is empty |

The six vertices of a hexagon are numbered clockwise from 0 to 5, beginning with `VertexDirection.TOP`. The channel index for each vertex is calculated as follows:

- Functional groups: channel $= 1 + \text{VertexDirection.value}$, which is channels 1 to 6.
- Atom types: channel $= 7 + \text{VertexDirection.value}$, which is channels 7 to 12.

A channel value is accessed with `pac[channel, column_index, row_index]`.

When a ring is assigned to a tile, `PAC.__setitem__` sets the atom type of every occupied vertex of the hexagon to `CARBON_SP2`. An sp<sup>3</sup> carbon or a heteroatom assigned to 1 vertex is copied to all coinciding vertices of the same atom.

## Coordinate Conventions

The axial coordinates are defined in `pac.geometry`. A tile is indexed by the position $(\text{column\_index}, \text{row\_index})$. For a hexagon circumradius of $R = 1$, the Cartesian centre $(x, y)$ of the hexagon at a position is

$$
x = \sqrt{3}\,\left(\text{column\_index} - \frac{\text{row\_index}}{2}\right),
\qquad
y = -1.5 \times \text{row\_index}
$$

The column coordinate increases to the right. The row coordinate increases down and to the left, at $120^\circ$ clockwise from the positive $x$ axis.

The following conventions are used for directions and rotations:

- Vertices are numbered clockwise from `VertexDirection.TOP`, and the neighbouring tiles clockwise from `EdgeDirection.TOP_RIGHT`.
- Lattice rotations proceed counterclockwise in $60^\circ$ increments. Their definitions are in `pac.matrix_manipulation` and `pac.geometry`.

## Categories

`PAC.category` returns a five-digit binary string that records whether each feature is present. The category records presence rather than the number of occurrences.

| Digit | Feature | When it is 1 |
|---|---|---|
| 1 | Six-membered rings | Every PAC in scope contains at least one. |
| 2 | Five-membered rings | The molecule has a 5-membered ring, as in fluorene or fluoranthene. |
| 3 | Heteroatoms or sp<sup>3</sup> carbons | The molecule has an atom other than aromatic sp<sup>2</sup> carbon, which is N, O, S or sp<sup>3</sup> C. |
| 4 | Linker tiles | Two rings are joined by a linker instead of being fused, as in biphenyl. |
| 5 | Functional groups | The molecule has a functional group, such as -NO<sub>2</sub>, -CN, -CH<sub>3</sub> or a halogen. |

For example, `10000` is an unsubstituted benzenoid polycyclic aromatic hydrocarbon. `11000` is a carbocyclic PAC that has a 5-membered ring.

The **PAC5-4** benchmark comprises the five categories with at most 1 optional feature: `10000`, `11000`, `10100`, `10010`, and `10001`. The cross-validation partitions are stratified by category and total ring count.

(pac-g-cnn-architecture)=
## Polycyclic Aromatic Compound Group-Equivariant Convolutional Neural Network (PAC G-CNN)

A molecular property is unchanged when the PAC grid is rotated or reflected. These transformations form the dihedral group $D_6$, which has 6 rotations and 6 reflections.

A conventional CNN is equivariant to translations but not to rotations or reflections. Data augmentation can expose the model to different orientations, but does not guarantee identical predictions across orientations.

The PAC G-CNN instead enforces D<sub>6</sub> equivariance in each convolutional layer. Transforming the input by any $g \in D_6$ and then applying a layer produces the same result as applying the layer first and transforming its feature map. Pooling over the spatial dimensions and averaging over the group dimension makes the final prediction invariant to orientation.

The `gcnn` package contains the following components:

- `D6LiftingKernel` maps an input grid of shape $(\text{batch}, C_{\text{in}}, H, W)$ to a PAC feature map of shape $(\text{batch}, C_{\text{out}}, 12, H, W)$.
- `D6GroupKernel` and `GroupConvolution` convolve over the spatial dimensions $(H, W)$ and the 12-element group dimension.
- `EquivariantCategoricalProjector` embeds the framework, functional-group, and atom-type codes. A D<sub>6</sub> transformation also permutes the functional-group and atom-type channels.
- `HexMixedInputLayer` combines the embedded framework channel with the functional-group and atom-type channels before the lifting convolution.
- `gcnn.transform` defines the kernel transformations. Its precomputed index tables apply all 12 transformations on a GPU.

## Conversion Pipeline and Validation

```{mermaid}
flowchart LR
    A[SMILES string] -->|smiles_to_pac| B[RDKit molecule]
    B -->|mol_to_nx| C[NetworkX graph]
    C -->|nx_to_pac| D[PAC grid]
    D -->|to_tensor| E[13-channel tensor]
    E --> F[GroupEquivariantCNN]
```

`pac.smiles_to_pac` performs the complete conversion. During the final stage, `nx_to_pac` uses backtracking to embed the ring system on the lattice.

The representation is limited to molecular graphs that can be embedded on the hexagonal lattice. This excludes non-planar systems, incompatible fused-ring topologies, and linker geometries that cannot be embedded.

`mol_to_pac` and `smiles_to_pac` check each completed grid by reconstructing its molecular graph and comparing the canonical SMILES with the input. When every grid fails the check, the first grid built is returned. A ValueError is raised only when the search does not build any grid. `dataset.pac5_database.smiles_to_serialized_pac` applies a stricter check and returns None when the InChIKeys differ.
