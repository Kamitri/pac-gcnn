# Glossary

This glossary defines the terminology used for the molecular representation, D<sub>6</sub> transformations, network architecture, and datasets.

## Index

**The molecule and the grid**
[PAC](#pac) ·
[PAC grid](#pac-grid) ·
[Channel](#channel) ·
[Tile](#tile) ·
[Tile stack](#tile-stack) ·
[Position](#position) ·
[Hexagon](#hexagon) ·
[Axial coordinates](#axial-coordinates) ·
[Vertex](#vertex) ·
[Node](#node) ·
[Neighbouring tiles](#neighbouring-tiles) ·
[Fusion](#fusion) ·
[Linking](#linking)

**Contents of the PAC grid**
[Framework](#framework) ·
[Occupancy](#occupancy) ·
[Atom type](#atom-type) ·
[Functional group](#functional-group) ·
[Coinciding vertices](#coinciding-vertices) ·
[Empty vertex](#empty-vertex) ·
[Linker](#linker) ·
[Apex](#apex) ·
[Seed ring](#seed-ring) ·
[Root ring](#root-ring)

**Transformation of the PAC grid**
[D<sub>6</sub>](#d6) ·
[Image](#image) ·
[Orbit](#orbit) ·
[Canonical orientation](#canonical-orientation) ·
[Realignment](#realignment) ·
[Wrap around](#wrap-around) ·d
[Serialisation](#serialisation)

**The network**
[Equivariance](#equivariance) ·
[Invariance](#invariance) ·
[Lifting](#lifting) ·
[PAC feature map](#pac-feature-map) ·
[Feature dimension](#feature-dimension) ·
[Group dimension](#group-dimension) ·
[Group convolution](#group-convolution) ·
[Pooling](#pooling)

**The data**
[Category](#category) ·
[PAC chemical space](#pac-chemical-space) ·
[Intensive and extensive](#intensive-and-extensive)

## The molecule and the grid

### PAC

**Relationship:** A PAC is the molecular entity represented by a [PAC grid](#pac-grid) and implemented by `pac.PAC`.

A **polycyclic aromatic compound (PAC)** contains at least 1 6-membered ring and may also contain 5-membered rings, heteroatoms, linkers, and functional groups. In the code, `PAC` also refers to an instance of `pac.PAC`, which represents a molecule as a [PAC grid](#pac-grid).

Naphthalene, pyrene, and carbazole are examples of PACs covered by the representation. Motifs such as 7-membered rings and non-planar fused systems are outside the scope of the representation.

### PAC grid

**Relationship:** A PAC grid is the tensor representation of a [PAC](#pac). Its 13 [channels](#channel) are defined over a 2D lattice of [positions](#position).

A **PAC grid** is the tensor representation of a [PAC](#pac). It has 13 [channels](#channel) defined on a 2D hexagonal lattice with periodic boundaries. The size of the square grid is defined by `const.GRID_SIZE`, which is 15 by default. A linear fragment can span at most `GRID_SIZE - 1` tiles because an extent of `GRID_SIZE` is interpreted as a loop around the torus.

The terms *PAC grid* and *grid* are used interchangeably. Anthracene, for example, occupies 3 of the 225 positions of the default grid.

```{image} static/fig_1_light.svg
:class: only-light
:align: center
:width: 46%
:alt: A hexagonal lattice with three occupied tiles labelled with their axial coordinates.
```

```{image} static/fig_1_dark.svg
:class: only-dark
:align: center
:width: 46%
:alt: A hexagonal lattice with three occupied tiles labelled with their axial coordinates.
```

**Fig. 1 | Positions on the PAC grid.** Part of the hexagonal lattice. The 3 occupied tiles contain a ring and are labelled with their axial coordinates.

### Channel

**Relationship:** A channel is 1 of the 13 discrete arrays that form the input channel dimension of a [PAC grid](#pac-grid).

A **channel** is a `(grid_size, grid_size)` discrete array within a [PAC grid](#pac-grid). The 13 channels form the input channel dimension. Channel 0 encodes the [framework](#framework). Channels 1-6 encode [functional groups](#functional-group) at the six vertex directions, and channels 7-12 encode the corresponding [atom types](#atom-type) and hybridisation states.

For example, `pac[7, 3, 4]` returns the atom type at the `TOP` vertex of the hexagon in column 3 and row 4.

```{image} static/fig_2_light.svg
:class: only-light
:align: center
:width: 55%
:alt: Thirteen boxes numbered 0 to 12, grouped as framework, functional group and atom type.
```

```{image} static/fig_2_dark.svg
:class: only-dark
:align: center
:width: 55%
:alt: Thirteen boxes numbered 0 to 12, grouped as framework, functional group and atom type.
```

**Fig. 2 | Channels of the PAC grid.** The 13 channels, numbered 0 to 12: the framework channel, the 6 functional-group channels, and the 6 atom-type channels.

### Tile

**Relationship:** A tile is an element of a [channel](#channel) of the [PAC grid](#pac-grid). The 13 tiles at 1 [position](#position) are a [tile stack](#tile-stack).

A **tile** is an integer of the grid, selected by a channel and a position. `pac[0, 7, 7]` is 1 tile, the framework tile at the centre. `pac[7, 7, 7]` is another, the atom type tile of the `TOP` vertex of the hexagon at the same position. A $15 \times 15$ grid has $13 \times 225 = 2925$ tiles.

### Tile stack

**Relationship:** A tile stack is the 13 [tiles](#tile) at 1 [position](#position), 1 per [channel](#channel).

A **tile stack** is 1 [framework](#framework) tile, 6 [functional group](#functional-group) tiles and 6 [atom type](#atom-type) tiles at the same position, all at once.

`PAC.realign` moves an entire tile stack so the 13 tiles at a position all move to the same new position. When `PAC.graph` removes a piece of a fragmented grid, it sets entire tile stacks to 0.

### Position

**Relationship:** A position is an [axial coordinate](#axial-coordinates) pair on the lattice of the PAC grid. A position and a [channel](#channel) together select 1 [tile](#tile).

A **position** is an axial coordinate pair $(\text{column\_index}, \text{row\_index})$ on the hexagonal lattice. Both coordinates wrap modulo `grid_size`. On a $15 \times 15$ grid, `(7, 7)` is the central position.

### Hexagon

**Relationship:** The lattice of the [PAC grid](#pac-grid) has 1 hexagon at each [position](#position). The hexagon of a [tile](#tile) is the hexagon at the position of the tile.

A **hexagon** is the regular hexagon at 1 position of the lattice. The [vertices](#vertex), the edges, the centre and the circumradius belong to the hexagon, because a tile is a number without a shape. The [tile stack](#tile-stack) at a position has 1 functional group tile and 1 atom type tile for each vertex of the hexagon.

```{image} static/fig_3_light.svg
:class: only-light
:align: center
:width: 30%
:alt: One hexagon with its six vertices numbered 0 to 5 clockwise from the top.
```

```{image} static/fig_3_dark.svg
:class: only-dark
:align: center
:width: 30%
:alt: One hexagon with its six vertices numbered 0 to 5 clockwise from the top.
```

**Fig. 3 | Vertices of a hexagon.** The 6 vertices of 1 hexagon, numbered 0 to 5 clockwise from the top.

### Axial coordinates

**Relationship:** Axial coordinates are the coordinate system for [positions](#position) on the hexagonal lattice.

**Axial coordinates** index positions on the hexagonal lattice. The Cartesian centre $(x, y)$ of the [hexagon](#hexagon) at a position is calculated from

$$
x = \sqrt{3}\,\left(\text{column\_index} - \frac{\text{row\_index}}{2}\right),
\qquad
y = -1.5 \times \text{row\_index}
$$

where distances are expressed in units of the circumradius of 1 hexagon. The position `(1, 0)` is directly to the right of `(0, 0)`, whereas `(0, 1)` is below and to the left.

### Vertex

**Relationship:** A vertex is 1 of the 6 geometric corners of a [hexagon](#hexagon). `pac.enums.VertexDirection` indexes them clockwise from the top.

A **vertex** is 1 of the 6 corners of a [hexagon](#hexagon). `pac.enums.VertexDirection` numbers them clockwise from 0 to 5, and this number is the vertex direction. A vertex may correspond to an atom in the molecular graph, contain a substituent, or remain [empty](#empty-vertex). An atom shared by fused rings is encoded at 1 vertex of each of their hexagons, and these vertices are [coinciding vertices](#coinciding-vertices). A 5-membered ring has an atom on five of its six [atom type](#atom-type) tiles and leaves the sixth empty.

### Node

**Relationship:** A node is an atom in the molecular graph. One node can correspond to more than 1 [vertex](#vertex). Despite this project mainly working grids, molecular graph is the intermediate representation when converting between SMILES and PAC grid so the distinction still needs to be made.

A **node** is an atom in the molecular graph. The same node may be encoded at several [coinciding vertices](#coinciding-vertices) when it is shared by fused rings. `VertexDirection` refers to the vertices of a hexagon rather than to graph nodes. For example, the two bridgehead carbons of naphthalene are represented by 2 graph nodes but 4 atom type tiles, 2 for each ring tile.

### Neighbouring tiles

**Relationship:** Neighbouring tiles are the six [tiles](#tile) in the same channel with [hexagons](#hexagon) adjacent to the hexagon of a given tile.

The **neighbouring tiles** are the six tiles in the same channel with hexagons that share an edge with the hexagon of a given [tile](#tile). `pac.enums.EdgeDirection` lists them clockwise from the upper right as `TOP_RIGHT`, `RIGHT`, `BOTTOM_RIGHT`, `BOTTOM_LEFT`, `LEFT`, and `TOP_LEFT`. Their axial offsets are defined by `pac.enums.TRANSFORMATIONS`.

Although the lattice is stored in a square array, the offsets `(+1, -1)` and `(-1, +1)` are not neighbours on the hexagonal lattice. The hexagonal convolution mask excludes the same two corners. From `(0, 0)`, for example, the `RIGHT` neighbour is `(1, 0)` and the `BOTTOM_LEFT` neighbour is `(0, 1)`.

```{image} static/fig_4_light.svg
:class: only-light
:align: center
:width: 26%
:alt: One tile surrounded by its six neighbours, each labelled with its direction.
```

```{image} static/fig_4_dark.svg
:class: only-dark
:align: center
:width: 26%
:alt: One tile surrounded by its six neighbours, each labelled with its direction.
```

**Fig. 4 | Neighbouring tiles.** The 6 neighbouring tiles of 1 tile, labelled with the direction from the tile.

### Fusion

**Relationship:** Fusion joins two rings of a [PAC](#pac) on [neighbouring tiles](#neighbouring-tiles).

**Fusion** means that two molecular rings share two atoms along a common edge. Fused rings occupy [neighbouring tiles](#neighbouring-tiles), and their shared atoms are encoded at both positions. Naphthalene is represented by two neighbouring ring tiles, whereas the three ring tiles of anthracene are in a line along 1 lattice axis.

```{image} static/fig_5_light.svg
:class: only-light
:align: center
:width: 85%
:alt: Three panels of tiles. Two ring tiles with hexagons that share an edge, labelled fusion. Two ring tiles and two linker tiles, labelled linking through a bond. Two ring tiles and three linker tiles, labelled linking through a chain.
```

```{image} static/fig_5_dark.svg
:class: only-dark
:align: center
:width: 85%
:alt: Three panels of tiles. Two ring tiles with hexagons that share an edge, labelled fusion. Two ring tiles and two linker tiles, labelled linking through a bond. Two ring tiles and three linker tiles, labelled linking through a chain.
```

**Fig. 5 | Fusion and linking.** The ring and linker tiles of the framework channel of naphthalene, biphenyl, and diphenylmethane, from left to right.

### Linking

**Relationship:** Linking joins two rings of a [PAC](#pac) through 1 or more [linker](#linker) tiles without fusing the rings.

**Linking** describes rings joined by a covalent bond or short chain rather than by fusion. The rings do not share any atoms and are not assigned to neighbouring tiles. Every position crossed by the connection gets framework tile 3, which prevents a ring from being placed there.

The central bond of biphenyl passes through two linker tiles. The one-carbon chain in diphenylmethane passes through three.

## Contents of the PAC grid

### Framework

**Relationship:** The framework is the categorical encoding in channel 0. It determines the [occupancy](#occupancy) and structural type at each [position](#position).

The **framework** is the categorical encoding in channel 0, defined by `pac.enums.Framework`. Each framework tile is empty (0), a 6-membered ring (1), a 5-membered ring (2), or part of a linker (3). Assigning a ring also initialises the corresponding atom type tiles.

For example, `pac[0, 7, 7] = Framework.SIX_MEMBERED` places a 6-membered carbon ring at the central position and initialises its vertices as sp<sup>2</sup> carbons.

### Occupancy

**Relationship:** Occupancy is the state of a [framework](#framework) tile determined by whether the tile is zero.

A [framework](#framework) tile is **occupied** when it is non-zero, encoding either a molecular ring or part of a linker. A 4-ring fused molecule on a $15 \times 15$ grid occupies 4 of the 225 positions.

```{image} static/fig_6_light.svg
:class: only-light
:align: center
:width: 17%
:alt: An occupied framework tile with six atoms beside an unoccupied framework tile of 0.
```

```{image} static/fig_6_dark.svg
:class: only-dark
:align: center
:width: 17%
:alt: An occupied framework tile with six atoms beside an unoccupied framework tile of 0.
```

**Fig. 6 | Occupancy.** An occupied tile, with the framework value 1, 2, or 3 and an atom at each vertex of the hexagon, beside an unoccupied tile with the framework value 0.

### Atom type

**Relationship:** Atom types are encoded in channels 7-12 and describe the atoms at the [vertices](#vertex) of a [hexagon](#hexagon).

An **atom type** specifies the element and hybridisation at 1 [vertex](#vertex) of a [hexagon](#hexagon). Channels 7-12 use the integer codes in `pac.enums.AtomType`; an atom type tile of 0 indicates an [empty vertex](#empty-vertex). The sp<sup>3</sup> carbon and heteroatom codes are synchronised between [coinciding vertices](#coinciding-vertices), and the aromatic and sp<sup>2</sup> carbon codes are set separately for each hexagon.

The pyridine nitrogen, for example, is encoded as `AtomType.NITROGEN_SP2_1PI` (19), while the surrounding aromatic carbons are `CARBON_AROMATIC` (15).

### Functional group

**Relationship:** Functional groups are encoded in channels 1-6 and describe substituents attached at the [vertices](#vertex) of a [hexagon](#hexagon).

A **functional group** is a substituent attached at 1 [vertex](#vertex) of a [hexagon](#hexagon). Channels 1-6 use the integer codes in `pac.enums.FunctionalGroup`; a functional group tile of 0 indicates that a substituent is not attached at that vertex. Conversion back to a molecular graph reconstructs the atoms represented by each code.

For example, `pac[1, 7, 7] = FunctionalGroup.METHYL` attaches a methyl group to the `TOP` vertex of the hexagon at the central position.

### Coinciding vertices

**Relationship:** Coinciding vertices are [vertices](#vertex) of different [hexagons](#hexagon) at the same point of the lattice. They describe how the [PAC grid](#pac-grid) encodes atoms shared by fused rings.

**Coinciding vertices** are the vertices of the 3 hexagons that meet at 1 point of the lattice. A fused-ring atom is encoded at 2 or 3 of these vertices, 1 in the hexagon of each ring tile that has the atom, and `PAC.identify_coinciding_vertices` returns these vertices. Each of the two atoms along a fused edge appears once for each ring tile, and the sp<sup>3</sup> carbon and heteroatom codes of all copies are kept consistent. The fused edge in naphthalene therefore corresponds to 2 molecular atoms but 4 encoded atom type tiles.

```{image} static/fig_7_light.svg
:class: only-light
:align: center
:width: 17%
:alt: The hexagons of two neighbouring ring tiles with the two shared atoms marked, each atom at 1 vertex of each hexagon.
```

```{image} static/fig_7_dark.svg
:class: only-dark
:align: center
:width: 17%
:alt: The hexagons of two neighbouring ring tiles with the two shared atoms marked, each atom at 1 vertex of each hexagon.
```

**Fig. 7 | Coinciding vertices.** The hexagons of 2 fused ring tiles. Each of the 2 shared atoms is encoded at 1 vertex of each hexagon.

### Empty vertex

**Relationship:** An empty vertex is a [vertex](#vertex) with [atom type](#atom-type) 0.

An **empty vertex** has an [atom type](#atom-type) tile of 0. Five-membered rings use five vertices of their [hexagon](#hexagon) and leave the sixth empty. Empty vertices can also occur in open linker chains. The hexagon of a thiophene ring tile, for example, has 5 occupied vertices and 1 empty vertex.

```{image} static/fig_8_light.svg
:class: only-light
:align: center
:width: 10%
:alt: A hexagon containing a five-membered ring, with five vertices filled and 1 drawn empty.
```

```{image} static/fig_8_dark.svg
:class: only-dark
:align: center
:width: 10%
:alt: A hexagon containing a five-membered ring, with five vertices filled and 1 drawn empty.
```

**Fig. 8 | Empty vertex.** The hexagon of a five-membered ring, with an atom at 5 vertices and an empty sixth vertex of atom type 0 (open circle).

### Linker

**Relationship:** A linker is a [framework](#framework) tile of 3, and is part of a connection between [linked](#linking) rings.

A **linker** tile is a framework tile of 3 (`pac.enums.Framework.LINKER`) and encodes part of the connection between rings that are [linked](#linking) rather than fused. The atom type tiles at its position describe the open chain, which may be a single bond, a double bond, or a short carbon chain. A connection usually crosses more than 1 linker tile; the central bond of biphenyl crosses two.

### Apex

**Relationship:** The apex is the vertex of a 5-membered ring opposite its [empty vertex](#empty-vertex).

The **apex** of a 5-membered ring is the vertex opposite its [empty vertex](#empty-vertex). This vertex corresponds to the nitrogen in pyrrole and carbazole, or to the bridging carbon in fluorene.

### Seed ring

**Relationship:** The seed ring is the initial 6-membered ring created by `PAC()`.

The **seed ring** is the initial 6-membered ring created by `PAC()`. It is placed at `init_pos`, which defaults to the centre of the grid. Calling `PAC()` with the default $15 \times 15$ grid therefore creates a carbon ring at `(7, 7)`.

### Root ring

**Relationship:** The root ring is the starting ring used when a molecular graph is converted into a [PAC grid](#pac-grid).

The **root ring** is the first molecular ring placed during conversion from a graph to a PAC grid. The remaining rings and linkers are embedded from this starting point. For anthracene, either a terminal ring or the central ring can be the root, after which the other two rings are added sequentially.

## Transformation of the PAC grid

### D<sub>6</sub>

**Relationship:** D<sub>6</sub> is the symmetry group used by the PAC G-CNN. Its 12 elements determine the size of the [group dimension](#group-dimension).

**D<sub>6</sub>** is the dihedral symmetry group of a regular hexagon. Its 12 elements comprise 6 rotations and 6 reflections. The implementation orders the rotations first, from $0^\circ$ to $300^\circ$ counterclockwise in $60^\circ$ increments, followed by a reflection composed with each rotation.

In algebraic notation, the elements are $e, r, r^2, r^3, r^4, r^5$ and $s, rs, r^2s, r^3s, r^4s, r^5s$, where $r$ is a $60^\circ$ rotation and $s$ is a reflection.

```{image} static/fig_9_light.svg
:class: only-light
:align: center
:width: 96%
:alt: Twelve hexagons, six rotations on the top row and six reflections on the bottom, each with a large and a small mark indicating the orientation of the hexagon.
```

```{image} static/fig_9_dark.svg
:class: only-dark
:align: center
:width: 96%
:alt: Twelve hexagons, six rotations on the top row and six reflections on the bottom, each with a large and a small mark indicating the orientation of the hexagon.
```

**Fig. 9 | The 12 elements of D<sub>6</sub>.** The rotations $e$ to $r^5$ (top row) and the reflections $s$ to $r^5s$ (bottom row) of a hexagon. The large and the small mark show the orientation of the hexagon after each transformation.

### Image

**Relationship:** An image is a [PAC grid](#pac-grid) transformed by 1 D<sub>6</sub> group element.

An **image** is the grid obtained by applying a [D<sub>6</sub>](#d6) transformation to a [PAC grid](#pac-grid). All 12 images encode the same molecule in different orientations. The PAC G-CNN therefore produces the same property prediction for every image of anthracene.

### Orbit

**Relationship:** An orbit is the set of [images](#image) generated by applying every D<sub>6</sub> group element to 1 [PAC grid](#pac-grid).

The **orbit** of a [PAC grid](#pac-grid) is the set of its transformed [images](#image), $\{g \cdot x \mid g \in D_6\}$. `pac.geometry.d6_orbit` returns these 12 grids in the transformation order defined above, beginning with the untransformed grid at index 0.

### Canonical orientation

**Relationship:** The canonical orientation is the deterministic [image](#image) selected during conversion to a PAC grid.

The **canonical orientation** is the deterministic [image](#image) selected during conversion from SMILES. It minimises the area of the molecular bounding box and then places the molecule as close to the grid centre as possible. Repeated conversion of the same molecular graph therefore produces the same serialised grid regardless of the input atom indexing.

### Realignment

**Relationship:** Realignment translates a [PAC grid](#pac-grid) so that its molecular bounding box is centred.

**Realignment** translates all 13 channels so that the molecular bounding box is centred on the PAC grid. `PAC.realign` applies the translation circularly across the torus. For example, `pac.rotated().realign()` realigns the rotated PAC grid at `(7, 7)` on the default grid.

### Wrap around

**Relationship:** Wrap around occurs when a connected structure crosses a periodic boundary of the [PAC grid](#pac-grid).

A connected structure **wraps around** the PAC grid when neighbouring tiles occur on opposite edges of the stored array. A 3-ring structure in columns 14, 0, and 1 of a $15 \times 15$ grid is an example. These columns are contiguous because the grid has periodic boundaries. `PAC.calculate_bounding_box` treats them as 1 continuous interval.

### Serialisation

**Relationship:** Serialisation is the compact text representation of a [PAC grid](#pac-grid).

**Serialisation** is the compact text representation produced by `PAC.serialize`. Each of the 13 flattened channels is run-length encoded, and the channel strings are separated by semicolons. `PAC.deserialize` reconstructs the grid from this string. The dataset loaders use this format to convert database records back into `PAC` instances.

## The network

### Equivariance

**Relationship:** Equivariance describes how the hidden representation changes when a D<sub>6</sub> transformation is applied to the [PAC grid](#pac-grid).

**Equivariance** requires a transformation of the input to produce the corresponding transformation of the output. A network layer $F_\ell$ is D<sub>6</sub>-equivariant when

$$
F_\ell(T_g x) = \rho_g F_\ell(x), \qquad g \in D_6,
$$

where $T_g$ acts on the input PAC grid and $\rho_g$ acts on the spatial dimensions and group dimension of the feature map. A $60^\circ$ rotation of the input therefore rotates the spatial activations and cyclically permutes the features along the group dimension.

### Invariance

**Relationship:** Invariance describes why every D<sub>6</sub> orientation of the same [PAC grid](#pac-grid) produces the same final prediction.

**Invariance** requires the final prediction to remain unchanged under every D<sub>6</sub> transformation:

$$
P(T_g x) = P(x), \qquad g \in D_6.
$$

The PAC G-CNN obtains an invariant representation by pooling over the spatial dimensions and averaging over the 12-element group dimension. All orientations of the same PAC grid consequently produce the same prediction up to numerical precision.

### Lifting

**Relationship:** Lifting converts the embedded representation of a [PAC grid](#pac-grid) into a [PAC feature map](#pac-feature-map) with an explicit [group dimension](#group-dimension).

**Lifting** applies all 12 D<sub>6</sub> transformations of the convolution kernel and produces a tensor of shape $(\text{batch}, C, 12, H, W)$. In the PAC G-CNN, `EquivariantCategoricalProjector` first embeds the categorical grid channels, after which `HexMixedInputLayer` performs the lifting convolution.

The tensor shape changes at each stage of the network:

| Stage | Tensor | Shape | Non-spatial dimensions |
| --- | --- | --- | --- |
| Model input | PAC grid | $(\text{batch}, 13, H, W)$ | Input channel dimension, size 13 |
| After embedding and flattening | Embedded feature map | $(\text{batch}, 208, H, W)$ | Feature dimension, size 208 |
| After lifting | PAC feature map | $(\text{batch}, 32, 12, H, W)$ | Feature dimension, size 32; group dimension, size 12 |

The PAC grid and PAC feature map are different tensors. The PAC grid stores the categorical molecular representation. The PAC feature map stores learned features produced by the network. The group dimension first appears during lifting.

### PAC feature map

**Relationship:** A PAC feature map is the learned representation produced by [lifting](#lifting) and processed by the [group convolutions](#group-convolution).

A **PAC feature map** has shape $(C, G, H, W)$ for 1 PAC, or $(\text{batch}, C, G, H, W)$ for a batch. The PAC G-CNN uses $C = 32$ feature channels and $G = 12$ group elements. The final two dimensions are the spatial dimensions of the hexagonal lattice.

### Feature dimension

**Relationship:** The feature dimension is the $C$ dimension of an embedded feature map or [PAC feature map](#pac-feature-map).

The **feature dimension** indexes learned feature channels. The embedded feature map has 208 feature channels before lifting, and each PAC feature map has 32 feature channels for every D<sub>6</sub> group element. In a PAC feature map with shape $(\text{batch}, C, G, H, W)$, $C$ is the feature dimension. These learned feature channels are different from the 13 categorical input channels of the PAC grid.

### Group dimension

**Relationship:** The group dimension is the $G$ dimension of a [PAC feature map](#pac-feature-map). It appears during [lifting](#lifting) and has 1 index for each D<sub>6</sub> group element.

The **group dimension** has size 12. In a PAC feature map with shape $(\text{batch}, C, G, H, W)$, $G$ is the group dimension. One index $g$ selects the features associated with 1 group element. A rotation or reflection permutes the values along this dimension so their mean remains unchanged.

### Group convolution

**Relationship:** Group convolutions process [PAC feature maps](#pac-feature-map) after [lifting](#lifting) while preserving [equivariance](#equivariance).

**Group convolutions** convolve over the two spatial dimensions and the group dimension simultaneously. Each kernel transformation changes the spatial coordinates and permutes the values along the group dimension according to the same D<sub>6</sub> element, which preserves equivariance between layers. The hexagonal mask sets 6 corner weights in each $5 \times 5$ spatial kernel to zero, leaving 19 active weights.

### Pooling

**Relationship:** Pooling reduces a [PAC feature map](#pac-feature-map) to an [invariant](#invariance) molecular representation.

**Pooling** converts an equivariant feature map into an invariant molecular representation. The spatial dimensions are reduced by max pooling for [intensive](#intensive-and-extensive) properties and sum pooling for extensive properties. The values along the 12-element group dimension are then averaged.

For an intensive property, `x.amax(dim=(-2, -1)).mean(dim=2)` reduces a tensor of shape $(\text{batch}, 32, 12, 15, 15)$ to $(\text{batch}, 32)$.

## The data

### Category

**Relationship:** A category records which optional molecular features are present in 1 PAC.

A **category** is the five-digit binary string returned by `PAC.category`. Its digits indicate the presence of 6-membered rings, 5-membered rings, heteroatoms or sp<sup>3</sup> carbons, linkers, and functional groups, in that order. The PAC5-4 partitions are stratified by this category. For example, `10000` is an unsubstituted carbocyclic system, while `11000` also has at least 1 5-membered ring.

### PAC chemical space

**Relationship:** A PAC chemical space is the set of [categories](#category) permitted by a five-digit code.

A **PAC chemical space** is the set of [categories](#category) permitted by a five-digit code. A category belongs to the space when every feature present in the category is also allowed by the space code. The first digit is always set because all molecules in scope contain a 6-membered ring. If `k` of the four optional features are allowed, the space contains 2<sup>k</sup> categories.

For example, `10101` allows heteroatoms and functional groups in addition to 6-membered rings. It comprises `10000`, `10100`, `10001`, and `10101`.

### Intensive and extensive

**Relationship:** The intensive or extensive class of a molecular property determines the spatial [pooling](#pooling) used by its prediction head.

The distinction between **intensive and extensive** properties determines the spatial [pooling](#pooling) operation. HOMO and LUMO energies and the HOMO-LUMO gap are treated as intensive properties and use max pooling. Heat of formation is treated as an extensive property and uses sum pooling because its magnitude scales with molecular size. Both pooling heads subsequently average over the group dimension.
