"""
Polycyclic aromatic compound group-equivariant convolutional neural network (PAC G-CNN) architecture.

Categorical PAC grid channels are embedded before a lifting convolution produces a PAC feature map. The network then
applies D6 group convolutions on the hexagonal lattice.
"""

import math
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn

from gcnn.transform import fast_d6_all_transforms, hexagon_axial, precompute_d6_indices


def init_hex_weights_(
    tensor: torch.Tensor,
    filter_mask: torch.Tensor,
    in_channels: int,
    num_groups: int = 1,
) -> None:
    """
    Initialise a hexagonal convolution kernel in place.

    The weights are sampled uniformly between `± 1/sqrt(fan_in)``. The fan-in includes only the kernel weights selected
    by the hexagonal mask because all other kernel weights are fixed at zero.

    Parameters
    ----------
    tensor : torch.Tensor
        Kernel weights to be initialised. The tensor is modified in place.
    filter_mask : torch.Tensor of shape (kernel_size, kernel_size)
        Hexagonal support mask. A value of 1.0 includes the corresponding kernel weight.
    in_channels : int
        Number of input channels per group element.
    num_groups : int, default 1
        Number of input group elements. This is 1 for a lifting convolution and 12 for a group convolution.
    """
    num_active_weights = filter_mask.sum().item()
    fan_in = in_channels * num_groups * num_active_weights
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


class FrozenKernelCache(nn.Module):
    """
    Cache transformed kernels during inference.

    Training and gradient-enabled evaluation rebuild the kernel on every call. Evaluation with gradients disabled reuses
    the transformed tensor until the weight version changes. Mode changes, device or dtype conversion, and state
    dictionary loading invalidate the cached tensor.
    """

    weight: nn.Parameter
    cached_kernel: torch.Tensor | None = None
    cached_weight_version: int | None = None

    def build_kernel(self) -> torch.Tensor:
        """
        Construct the transformed kernel.

        Raises
        ------
        NotImplementedError
            Raised if a subclass does not implement kernel construction.
        """
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        """
        Return the transformed kernel for the current weights.

        Returns
        -------
        torch.Tensor
            Transformed kernel. Inference may reuse the cached tensor.
        """
        if self.training or torch.is_grad_enabled():
            return self.build_kernel()
        if (
            self.cached_kernel is None
            or self.cached_weight_version != self.weight._version
        ):
            self.cached_kernel = self.build_kernel()
            self.cached_weight_version = self.weight._version
        return self.cached_kernel

    def train(self, mode: bool = True) -> Self:
        """
        Set the training mode and invalidate the cached kernel.

        Parameters
        ----------
        mode : bool, default True
            Whether to enable training mode.

        Returns
        -------
        torch.nn.Module
            This module.
        """
        self.cached_kernel = None
        return super().train(mode)

    def _apply(self, *args, **kwargs):
        self.cached_kernel = None
        return super()._apply(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """
        Load a state dictionary and invalidate the cached kernel.

        Parameters
        ----------
        *args
            Positional arguments accepted by `torch.nn.Module.load_state_dict`.
        **kwargs
            Keyword arguments accepted by `torch.nn.Module.load_state_dict`.

        Returns
        -------
        torch.nn.modules.module._IncompatibleKeys
            Missing and unexpected keys, as reported by the parent method.
        """
        self.cached_kernel = None
        return super().load_state_dict(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        # Loading through a parent module calls this method on every submodule
        self.cached_kernel = None
        return super()._load_from_state_dict(*args, **kwargs)


class D6LiftingKernel(FrozenKernelCache):
    """
    Construct a D6 lifting kernel for the framework channel.

    A single learnable kernel is transformed into the 12 rotations and reflections of D6. This kernel is used for the
    framework channel, which is not associated with a vertex direction.

    Parameters
    ----------
    kernel_size : int
        Side length of the square array containing the hexagonal kernel. The value must be odd.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels per group element.
    """

    def __init__(self, kernel_size: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.register_buffer(
            "filter_mask", hexagon_axial(kernel_size), persistent=False
        )
        self.register_buffer(
            "perm_indices", precompute_d6_indices(kernel_size), persistent=False
        )
        init_hex_weights_(self.weight, self.filter_mask, in_channels, num_groups=1)

    def build_kernel(self) -> torch.Tensor:
        """
        Generate the D6 transformations of the kernel.

        Returns
        -------
        torch.Tensor
            Transformed kernels with shape ``(out_channels, 12, in_channels, K, K)``.
        """
        masked_weight = self.weight * self.filter_mask
        w = masked_weight.view(
            self.out_channels * self.in_channels, self.kernel_size, self.kernel_size
        )
        tw = fast_d6_all_transforms(w, self.perm_indices)
        tw = tw.view(
            12, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return tw.transpose(0, 1)


class EquivariantCategoricalProjector(nn.Module):
    """
    Embed the categorical values of all 13 PAC grid channels.

    A single embedding table is used across the framework, functional group, and atom-type channels.

    Parameters
    ----------
    max_classes : int, default 30
        Number of categorical values in the embedding table.
    embedding_dim : int, default 16
        Dimension of each embedding vector.
    """

    def __init__(self, max_classes: int = 30, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Index 0 is empty space and stays at zero
        self.embedding = nn.Embedding(
            num_embeddings=max_classes, embedding_dim=embedding_dim, padding_idx=0
        )

        with torch.no_grad():
            nn.init.normal_(self.embedding.weight[1:], mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of categorical PAC grids into embedded feature maps.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 13, grid_size, grid_size)
            Categorical PAC grids.

        Returns
        -------
        torch.Tensor
            Embedded grids with shape ``(batch_size, 13 * embedding_dim, grid_size, grid_size)``. The feature dimension
            contains the framework embedding followed by the six functional group embeddings and the six atom-type
            embeddings.
        """
        B, C, H, W = x.shape
        E = self.embedding_dim

        ch_fw = x[:, 0:1, :, :]  # Framework (B, 1, H, W)
        ch_fn = x[:, 1:7, :, :]  # Functional (B, 6, H, W)
        ch_at = x[:, 7:13, :, :]  # Atom types (B, 6, H, W)

        # 1 embedding table shared across 6 directions
        emb_fw = self.embedding(ch_fw.long())  # (B, 1, H, W, E)

        emb_fn = self.embedding(ch_fn.long().view(B * 6, H, W))  # (B * 6, H, W, E)
        emb_fn = emb_fn.view(B, 6, H, W, E)

        emb_at = self.embedding(ch_at.long().view(B * 6, H, W))  # (B * 6, H, W, E)
        emb_at = emb_at.view(B, 6, H, W, E)

        emb_fw = emb_fw.permute(0, 1, 4, 2, 3)  # (B, 1, E, H, W)
        emb_fn = emb_fn.permute(0, 1, 4, 2, 3)  # (B, 6, E, H, W)
        emb_at = emb_at.permute(0, 1, 4, 2, 3)  # (B, 6, E, H, W)

        emb_fw = emb_fw.permute(0, 2, 1, 3, 4)  # (B, E, 1, H, W)
        emb_fn = emb_fn.permute(0, 2, 1, 3, 4)  # (B, E, 6, H, W)
        emb_at = emb_at.permute(0, 2, 1, 3, 4)  # (B, E, 6, H, W)

        fw_flat = emb_fw.reshape(B, E * 1, H, W)
        fn_flat = emb_fn.reshape(B, E * 6, H, W)
        at_flat = emb_at.reshape(B, E * 6, H, W)

        # Re-concatenate into (B, E * 13, H, W)
        return torch.cat([fw_flat, fn_flat, at_flat], dim=1)


class VectorLiftingKernel(FrozenKernelCache):
    """
    Construct a D6 lifting kernel for functional-group and atom-type channels.

    Each spatial rotation or reflection is accompanied by the corresponding permutation of the six vertex directions.
    This keeps the directional functional group and atom-type channels consistent with the transformed lattice.

    Parameters
    ----------
    in_features : int
        Embedding dimension for each functional-group or atom-type channel.
    out_channels : int
        Number of output channels per group element.
    kernel_size : int
        Side length of the square array containing the hexagonal kernel. The value must be odd.
    """

    def __init__(self, in_features: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_features, 6, kernel_size, kernel_size)
        )
        self.register_buffer(
            "filter_mask", hexagon_axial(self.kernel_size), persistent=False
        )
        self.register_buffer(
            "perm_indices", precompute_d6_indices(self.kernel_size), persistent=False
        )

        # 6 directions * in_features
        init_hex_weights_(
            self.weight, self.filter_mask, in_channels=in_features * 6, num_groups=1
        )

        rotation_indices = torch.zeros((12, 6), dtype=torch.long)
        for g in range(12):
            for c in range(6):
                if g < 6:
                    rotation_indices[g, c] = (c + g) % 6
                else:
                    k = g - 6
                    rotation_indices[g, c] = (-c - k) % 6

        # Broadcast gathered index across new feature dimension
        idx_expanded = (
            rotation_indices.unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                12,
                self.out_channels,
                self.in_features,
                6,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.register_buffer("gather_idx", idx_expanded, persistent=False)

    def build_kernel(self) -> torch.Tensor:
        """
        Generate the spatial and directional transformations of the kernel.

        Returns
        -------
        torch.Tensor
            Transformed kernels with shape ``(out_channels, 12, in_features * 6, K, K)``.
        """
        masked_w = self.weight * self.filter_mask
        w_flat = masked_w.view(
            self.out_channels * self.in_features * 6, self.kernel_size, self.kernel_size
        )

        all_rot = fast_d6_all_transforms(w_flat, self.perm_indices)
        all_rot = all_rot.view(
            12,
            self.out_channels,
            self.in_features,
            6,
            self.kernel_size,
            self.kernel_size,
        )

        W_gathered = torch.gather(all_rot, dim=3, index=self.gather_idx)
        W_gathered = W_gathered.view(
            12,
            self.out_channels,
            self.in_features * 6,
            self.kernel_size,
            self.kernel_size,
        )

        W_final = W_gathered.permute(1, 0, 2, 3, 4).contiguous()
        return W_final


class HexMixedInputLayer(nn.Module):
    """
    Lift an embedded PAC grid into a PAC feature map.

    The non-directional framework embedding is convolved with a `D6LiftingKernel`. The 2 six-directional blocks for
    functional groups and atom types are convolved with separate `VectorLiftingKernel` instances. The 3 results are
    added within a single convolution and retain an explicit 12-element group dimension.

    Parameters
    ----------
    in_features : int
        Dimension of the embedding vector for each input channel.
    out_channels : int
        Number of output channels per group element.
    kernel_size : int
        Side length of the square array containing the hexagonal kernel. The value must be odd.
    padding : int, default 2
        Width of the circular padding applied along both spatial dimensions.
    """

    def __init__(
        self, in_features: int, out_channels: int, kernel_size: int, padding: int = 2
    ) -> None:
        super().__init__()
        self.padding = padding
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.scalar_kernel = D6LiftingKernel(
            kernel_size, in_channels=in_features, out_channels=out_channels
        )
        self.vec_a_kernel = VectorLiftingKernel(in_features, out_channels, kernel_size)
        self.vec_b_kernel = VectorLiftingKernel(in_features, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the mixed-input lifting convolution.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, in_features * 13, H, W)
            Embedded PAC grids.

        Returns
        -------
        torch.Tensor of shape (batch_size, out_channels, 12, H, W)
            Feature maps with an explicit D6 group dimension.
        """
        w_s = self.scalar_kernel.sample()  # Shape: (out, 12, in_f, K, K)
        w_a = self.vec_a_kernel.sample()  # Shape: (out, 12, in_f * 6, K, K)
        w_b = self.vec_b_kernel.sample()  # Shape: (out, 12, in_f * 6, K, K)

        # Total channels concatenated = in_f + (in_f * 6) + (in_f * 6) = in_f * 13
        w_combined = torch.cat([w_s, w_a, w_b], dim=2)

        w_combined_flat = w_combined.view(
            self.out_channels * 12,
            self.in_features * 13,
            self.kernel_size,
            self.kernel_size,
        )
        x_padded = F.pad(
            x,
            pad=(self.padding, self.padding, self.padding, self.padding),
            mode="circular",
        )
        y = F.conv2d(x_padded, w_combined_flat, padding=0)

        B, _, H, W = y.shape
        return y.view(B, self.out_channels, 12, H, W)


class D6GroupKernel(FrozenKernelCache):
    """
    Construct the kernel for a regular D6 group convolution.

    A single learnable kernel is transformed under each D6 element. The spatial kernel is rotated or reflected while its
    indices along the input group dimension are permuted by ``h -> k^-1 h``.

    Parameters
    ----------
    in_channels : int
        Number of input channels per group element.
    out_channels : int
        Number of output channels per group element.
    kernel_size : int
        Side length of the square array containing the hexagonal kernel. The value must be odd.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, 12, kernel_size, kernel_size)
        )
        self.register_buffer(
            "filter_mask", hexagon_axial(kernel_size), persistent=False
        )
        self.register_buffer(
            "perm_indices", precompute_d6_indices(self.kernel_size), persistent=False
        )

        init_hex_weights_(self.weight, self.filter_mask, in_channels, num_groups=12)

        perm12 = torch.zeros((12, 12), dtype=torch.long)
        for k in range(12):
            for h in range(12):
                if k < 6:
                    perm12[k, h] = (h - k) % 6 if h < 6 else 6 + (h - 6 - k) % 6
                else:
                    m = k - 6
                    perm12[k, h] = 6 + (m - h) % 6 if h < 6 else (m - (h - 6)) % 6

        idx_expanded = (
            perm12.unsqueeze(1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                12,
                self.out_channels * self.in_channels,
                12,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.register_buffer("gather_idx", idx_expanded, persistent=False)

    def build_kernel(self) -> torch.Tensor:
        """
        Generate the D6 transformations of the group kernel.

        Returns
        -------
        torch.Tensor
            Transformed kernels with shape ``(out_channels, 12, in_channels, 12, K, K)``.
        """
        masked_weight = self.weight * self.filter_mask
        w_flat = masked_weight.view(
            self.out_channels * self.in_channels * 12,
            self.kernel_size,
            self.kernel_size,
        )

        all_rot = fast_d6_all_transforms(w_flat, self.perm_indices)
        all_rot = all_rot.view(
            12,
            self.out_channels * self.in_channels,
            12,
            self.kernel_size,
            self.kernel_size,
        )

        W_gathered = torch.gather(all_rot, dim=2, index=self.gather_idx)
        W_final = W_gathered.view(
            12,
            self.out_channels,
            self.in_channels,
            12,
            self.kernel_size,
            self.kernel_size,
        )
        return W_final.transpose(0, 1)


class GroupConvolution(nn.Module):
    """
    Apply a regular D6 group convolution.

    Circular padding is used along the spatial dimensions because the PAC grid has periodic boundaries.

    Parameters
    ----------
    in_channels : int
        Number of input channels per group element.
    out_channels : int
        Number of output channels per group element.
    kernel_size : int
        Side length of the square array containing the hexagonal kernel. The value must be odd.
    padding : int
        Width of the circular padding applied along both spatial dimensions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.kernel = D6GroupKernel(in_channels, out_channels, kernel_size)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolve a feature map over its spatial dimensions and group dimension.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, in_channels, 12, H, W)
            Input feature maps.

        Returns
        -------
        torch.Tensor of shape (batch, out_channels, 12, H, W)
            Output feature maps.
        """
        batch, in_c, in_g, ny, nx = x.shape
        x = x.reshape(batch, in_c * in_g, ny, nx)

        tw = self.kernel.sample()
        weight = tw.reshape(
            self.kernel.out_channels * 12,
            self.kernel.in_channels * 12,
            self.kernel.kernel_size,
            self.kernel.kernel_size,
        )

        x_padded = F.pad(
            x,
            pad=(self.padding, self.padding, self.padding, self.padding),
            mode="circular",
        )
        y = F.conv2d(x_padded, weight, padding=0)

        batch, _, ny_out, nx_out = y.shape
        return y.view(batch, self.kernel.out_channels, 12, ny_out, nx_out)


class GroupEquivariantCNN(nn.Module):
    """
    Predict molecular properties with a D6-equivariant convolutional network.

    The network ends in separate pooling heads for intensive and extensive properties. HOMO and LUMO energies and the
    HOMO-LUMO gap use spatial max pooling, whereas heat of formation uses spatial sum pooling. Averaging over the group
    dimension makes both representations invariant under D6.

    Parameters
    ----------
    property_types : list of str
        Whether each target is intensive or extensive. Every value must be either `'intensive'` or `'extensive'`.
    kernel_size : int
        Side length of the square array containing each hexagonal kernel.
    num_hidden : int
        Number of group convolution layers after the lifting layer.
    hidden_channels : int
        Number of hidden channels per group element.
    embedding_dim : int, default 16
        Dimension of each categorical embedding vector.
    """

    def __init__(
        self,
        property_types: list[str],
        kernel_size: int,
        num_hidden: int,
        hidden_channels: int,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.property_types = property_types
        self.num_intensive: int = property_types.count("intensive")
        self.num_extensive: int = property_types.count("extensive")

        self.projector = EquivariantCategoricalProjector(
            max_classes=30, embedding_dim=embedding_dim
        )
        self.lifting_conv = HexMixedInputLayer(
            in_features=embedding_dim,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.lift_bn = nn.BatchNorm3d(hidden_channels)
        self.gbns = nn.ModuleList()
        self.gconvs = nn.ModuleList()

        for _ in range(num_hidden):
            self.gconvs.append(
                GroupConvolution(
                    hidden_channels, hidden_channels, kernel_size, padding=2
                )
            )
            self.gbns.append(nn.BatchNorm3d(hidden_channels))

        if self.num_intensive > 0:
            self.electronic_head: nn.Module = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, self.num_intensive),
            )

        if self.num_extensive > 0:
            self.thermodynamic_head: nn.Module = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, self.num_extensive),
            )

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the invariant features used by the prediction heads.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 13, H, W)
            Input PAC grids.

        Returns
        -------
        tuple of torch.Tensor
            Max-pooled features followed by sum-pooled features. Each tensor has shape ``(batch_size,
            hidden_channels)``.
        """
        x = self.projector(x)
        x = self.lifting_conv(x)
        x = self.lift_bn(x)
        x = F.silu(x, inplace=True)

        for gconv, gbn in zip(self.gconvs, self.gbns):
            x = gconv(x)
            x = gbn(x)
            x = F.silu(x, inplace=True)

        x_sp_max = x.amax(dim=(-2, -1))
        x_sp_sum = x.sum(dim=(-2, -1))

        x_max_mean = x_sp_max.mean(dim=2)
        x_sum_mean = x_sp_sum.mean(dim=2)

        return x_max_mean, x_sum_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict all requested properties for a batch of PAC grids.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 13, H, W)
            Input PAC grids.

        Returns
        -------
        torch.Tensor of shape (batch_size, num_properties)
            Predicted properties in the same order as `property_types`.

        Raises
        ------
        RuntimeError
            Raised when `property_types` contains a property type that does not have a prediction head. This happens
            only when `property_types` changes after the model is constructed.
        """
        x_max_mean, x_sum_mean = self.extract_features(x)

        int_preds = self.electronic_head(x_max_mean) if self.num_intensive > 0 else None
        ext_preds = (
            self.thermodynamic_head(x_sum_mean) if self.num_extensive > 0 else None
        )

        out = []
        i_idx, e_idx = 0, 0

        for p in self.property_types:
            if p == "intensive":
                if int_preds is None:
                    raise RuntimeError(
                        "The model does not have a prediction head for intensive properties."
                    )
                out.append(int_preds[:, i_idx : i_idx + 1])
                i_idx += 1
            elif p == "extensive":
                if ext_preds is None:
                    raise RuntimeError(
                        "The model does not have a prediction head for extensive properties."
                    )
                out.append(ext_preds[:, e_idx : e_idx + 1])
                e_idx += 1

        return torch.cat(out, dim=1)
