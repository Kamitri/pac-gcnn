"""
Conventional 2D CNN baseline.

Uses the same hexagonal kernel masking, weight initialisation, and circular padding as
`gcnn.base_model.GroupEquivariantCNN`, but without group convolutions over D6.
"""

import torch
import torch.nn.functional as F
from torch import nn

from gcnn.base_model import init_hex_weights_
from gcnn.transform import hexagon_axial


class CategoricalProjector(nn.Module):
    """
    Embed the categorical values of all 13 PAC grid channels into a dense feature map.

    Parameters
    ----------
    max_classes : int, default 30
        Number of categorical values in the shared embedding table.
    embedding_dim : int, default 16
        Dimension of each embedding vector.
    """

    def __init__(self, max_classes: int = 30, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=max_classes, embedding_dim=embedding_dim, padding_idx=0
        )
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight[1:], mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert categorical PAC grids into embedded feature maps.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 13, grid_size, grid_size)
            Categorical PAC grids.

        Returns
        -------
        torch.Tensor of shape (batch_size, 13 * embedding_dim, grid_size, grid_size)
            Embedded feature maps.
        """
        B, C, H, W = x.shape

        emb = self.embedding(x.long())

        emb = emb.permute(0, 1, 4, 2, 3).contiguous()

        return emb.view(B, C * self.embedding_dim, H, W)


class HexagonalConvolution(nn.Module):
    """
    Apply a 2D convolution with a hexagonal kernel mask and circular padding.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Side length of the square array containing the hexagonal kernel. The value must be odd.
    padding : int
        Width of circular padding applied along both spatial dimensions.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.padding = padding
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.register_buffer(
            "filter_mask", hexagon_axial(kernel_size), persistent=False
        )
        init_hex_weights_(self.weight, self.filter_mask, in_channels, num_groups=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Convolve feature maps with the masked hexagonal kernel.

        Parameters
        ----------
        feature_map : torch.Tensor of shape (batch_size, in_channels, height, width)
            Input feature maps.

        Returns
        -------
        torch.Tensor of shape (batch_size, out_channels, height, width)
            Convolved feature maps.
        """
        masked_weight = self.weight * self.filter_mask
        padded_feature_map = F.pad(
            feature_map,
            pad=(self.padding, self.padding, self.padding, self.padding),
            mode="circular",
        )
        return F.conv2d(padded_feature_map, masked_weight, padding=0)


class CNN(nn.Module):
    """
    Predict molecular properties from PAC grids with a conventional 2D convolutional baseline.

    The categorical PAC grid is embedded, processed by hexagonal convolutions with circular padding, and mapped to
    separate pooling heads for intensive and extensive properties. Intensive properties use spatial max pooling, whereas
    extensive properties use spatial sum pooling.

    Parameters
    ----------
    property_types : list of str
        Whether each target is intensive or extensive. Every value must be either `'intensive'` or `'extensive'`.
    kernel_size : int
        Side length of the square array containing each hexagonal kernel.
    num_hidden : int
        Number of convolutional layers after the input layer.
    hidden_channels : int
        Number of hidden channels in each convolutional layer.
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

        self.projector = CategoricalProjector(
            max_classes=30, embedding_dim=embedding_dim
        )

        self.input_conv = HexagonalConvolution(
            in_channels=13 * embedding_dim,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.input_bn = nn.BatchNorm2d(hidden_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_hidden):
            self.convs.append(
                HexagonalConvolution(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.bns.append(nn.BatchNorm2d(hidden_channels))

        if self.num_intensive > 0:
            self.electronic_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, self.num_intensive),
            )

        if self.num_extensive > 0:
            self.thermodynamic_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, self.num_extensive),
            )

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the pooled features used by the prediction heads.

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
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.silu(x, inplace=True)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.silu(x, inplace=True)

        x_max = x.amax(dim=(-2, -1))
        x_sum = x.sum(dim=(-2, -1))

        return x_max, x_sum

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
        x_max, x_sum = self.extract_features(x)

        int_preds = self.electronic_head(x_max) if self.num_intensive > 0 else None
        ext_preds = self.thermodynamic_head(x_sum) if self.num_extensive > 0 else None

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
