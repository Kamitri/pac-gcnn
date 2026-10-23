"""
Bidirectional LSTM baseline.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMNetwork(nn.Module):
    """
    Predict molecular properties from tokenised SMILES strings with a bidirectional LSTM.

    The embedded tokens are processed by `num_layers` bidirectional LSTM layers as a packed sequence, which excludes the
    padding. The element-wise maximum of the outputs of the last layer over the tokens of a SMILES string is the input
    of the regression head, Linear(2 * `hidden_dim`, `hidden_dim`), LayerNorm, SiLU, and Linear(`hidden_dim`,
    `num_tasks`).

    The input batches are produced by the collate function of `dataset.smiles.get_collate_fn`.

    Parameters
    ----------
    vocab_size : int
        Number of token indices, including the padding index 0.
    embed_dim : int, default 128
        Number of features of each token embedding.
    hidden_dim : int, default 128
        Number of hidden features in each LSTM direction.
    num_layers : int, default 3
        Number of stacked bidirectional LSTM layers.
    dropout : float, default 0
        Dropout probability on the output of each LSTM layer except the last layer.
    num_tasks : int, default 1
        Number of target properties.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        sequence_lengths = (x != 0).sum(dim=1).cpu()
        packed_embedded = pack_padded_sequence(
            embedded, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_embedded)
        lstm_out, _ = pad_packed_sequence(
            packed_output, batch_first=True, padding_value=float("-inf")
        )

        # Padding is -inf, never max
        pooled, _ = torch.max(lstm_out, dim=1)
        return self.fc(pooled)
