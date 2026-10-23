"""
Prepare tokenised SMILES strings for the bidirectional LSTM baseline.
"""

import re
from collections.abc import Callable, Sequence
from functools import partial

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from dataset.common import *
from pac import PAC

SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)


def tokenize_smiles(smiles: str) -> list[str]:
    """
    Split a SMILES string into atom-level tokens.

    The tokens are the matches of `SMILES_TOKEN_PATTERN` in the order of the string, the atom-level tokens of the
    Molecular Transformer (Schwaller et al., ACS Cent. Sci. 2019, 5, 1572). A bracket atom such as ``[nH]``, a 2-letter
    element symbol such as ``Cl`` or ``Br``, and a 2-digit ring-closure number such as ``%10`` are each 1 token.

    Parameters
    ----------
    smiles : str
        SMILES string to split.

    Returns
    -------
    list of str
        Tokens in the order of the SMILES string.

    Raises
    ------
    ValueError
        Raised when a character is not part of any token.
    """
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(
            f"SMILES string {smiles!r} has a character that is not part of any token."
        )
    return tokens


class SMILESTokenizer:
    """
    Build an atom-level token vocabulary from SMILES strings and encode SMILES strings as token indices.

    The vocabulary is the set of the tokens of `tokenize_smiles` in `smiles_list`. The tokens are sorted, and token i of
    the sorted list has the index i + 1. Index 0 is the padding index.

    Parameters
    ----------
    smiles_list : sequence of str
        SMILES strings of the vocabulary.

    Attributes
    ----------
    token_to_int : dict of str to int
        Index of each token.
    vocab_size : int
        Number of tokens plus 1 for the padding index.
    """

    def __init__(self, smiles_list: Sequence[str]) -> None:
        tokens = set()
        for smiles in smiles_list:
            tokens.update(tokenize_smiles(smiles))
        self.token_to_int = {
            token: index + 1 for index, token in enumerate(sorted(tokens))
        }
        self.vocab_size = len(self.token_to_int) + 1

    def encode(self, smiles: str) -> torch.Tensor:
        """
        Convert a SMILES string into a tensor of token indices.

        Parameters
        ----------
        smiles : str
            SMILES string to encode.

        Returns
        -------
        torch.Tensor of shape (num_tokens,)
            Token indices of type torch.long.

        Raises
        ------
        KeyError
            Raised when the SMILES string has a token that is not in the vocabulary.
        ValueError
            Raised by `tokenize_smiles` when a character is not part of any token.
        """
        return torch.tensor(
            [self.token_to_int[token] for token in tokenize_smiles(smiles)],
            dtype=torch.long,
        )


def collate_smiles(
    batch: list[tuple[str, torch.Tensor]], tokenizer: SMILESTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode and pad the SMILES strings of a batch and stack the target values.

    The token sequences are padded at the end with index 0 to the length of the longest sequence of the batch.

    Parameters
    ----------
    batch : list of tuple
        ``(smiles, targets)`` pairs.
    tokenizer : SMILESTokenizer
        Tokeniser with the vocabulary of the dataset.

    Returns
    -------
    token_indices : torch.Tensor of shape (batch_size, max_num_tokens)
        Padded token indices of type torch.long.
    targets : torch.Tensor of shape (batch_size, num_targets)
        Stacked target values.
    """
    sequences = [tokenizer.encode(item[0]) for item in batch]
    targets = [item[1] for item in batch]
    padded_seqs = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    return padded_seqs, torch.stack(targets)


def get_collate_fn(
    tokenizer: SMILESTokenizer,
) -> Callable[[list[tuple[str, torch.Tensor]]], tuple[torch.Tensor, torch.Tensor]]:
    """
    Create the collate function of a data loader for batches of SMILES strings.

    The function returns `collate_smiles` with `tokenizer` bound by `functools.partial`. The result is picklable, which
    the worker processes of a data loader require when the processes are started by spawn.

    Parameters
    ----------
    tokenizer : SMILESTokenizer
        Tokeniser with the vocabulary of the dataset.

    Returns
    -------
    callable
        Function of a list of ``(smiles, targets)`` pairs that returns the padded token indices and the stacked target
        values.
    """
    return partial(collate_smiles, tokenizer=tokenizer)


def process_smiles(row: tuple[str, object, str]) -> tuple[str, object, PAC]:
    """
    Reconstruct the PAC grid of 1 database row with the SMILES string and the target value.

    Parameters
    ----------
    row : tuple
        ``(serialization, target, smiles)`` from a database query.

    Returns
    -------
    tuple of (str, object, PAC)
        ``(smiles, target, pac)``, with the unchanged target value and the PAC grid of `PAC.deserialize`.
    """
    serialization, bg, smiles = row
    pac = PAC.deserialize(serialization)
    return smiles, bg, pac


class PAC5SMILESDataset(Dataset):
    """
    Load the SMILES strings and the target values of a partition.

    Rows with a missing value in `properties` are excluded. Each item is the SMILES string and the target values as
    float32, and the collate function of `get_collate_fn` encodes the SMILES strings of each batch.

    Parameters
    ----------
    csv_path : str
        Partition CSV file with the column ``smiles`` and the target columns.
    properties : list of str
        Columns of the target values, in the order of the targets of each item.

    Attributes
    ----------
    data : list of str
        SMILES strings in the order of the CSV file.
    targets : torch.Tensor of shape (num_molecules, num_properties)
        Target values as float32.
    """

    def __init__(self, csv_path: str, properties: list[str]) -> None:
        df = pd.read_csv(csv_path).dropna(subset=properties)
        self.data = df["smiles"].tolist()
        self.targets = torch.tensor(df[properties].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        return self.data[index], self.targets[index]
