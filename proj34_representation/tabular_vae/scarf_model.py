"""
scarf_model.py
==============

ONLY RELEVANT FOR PROJECT 4

SCARF = "Self-supervised Contrastive learning using Random Feature corruption"
(Bahri et al., 2021, https://arxiv.org/abs/2106.15147).

Adapted from https://github.com/clabrugere/pytorch-scarf (MIT License,
Copyright (c) 2022 Clement Labrugere), simplified for teaching and changed to work
with the mixed one-hot / encoded matrix made by TabularPreprocessor.

The idea, in three steps
------------------------
  1. Take a row x (the "anchor").
  2. Make a corrupted copy of it: for each VARIABLE, with probability `corruption_rate`,
     replace its value by the value of that variable from a random other row.
  3. Train an encoder so that the embedding of x and the embedding of its corrupted copy
     are SIMILAR, while embeddings of different rows in the batch are DIFFERENT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


def variable_blocks(preprocessor):
    """
    Start and end column of every ORIGINAL VARIABLE in the preprocessed matrix.

    A categorical variable is a whole one-hot block, and a continuous variable is a block
    of m >= 1 numbers. We corrupt whole blocks, never single columns, otherwise we would
    create impossible rows (e.g. a one-hot block with two ones).
    """
    widths = list(preprocessor.cat_dims_) + list(preprocessor.cont_dims_)
    blocks, start = [], 0
    for width in widths:
        blocks.append((start, start + width))
        start += width
    return blocks


def corrupt(x, blocks, corruption_rate):
    """
    Return a corrupted copy of the batch x.

    For every row and every variable: with probability `corruption_rate`, the variable's
    value is replaced by the value of the same variable from a random row of the batch.
    """
    batch_size = x.size(0)
    x_corrupted = x.clone()

    for start, end in blocks:
        replace = torch.rand(batch_size, 1) < corruption_rate     # which rows get replaced?
        donors = x[torch.randperm(batch_size), start:end]         # this variable, from random rows
        x_corrupted[:, start:end] = torch.where(replace, donors, x[:, start:end])

    return x_corrupted


class SCARF(nn.Module):
    def __init__(self, input_dim, blocks, emb_dim=16, hidden_dim=64, corruption_rate=0.6):
        """
        input_dim        width of the preprocessed matrix (preprocessor.input_dim_)
        blocks           from variable_blocks(preprocessor)
        emb_dim          size of the embedding you will use later
        hidden_dim       size of the hidden layers
        corruption_rate  share of variables replaced in the corrupted copy
        """
        super().__init__()
        self.blocks = blocks
        self.corruption_rate = corruption_rate

        # The encoder is the part we keep: row -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        # The head is only used during pre-training (as in SimCLR); we throw it away afterwards.
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x_corrupted = corrupt(x, self.blocks, self.corruption_rate)
        z_anchor = self.head(self.encoder(x))
        z_corrupted = self.head(self.encoder(x_corrupted))
        return z_anchor, z_corrupted

    @torch.no_grad()
    def embed(self, x):
        """The embedding of each row (call model.eval() first)."""
        return self.encoder(x)


def nt_xent(z_anchor, z_corrupted, temperature=1.0):
    """
    Contrastive loss (NT-Xent, from SimCLR). Returns (loss, accuracy).

    Take the 2B embeddings of a batch (B anchors + B corrupted copies). For each one, the
    "right answer" is its partner (anchor <-> its own corrupted copy); the other 2B - 2
    embeddings are the wrong answers. The loss is a cross-entropy over cosine similarities.

    accuracy = how often the most similar embedding really is the partner
               (chance level is 1 / (2B - 1)).
    """
    batch_size = z_anchor.size(0)

    z = F.normalize(torch.cat([z_anchor, z_corrupted]), dim=1)      # length 1, so dot product = cosine
    similarity = z @ z.T / temperature                              # (2B, 2B)

    # A row must not be compared with itself.
    self_pairs = torch.eye(2 * batch_size, dtype=torch.bool)
    similarity = similarity.masked_fill(self_pairs, float("-inf"))

    # Row i (an anchor) has partner i + B, and row i + B has partner i.
    partner = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)])

    loss = F.cross_entropy(similarity, partner)
    accuracy = (similarity.argmax(dim=1) == partner).float().mean().item()
    return loss, accuracy


# -------------------
## Used in Week 4 to generate synthetic nuisance features for the SCARF pretraining task.

def make_nuisance_features(
    n_samples,
    nuisance_types,
    seed=0,
    high_variance_std=5.0,
    n_categories=50,
):
    """
    Generate 1-2 synthetic nuisance features.

    Parameters
    ----------
    n_samples : int
        Number of rows/patients.

    nuisance_types : list[str]
        Choose up to 2 from:
            "high_variance"
            "high_cardinality"
            "skewed_categorical"

        Use [] for the control condition.

    seed : int
        Random seed.

    high_variance_std : float
        Standard deviation of high-variance numerical nuisance features.

    n_categories : int
        Number of categories for high-cardinality nuisance features.

    Returns
    -------
    N : pd.DataFrame
        Synthetic nuisance variables.
    """
    rng = np.random.default_rng(seed)

    if len(nuisance_types) > 2:
        raise ValueError("Choose at most 2 nuisance features.")

    N = pd.DataFrame(index=np.arange(n_samples))

    for i, nuisance_type in enumerate(nuisance_types):

        # 1. High-variance numerical nuisance
        if nuisance_type == "high_variance":
            N[f"N{i+1}_high_variance"] = rng.normal(
                loc=0,
                scale=high_variance_std,
                size=n_samples
            )

        # 2. High-cardinality categorical nuisance
        elif nuisance_type == "high_cardinality":
            values = rng.integers(
                low=0,
                high=n_categories,
                size=n_samples
            )

            N[f"N{i+1}_high_cardinality"] = pd.Categorical(values)

        # 3. Highly skewed categorical nuisance
        elif nuisance_type == "skewed_categorical":
            categories = [0, 1, 2, 3]

            # One category dominates
            probabilities = [0.75, 0.15, 0.07, 0.03]

            values = rng.choice(
                categories,
                size=n_samples,
                p=probabilities
            )

            N[f"N{i+1}_skewed"] = pd.Categorical(values)

        else:
            raise ValueError(
                f"Unknown nuisance type: {nuisance_type}"
            )

    return N
## This was made using help from Claude. 