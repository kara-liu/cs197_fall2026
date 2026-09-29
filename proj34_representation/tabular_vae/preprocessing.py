"""
preprocessing.py
================

Turns a pandas DataFrame into design matrix to input into the VAE.
Also turns matrices produced by the VAE back into a DataFrame.

Every row of the matrix has this layout (used everywhere else too!):

    [ one-hot(cat_1) | ... | one-hot(cat_M) | enc(cont_1) | ... | enc(cont_K) ]
      (categorical part comes first)           (continuous part comes last)

  - each categorical variable  -> a one-hot block (width = number of categories)
  - each continuous variable   -> preprocessed numerical values; for proj 3 this is a block of m >= 1 dim made by encoder; 

Good habit: fit the preprocessor on the TRAINING data only. Otherwise information
from the validation set leaks into training ("data leakage").
"""

import copy

import numpy as np
import pandas as pd

from .encoders import StandardEncoder


class TabularPreprocessor:
    def __init__(self, cat_cols, cont_cols, cont_encoders=None):
        """
        cat_cols       list of categorical column names
        cont_cols      list of continuous column names
        cont_encoders  optional dict {column name: encoder}. 
                        Default: use StandardEncoder.
        """
        self.cat_cols = list(cat_cols)
        self.cont_cols = list(cont_cols)
        self.cont_encoders = cont_encoders or {}

    def fit(self, df):
        # Categorical: remember which categories exist in the training data.
        self.categories_ = {c: sorted(df[c].unique(), key=str) for c in self.cat_cols}
        self.cat_dims_ = [len(self.categories_[c]) for c in self.cat_cols]

        # Continuous: every column gets its own encoder, fitted on that column.
        self.encoders_ = {}
        for c in self.cont_cols:
            encoder = copy.deepcopy(self.cont_encoders.get(c, StandardEncoder()))
            self.encoders_[c] = encoder.fit(df[c].to_numpy())
        self.cont_dims_ = [self.encoders_[c].output_dim_ for c in self.cont_cols]

        # Sizes we will need later.
        self.cat_total_ = sum(self.cat_dims_)
        self.cont_total_ = sum(self.cont_dims_)
        self.input_dim_ = self.cat_total_ + self.cont_total_
        return self

    def drop_unseen(self, df):
        """Remove rows that contain a category the training data never had."""
        keep = np.ones(len(df), dtype=bool)
        for c in self.cat_cols:
            keep &= df[c].isin(self.categories_[c]).to_numpy()
        if not keep.all():
            print(f"Dropped {(~keep).sum()} row(s) with unseen categories")
        return df[keep].reset_index(drop=True)

    def transform(self, df):
        """DataFrame -> float32 numpy array of shape (N, input_dim_)."""
        if df[self.cat_cols + self.cont_cols].isna().any().any():
            raise ValueError("Missing values found. Drop or fill them first.")

        blocks = []

        for c in self.cat_cols:  # categorical -> one-hot
            index = {category: i for i, category in enumerate(self.categories_[c])}
            codes = df[c].map(index)
            if codes.isna().any():
                raise ValueError(f"Column '{c}' has unseen categories. Use drop_unseen(df) first.")
            blocks.append(np.eye(len(index))[codes.to_numpy(dtype=int)])

        for c in self.cont_cols:  # continuous -> m numbers each
            blocks.append(self.encoders_[c].transform(df[c].to_numpy()))

        return np.hstack(blocks).astype(np.float32)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def inverse_transform(self, matrix):
        """Numpy array of shape (N, input_dim_) -> DataFrame with the original columns.
        For example, if the variables have been one hot encoded and Gaussian normalized,
        this will return the original categories and unscaled values that are 
        interpretable by humans / the raw format of eICU!
        """
        matrix = np.asarray(matrix, dtype=float)
        columns, start = {}, 0

        for c, width in zip(self.cat_cols, self.cat_dims_):
            block = matrix[:, start:start + width]
            categories = np.array(self.categories_[c], dtype=object)
            columns[c] = categories[block.argmax(axis=1)]   # pick the most likely category
            start += width

        for c, width in zip(self.cont_cols, self.cont_dims_):
            block = matrix[:, start:start + width]
            columns[c] = self.encoders_[c].inverse_transform(block)
            start += width

        return pd.DataFrame(columns)
    
    def get_feature_names(self):
        names = []
        for c in self.cat_cols:
            names += [f"{c}={category}" for category in self.categories_[c]]
        for c, width in zip(self.cont_cols, self.cont_dims_):
            if width == 1:
                names.append(c)
            else:
                names += [f"{c}_{i}" for i in range(width)]
        return names
## This was generated with help from Claude.
