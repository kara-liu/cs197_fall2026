"""
encoders.py
===========

Numerical preprocessing, also called encoding. 
Only relevant for Project 3!

Every encoder has the same four parts (the "interface"):

    encoder.fit(x)                  learn from a 1-D array of training values
    encoder.transform(x)            array of shape (N,)   ->  array of shape (N, m)
    encoder.inverse_transform(z)    array of shape (N, m) ->  array of shape (N,)
"""

import numpy as np


class StandardEncoder:
    """x  ->  (x - mean) / std.   Output has m = 1 dimension."""
    """
    This is equivalent to sklearn StandardScaler
    """
    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean()
        self.std_ = x.std() if x.std() > 1e-8 else 1.0   # avoid dividing by zero
        self.output_dim_ = 1
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, z):
        z = np.asarray(z, dtype=float)
        return z[:, 0] * self.std_ + self.mean_


# TODO: add the rest from Week 2!!

