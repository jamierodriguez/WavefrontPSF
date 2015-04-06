#!/usr/bin/env python
"""
File: psf_interpolator.py
Author: Chris Davis
Description: Module that takes focal plane coordinates and other relevant data
             and returns some kind of representation of the basis.

"""

from pandas import DataFrame
from numpy import vstack

class PSF_Interpolator(object):
    """Class that returns some sort of PSF representation.

    Attributes
    ----------

    Methods
    -------
    interpolate
        Returns the psf at some location. By default your basis is just
        whatever you put in.

    """

    def interpolate(self, x, y, **kwargs):
        # by default just return the params back!
        return x, y, kwargs

    def __call__(self, x, y, **kwargs):
        return self.interpolate(x, y, **kwargs)

class kNN_Interpolator(PSF_Interpolator):
    """Impliment my own version of Aaron's base inverse distance weighted
    interpolation. Using a Digestor to create your data, then interpolate using
    k nearest neighbors.

    """

    def __init__(self, data,
                 y_keys=['z{0}'.format(i) for i in range(4, 12)],
                 x_keys=['x', 'y'], **kwargs):
        """
        x_keys, y_keys : lists of strings
            The keys which we want to use as input (x) and output (y).
        """
        from sklearn.neighbors import KNeighborsRegressor

        knn_args = {'n_neighbors': 45,
                    'weights': 'uniform',
                    'p': 1,
                    }

        knn_args.update(kwargs)
        self.x_keys = x_keys
        self.y_keys = y_keys

        self.data = data

        # train the interpolant for each key
        # since training is just copying the data... this should be quick
        self.knn = {}
        for key in self.y_keys:
            self.knn[key] = KNeighborsRegressor(**knn_args)
            self.knn[key].fit(self.data[x_keys], self.data[key])

    def interpolate(self, inputs, **kwargs):
        interpolated = {}
        for key in self.x_keys:
            interpolated[key] = inputs[key]
        for key in self.y_keys:
            interpolated[key] = self.knn[key].predict(inputs[self.x_keys])
        interpolated = DataFrame(interpolated)

        return interpolated
