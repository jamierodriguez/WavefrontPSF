#!/usr/bin/env python
"""
File: psf_interpolator.py
Author: Chris Davis
Description: Module that takes focal plane coordinates and other relevant data
             and returns some kind of representation of the basis.

TODO: I broke something here in the kNN_Interpolator?

"""

from pandas import DataFrame
from numpy import vstack

class PSF_Interpolator(object):
    """Class that returns some sort of PSF representation.

    Attributes
    ----------

    x_keys : list of strings
        Keys that an interpolator takes to create output
    y_keys : list of strings
        Names of output keys

    Methods
    -------
    interpolate
        Returns the psf at some location. By default your basis is just
        whatever you put in.

    """

    def __init__(self, y_keys=[], x_keys=[], **kwargs):
        self.y_keys = y_keys
        self.x_keys = x_keys

    def check_data_for_keys(self, data):
        # very simple: just try to access
        can_do_x = True
        for key in self.x_keys:
            try:
                #print(key, data[key])
                data[key]
            except E:
                print('Problems with', key, E)
                can_do_x = False
        can_do_y = True
        for key in self.y_keys:
            try:
                #print(key, data[key])
                data[key]
            except E:
                print('Problems with', key, E)
                can_do_y = False
        return can_do_x, can_do_y

    def interpolate(self, X, **kwargs):
        interpolated = {}
        for key in self.x_keys:
            interpolated[key] = X[key]
        # for key in self.y_keys:
        #     interpolated[key] = func[key](X[self.x_keys])
        interpolated = DataFrame(interpolated)

        return interpolated

    def __call__(self, X, **kwargs):
        return self.interpolate(X, **kwargs)

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
        super(kNN_Interpolator, self).__init__(y_keys=y_keys, x_keys=x_keys, **kwargs)

        from sklearn.neighbors import KNeighborsRegressor

        knn_args = {'n_neighbors': 45,
                    'weights': 'uniform',
                    'p': 1,
                    }

        knn_args.update(kwargs)

        self.data = data

        # train the interpolant for each key
        # since training is just copying the data... this should be quick
        self.knn = {}
        for key in self.y_keys:
            self.knn[key] = KNeighborsRegressor(**knn_args)
            self.knn[key].fit(self.data[x_keys], self.data[key])

    def interpolate(self, X, force_interpolation=True, **kwargs):
        # force_interpolation: if false, if ALL interpolant variables already
        # present in X, then do not actually create new interpolation
        do_interpolation = force_interpolation
        for key in self.y_keys:
            # if a y_key is not present, force the interpolation
            if key not in X:
                do_interpolation = True

        if do_interpolation:
            interpolated = {}
            # for key in X.keys():#self.x_keys:
            #     interpolated[key] = X[key]
            for key in self.y_keys:
                interpolated[key] = self.knn[key].predict(X[self.x_keys])
            interpolated = DataFrame(interpolated)
            # want to overwrite any preexisting x
            X = interpolated.combine_first(X)
        else:
            # do nothing
            pass

        return X

    def __call__(self, X, **kwargs):
        return self.interpolate(X, **kwargs)


