#!/usr/bin/env python
"""
File: determine_zernike_moment_relation.py
Author: Chris Davis
Description: Script to perform l1 feature selection to determine the right
polynomial relationship between zernikes, rzero, focal plane coordinates and
the moments observed. The idea is that then we can compare it with the analytic
zernike relations derived a long while ago.


TODO: dig up code from my strongcnn project for iterating over hyperparameters
"""
# load up relevant packages
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.cross_validation import train_test_split

# Define some functions
from WavefrontPSF.donutengine import Zernike_to_Pixel_Interpolator
from WavefrontPSF.psf_evaluator import Moment_Evaluator

# this part taken from https://github.com/mrocklin/multipolyfit/blob/master/multipolyfit/core.py
from numpy import linalg, zeros, ones, hstack, asarray
from itertools import combinations_with_replacement

def basis_vector(n, i):
    """ Return an array like [0, 0, ..., 1, ..., 0, 0]
    >>> from multipolyfit.core import basis_vector
    >>> basis_vector(3, 1)
    array([0, 1, 0])
    >>> basis_vector(5, 4)
    array([0, 0, 0, 0, 1])
    """
    x = zeros(n, dtype=int)
    x[i] = 1
    return x

def as_tall(x):
    """ Turns a row vector into a column vector """
    return x.reshape(x.shape + (1,))

def stack_x(xs, deg):

    num_covariates = xs.shape[1]
    xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype) , xs))

    generators = [basis_vector(num_covariates+1, i)
                  for i in range(num_covariates+1)]

    # All combinations of degrees
    powers = map(sum, combinations_with_replacement(generators, deg))

    # Raise data to specified degree pattern, stack in order
    A = hstack(asarray([as_tall((xs**p).prod(1)) for p in powers]))

    return A, powers

def power_selector(zi, zj, powers, beta):
    powersum = np.sum(powers[0])
    powers_desired = np.zeros(len(powers[0]), dtype=np.int)
    powers_desired[zi - 3] += 1
    powers_desired[zj - 3] += 1
    powers_desired[0] = powersum - np.sum(powers_desired)
    ith = np.array([np.all(power == powers_desired) for power in powers])
    return beta[ith]

# Create PSF Data
PSF_Drawer = Zernike_to_Pixel_Interpolator()
PSF_Evaluator = Moment_Evaluator()
def evaluate_psf(data):
    stamps, data = PSF_Drawer(data)
    evaluated_psfs = PSF_Evaluator(stamps)
    # this is sick:
    combined_df = evaluated_psfs.combine_first(data)

    return combined_df

rzero = 0.14
x = 0
y = 0

# y goes from -221 to 221
# x goes fro -200 to 200
# rzero goes from 0.1 to 0.25

Ndim = 8  # z4 through 11
Nsample = 10000
zmax = 0.5

# zernikes = np.meshgrid(*[np.linspace(-zmax, zmax, Nsample) for i in xrange(Ndim)])
# # z4 = zernikes[0], z5 = zernikes[1], etc
# N = zernikes[0].size

zernikes = np.random.random(size=(Ndim, Nsample)) * (zmax + zmax) - zmax

data = {'rzero': np.ones(Nsample) * rzero,
        'x': np.ones(Nsample) * x,
        'y': np.ones(Nsample) * y}

x_keys = []
for zi, zernike in enumerate(zernikes):
    zkey = 'z{0}'.format(zi + 4)
    data[zkey] = zernike.flatten()
    x_keys.append(zkey)
df = pd.DataFrame(data)
df = evaluate_psf(df)

y_keys = ['flux', 'e0prime', 'e0', 'e1', 'e2',
          'delta1', 'delta2', 'zeta1', 'zeta2']

# stack xs
yith = 2  # e0
deg = 2  # want 2d polynomial

xs = df[x_keys].values
ys = df[y_keys[yith]].values

x_powers, powers = stack_x(xs, deg)

# normalize the data
x_powers_std = x_powers.std(axis=0)
print(x_powers_std)
x_powers[:, 1:] /= x_powers_std[1:]

# split dataset. 0.6 train, 0.2 val, 0.2 test
x_powers_train, x_powers_test, ys_train, ys_test = train_test_split(x_powers, ys, test_size=0.4)
x_powers_test, x_powers_val, ys_test, ys_val = train_test_split(x_powers_test, ys_test, test_size=0.5)

# perform fit and compare for hyperparameters

from sklearn.linear_model import Lasso, LassoLars, Ridge
best_model = None
best_score = -10000
for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    lasso = Lasso(alpha=alpha, normalize=True, positive=True,
                  fit_intercept=False,
                  max_iter=10000)
    lasso.fit(x_powers_train, ys_train)
    print('lasso', lasso.get_params(), lasso.score(x_powers_test, ys_test))
    if lasso.score(x_powers_test, ys_test) > best_score:
        best_score = lasso.score(x_powers_test, ys_test)
        best_model = lasso
    lassolars = LassoLars(alpha=alpha, normalize=True, max_iter=10000, fit_intercept=False)
    lassolars.fit(x_powers_train, ys_train)
    if lassolars.score(x_powers_test, ys_test) > best_score:
        best_score = lassolars.score(x_powers_test, ys_test)
        best_model = lassolars
    print('lassolars', lassolars.get_params(), lassolars.score(x_powers_test, ys_test))
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(x_powers_train, ys_train)
    print('ridge', ridge.get_params(), ridge.score(x_powers_test, ys_test))
    if ridge.score(x_powers_test, ys_test) > best_score:
        best_score = ridge.score(x_powers_test, ys_test)
        best_model = ridge

plt.figure()
plt.plot(best_model.predict(x_powers_val), ys_val, '.')
plt.show()
import ipdb; ipdb.set_trace()

# this is what a simple least squares (unweighted) is:

y = ys[:, yith]
beta = np.linalg.lstsq(x_powers_train, y_train)

# look at results, do feature selection.



# now do rzero



