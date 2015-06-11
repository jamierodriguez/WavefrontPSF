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
from sklearn.linear_model import Lasso, LassoLars, Ridge, LinearRegression


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

def mk_pretty_function(beta, powers):
    num_covariates = len(powers[0]) - 1
    xs = ['', 'r0'] + ['z{0}'.format(i) for i in xrange(4, 12)]
    terms = []
    for ith in xrange(len(beta)):
        coef = beta[ith]
        power = powers[ith]
        term = '{0:.2e}'.format(coef)
        if term.count('0') == 5:
            continue
        term += ' '
        for power_ith in xrange(len(power)):
            term_power = xs[power_ith]
            number_terms = power[power_ith]
            for j in xrange(number_terms):
                term += term_power
        if len(term) > 0:
            terms.append(term)
    # print the terms
    string = ''
    for term in terms:
        string += term
        string += ' + '
    return terms, string


"""
# # Define some functions
# from WavefrontPSF.donutengine import Zernike_to_Pixel_Interpolator
# from WavefrontPSF.psf_evaluator import Moment_Evaluator
# 
# # Create PSF Data
# PSF_Drawer = Zernike_to_Pixel_Interpolator()
# PSF_Evaluator = Moment_Evaluator()
# def evaluate_psf(data):
#     stamps, data = PSF_Drawer(data)
#     evaluated_psfs = PSF_Evaluator(stamps)
#     # this is sick:
#     combined_df = evaluated_psfs.combine_first(data)
# 
#     return combined_df
# 
# rzero = 0.14
# x = 0
# y = 0
# 
# # y goes from -221 to 221
# # x goes fro -200 to 200
# # rzero goes from 0.1 to 0.25
# 
# Ndim = 8  # z4 through 11
# Nsample = 10000
# zmax = 0.5
# 
# # zernikes = np.meshgrid(*[np.linspace(-zmax, zmax, Nsample) for i in xrange(Ndim)])
# # # z4 = zernikes[0], z5 = zernikes[1], etc
# # N = zernikes[0].size
# 
# zernikes = np.random.random(size=(Ndim, Nsample)) * (zmax + zmax) - zmax
# 
# data = {'rzero': np.ones(Nsample) * rzero,
#         'x': np.ones(Nsample) * x,
#         'y': np.ones(Nsample) * y}
# 
# x_keys = []
# for zi, zernike in enumerate(zernikes):
#     zkey = 'z{0}'.format(zi + 4)
#     data[zkey] = zernike.flatten()
#     x_keys.append(zkey)
# df = pd.DataFrame(data)
# df = evaluate_psf(df)

df_in = pd.read_csv('/Users/cpd/Projects/WavefrontPSF/meshes/donuts_fixed_rzeros.csv', index_col=0)


x_keys = []
for zi in xrange(4, 12):
    zkey = 'z{0}'.format(zi)
    x_keys.append(zkey)

y_keys = ['flux', 'Mx', 'My', 'e0prime', 'e0', 'e1', 'e2',
          'delta1', 'delta2', 'zeta1', 'zeta2']

# stack xs
deg = 4  # want 2d polynomial
selection = np.isclose(df_in['x'], 10) * np.isclose(df_in['y'], 0)
df = df_in[selection]
xs = df[x_keys].values
ys = df[y_keys].values
x_powers, powers = stack_x(xs, deg)
# split dataset. 0.6 train, 0.2 val, 0.2 test
x_powers_train, x_powers_test, ys_train, ys_test = train_test_split(x_powers, ys, test_size=0.2)
#x_powers_test, x_powers_val, ys_test, ys_val = train_test_split(x_powers_test, ys_test, test_size=0.5)


# select rzero near 0.10, x near 10, y near 0
from sklearn.linear_model import Lasso, LassoLars, Ridge, LinearRegression
models = []
for rzero in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
    selection = np.isclose(df['rzero'], rzero)

    # perform fit and compare for hyperparameters
    #regressor = LinearRegression(fit_intercept=False, normalize=False)
    regressor = Lasso(alpha=1e-3, normalize=False, fit_intercept=False, tol=1e-8)
    regressor.fit(x_powers_train[selection], ys_train[selection])

    models.append([regressor.coef_.copy(), powers])

models = np.array(models)
# cool we have a model. now we try to fit a linear regression for each sample
# across rzero
"""
donut_dir = '/Users/cpd/Projects/WavefrontPSF/meshes'
donut_dir = '/nfs/slac/g/ki/ki18/des/cpd/donuts'
df = pd.read_csv(donut_dir + '/donuts_fixed_rzeros.csv', index_col=0)

x_keys = ['rzero']

for zi in xrange(4, 12):
    zkey = 'z{0}'.format(zi)
    x_keys.append(zkey)

y_keys = ['flux', 'Mx', 'My', 'e0prime', 'e0', 'e1', 'e2',
          'delta1', 'delta2', 'zeta1', 'zeta2']
# stack xs
deg = 4  # want 2d polynomial
xs = df[x_keys].values
ys = df[y_keys].values
x_powers, powers = stack_x(xs, deg)
# split dataset. 0.6 train, 0.2 val, 0.2 test
x_powers_train, x_powers_test, ys_train, ys_test = train_test_split(x_powers, ys, test_size=0.2)


print('training linear')
regressor = LinearRegression(fit_intercept=False, normalize=False)
regressor.fit(x_powers_train, ys_train)
ys_pred = regressor.predict(x_powers_test)
for yith in xrange(len(y_keys)):
    print(y_keys[yith], np.mean(np.sqrt(np.square(ys_pred[:, yith] - ys_test[:, yith]))))
    terms, string = mk_pretty_function(regressor.coef_[yith], powers)
    print(string)
np.save(donut_dir + '/coeffs_linear.npy', regressor.coef_)


print('training lasso')
# perform fit and compare for hyperparameters
#regressor = LinearRegression(fit_intercept=False, normalize=False)
regressor = Lasso(alpha=1e-3, normalize=False, fit_intercept=False, tol=1e-8)
regressor.fit(x_powers_train, ys_train)
ys_pred = regressor.predict(x_powers_test)
for yith in xrange(len(y_keys)):
    print(y_keys[yith], np.mean(np.sqrt(np.square(ys_pred[:, yith] - ys_test[:, yith]))))
    terms, string = mk_pretty_function(regressor.coef_[yith], powers)
    print(string)
np.save(donut_dir + '/coeffs_lasso.npy', regressor.coef_)


