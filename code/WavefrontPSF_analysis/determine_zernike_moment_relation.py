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

def stack_powers(num_covariates, deg, keys):
    generators = [basis_vector(num_covariates+1, i)
                  for i in range(num_covariates+1)]

    # All combinations of degrees
    powers = map(sum, combinations_with_replacement(generators, deg))

    # remove some powers we don't like
    # first find the index corresponding to our rzero-like term
    # the plus 1 is because the first term in the column is 1s
    if 'one_over_rzero' in keys:
        indx = keys.index('one_over_rzero') + 1
    elif 'rzero' in keys:
        indx = keys.index('rzero') + 1
    else:
        # skip and move on
        indx = -1
    if indx > -1:
        # we want to cut anything with rzero dep > 2
        powers_temp = []
        for p in powers:
            # any combo of variable dep > 4 and no rzero
            if sum(p) - p[0] - p[indx] > 4:
                continue
            # rzero dep > 2 and not another variable
            if p[indx] > 2 and p[indx] + p[0] != sum(p):
                continue
            # if rzero is by itself and > 3
            if p[indx] > 3 and sum(p) - p[0] - p[indx] == 0:
                continue
            powers_temp.append(p)
        powers = powers_temp

    return powers

def stack_x(xs, deg, keys):

    num_covariates = xs.shape[1]
    xs = hstack((ones((xs.shape[0], 1), dtype=xs.dtype) , xs))

    powers = stack_powers(num_covariates, deg, keys)

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

def power_to_string(power, keys):
    # assume len(keys) = 1 + len(powers[0])
    string = ''
    for power_ith in xrange(len(keys)):
        term_power = keys[power_ith]
        number_terms = power[power_ith + 1]
        for j in xrange(number_terms):
            string += term_power
            string += ' '
    return string

def mk_pretty_function(beta, powers, x_keys=['', 'r0'] + ['z{0}'.format(i) for i in xrange(4, 12)],
                       min_threshold=-20):
    num_covariates = len(powers[0]) - 1
    xs = [''] + x_keys
    terms = []
    keys = []
    for ith in xrange(len(beta)):
        coef = beta[ith]
        # since z is in the range -1 to 1, you can imagine that 
        # each additional zernike requires a compensation in the
        # coefficient by another order of magnitude
        if 'x' in x_keys:
            if_coord = powers[ith][xs.index('x')] * 2
            if_coord += powers[ith][xs.index('y')] * 2
        else:
            if_coord = 0
        if np.abs(coef) < 10 ** (min_threshold - powers[ith][0] + if_coord + np.sum(powers[ith])):
            coef = 0
        power = powers[ith]
        term = '{0:.2e}'.format(coef)
        if term.count('0') == 5:
            continue
        key = ''
        for power_ith in xrange(len(power)):
            term_power = xs[power_ith]
            number_terms = power[power_ith]
            for j in xrange(number_terms):
                key += term_power
        keys.append(key)
        if len(term) > 0:
            terms.append(term)
    # print the terms
    string = ''
    for key, term in zip(keys, terms):
        string += term
        string += ' '
        string += key
        string += ' + '
    return terms, string, keys

# part 1:
# given fixed rzero, fit coordinates and create interpolation
donut_dir = '/Users/cpd/Projects/WavefrontPSF/meshes'

df = pd.read_csv(donut_dir + '/indivi_15/donuts.csv', index_col=0)

# do fit with one over rzero
df['one_over_rzero'] = 1. / df['rzero']
# stack and fit
x_keys = ['one_over_rzero', 'x', 'y']
for zi in xrange(4, 12):
    zkey = 'z{0}'.format(zi)
    x_keys.append(zkey)
y_keys = ['flux', 'Mx', 'My', 'e0prime', 'e0', 'e1', 'e2',
          'delta1', 'delta2', 'zeta1', 'zeta2', 'a4']
deg = 4 + 2  # to let the rzero factors come into play as well
xs = df[x_keys].values
ys = df[y_keys].values
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys,
    test_size=0.1)
x_powers_train, powers = stack_x(xs_train, deg, x_keys)
x_powers_test, powers = stack_x(xs_test, deg, x_keys)


regressor = LinearRegression(fit_intercept=False, normalize=False)
regressor.fit(x_powers_train, ys_train)
ys_pred = regressor.predict(x_powers_test)
ys_train_pred = regressor.predict(x_powers_train)
print('Training complete. rms train = {0:.2e}. rms test = {1:.2e}'.format(
    np.sqrt(np.mean(np.square(ys_train_pred - ys_train))),
    np.sqrt(np.mean(np.square(ys_pred - ys_test))),
    ))

# save results
model = {'x_keys': x_keys, 'y_keys': y_keys,
         'powers': powers, 'deg': deg,
         'coef': regressor.coef_.copy(),
         'intercept': regressor.intercept_.copy()}
tests = {'x_train': xs_train, 'x_test': xs_test,
         'y_train': ys_train, 'y_test': ys_test,
         'y_train_predict': ys_train_pred, 'y_test_predict': y_pred}
np.save(donut_dir + '/Analytic_Coeffs/model_rzero.npy', model)
np.save(donut_dir + '/Analytic_Coeffs/tests_rzero.npy', tests)

# repeat with normalized
regressor = LinearRegression(fit_intercept=False, normalize=True)
regressor.fit(x_powers_train, ys_train)
ys_pred = regressor.predict(x_powers_test)
ys_train_pred = regressor.predict(x_powers_train)
print('Normalized Training complete. rms train = {0:.2e}. rms test = {1:.2e}'.format(
    np.sqrt(np.mean(np.square(ys_train_pred - ys_train))),
    np.sqrt(np.mean(np.square(ys_pred - ys_test))),
    ))

# save results
model = {'x_keys': x_keys, 'y_keys': y_keys,
         'powers': powers, 'deg': deg,
         'coef': regressor.coef_.copy(),
         'intercept': regressor.intercept_.copy()}
tests = {'x_train': xs_train, 'x_test': xs_test,
         'y_train': ys_train, 'y_test': ys_test,
         'y_train_predict': ys_train_pred, 'y_test_predict': y_pred}
np.save(donut_dir + '/Analytic_Coeffs/model_rzero_normalized.npy', model)
np.save(donut_dir + '/Analytic_Coeffs/tests_rzero_normalized.npy', tests)


groups = df.groupby('rzero')
group_keys = sorted(groups.groups.keys())
rzeros = ['{0:.2f}'.format(0.08 + 0.01 * i) for i in xrange(15)]

# stack and fit
x_keys = ['x', 'y']
for zi in xrange(4, 12):
    zkey = 'z{0}'.format(zi)
    x_keys.append(zkey)
y_keys = ['flux', 'Mx', 'My', 'e0prime', 'e0', 'e1', 'e2',
          'delta1', 'delta2', 'zeta1', 'zeta2', 'a4']

# rzero coefficients from creating many rzeros at no distortion and fitting
"""
def r0_guess(e0, poly):
    a, m, b = poly
    part1 = m / (2 * e0 - b)
    part2 = part1**2 + a / (e0 - b)
    return part1 + np.sqrt(part2)#, part1 - np.sqrt(part2)
"""
model = {'rzero': [0.00418117,  0.00106946,  0.00731245]}
tests = {}
deg = 4
coef = []
for group_key_i, group_key in enumerate(group_keys):
    print(group_key, rzeros[group_key_i])
    dfi = groups.get_group(group_key)

    xs = dfi[x_keys].values  # no rzero
    ys = dfi[y_keys].values
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys,
        test_size=0.3)
    x_powers_train, powers = stack_x(xs_train, deg, x_keys)
    x_powers_test, powers = stack_x(xs_test, deg, x_keys)
    regressor = LinearRegression(fit_intercept=False, normalize=False)
    regressor.fit(x_powers_train, ys_train)
    ys_pred = regressor.predict(x_powers_test)
    coef.append(regressor.coef_.copy())
    model[rzeros[group_key_i]] = [regressor.coef_.copy(), powers, deg, x_keys, y_keys]
    tests[rzeros[group_key_i]] = [xs_train, ys_train, xs_test, ys_test, ys_pred]
coef = np.array(coef)
# save it all!
np.save(donut_dir + '/Analytic_Coeffs/coef.npy', coef)
np.save(donut_dir + '/Analytic_Coeffs/model.npy', model)
np.save(donut_dir + '/Analytic_Coeffs/tests.npy', tests)
