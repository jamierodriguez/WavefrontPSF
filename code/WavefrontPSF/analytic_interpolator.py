#!/usr/bin/env python
"""
File: analytic_interpolator.py
Author: Chris Davis
Description: Using relations derived elsewhere, create zernike - moment interpolator
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from WavefrontPSF.psf_evaluator import PSF_Evaluator
from WavefrontPSF.wavefront import Wavefront
from WavefrontPSF.donutengine import Generic_Donutengine_Wavefront, Zernike_to_Misalignment_to_Pixel_Interpolator
from WavefrontPSF.psf_interpolator import kNN_Interpolator, PSF_Interpolator

class DECAM_Analytic_Wavefront(Generic_Donutengine_Wavefront):
    """
    Includes misalignments. Convenience class.

    Translates r0 to coefficients
    """

    def __init__(self, rzero, PSF_Interpolator_data=None,
            PSF_Evaluator_data=None):

        #TODO: make the csv here path independent
        if type(PSF_Interpolator_data) == type(None):
            PSF_Interpolator_data=pd.read_csv('/Users/cpd/Projects/WavefrontPSF/meshes/ComboMeshes2/Mesh_Science-20140212s2-v1i2_All_train.csv', index_col=0)
        if type(PSF_Evaluator_data) == type(None):
            PSF_Evaluator_data=np.load('/Users/cpd/Projects/WavefrontPSF/meshes/Analytic_Coeffs/model.npy').item()

        # take z4 and divide by 172 to put it in waves as it should be
        PSF_Interpolator_data['z4'] /= 172.
        PSF_Interpolator = kNN_Interpolator(PSF_Interpolator_data)

        # set the drawer
        PSF_Drawer = Zernike_to_Misalignment_to_Pixel_Interpolator()

        # translate rzero to coefficients
        rzeros_float = np.array([0.08 + 0.01 * i for i in xrange(15)])
        rzeros = ['{0:.2f}'.format(0.08 + 0.01 * i) for i in xrange(15)]
        # always want the rzero one above so that we underestimate it
        # note that this also implies that we want a dc component for e0 in
        # addition to e1 and e2
        rzero_i = np.searchsorted(rzeros_float, rzero)
        rzero_key = rzeros[rzero_i]
        PSF_Evaluator = Zernike_Evaluator(*PSF_Evaluator_data[rzero_key])

        super(DECAM_Analytic_Wavefront, self).__init__(
                PSF_Evaluator=PSF_Evaluator,
                PSF_Interpolator=PSF_Interpolator,
                PSF_Drawer=PSF_Drawer,
                model=None)


    def evaluate_psf(self, data, misalignment={}, force_interpolation=False, **kwargs):
        # cut out the actual stamp drawing in evaluate_psf
        data = self.PSF_Interpolator(data,
                force_interpolation=force_interpolation, **kwargs)
        # but still use the misaligning of zernikes from the PSF Drawer
        data_reduced, data_with_misalignments = self.PSF_Drawer.misalign_zernikes(data, misalignment)
        # evaluate
        evaluated_psfs = self.PSF_Evaluator(data_reduced, **kwargs)
        # this is sick:
        combined_df = evaluated_psfs.combine_first(data_with_misalignments)

        return combined_df

class Zernike_Evaluator(PSF_Evaluator):
    """Takes zernikes and coordinates and returns moments given coefficients"""

    def __init__(self, coef, powers, deg, x_keys, y_keys,
                 **kwargs):
        self.regressor = LinearRegression(fit_intercept=False, normalize=False)
        self.regressor.coef_ = coef
        self.regressor.intercept_ = 0
        self.deg = deg
        self.powers = powers
        self.x_keys = x_keys
        self.keys = y_keys

    def evaluate(self, psfs, **kwargs):
        # stack psfs
        psfs_stacked, _ = stack_x(psfs[self.x_keys].values, self.deg)
        # evaluate from regressor
        ydata = self.regressor.predict(psfs_stacked)
        # add to evaluated psf
        evaluated_psfs = {}
        for y_key_i, y_key in enumerate(self.keys):
            evaluated_psfs[y_key] = ydata[:, y_key_i]
        evaluated_psfs = pd.DataFrame(evaluated_psfs)
        return evaluated_psfs

def r0_guess(e0, poly=[0.00418117,  0.00106946,  0.00731245]):
    a, m, b = poly
    part1 = m / (2 * e0 - b)
    part2 = part1**2 + a / (e0 - b)
    return part1 + np.sqrt(part2)#, part1 - np.sqrt(part2)


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
