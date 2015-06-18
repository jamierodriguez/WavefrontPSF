#!/usr/bin/env python
"""
File: psf_evaluator.py
Author: Chris Davis
Description: Module for evaluating psfs from images or parameters.

"""

from adaptive_moments import adaptive_moments, convert_moments
from pandas import DataFrame
import numpy as np

class PSF_Evaluator(object):
    """Class that evaluates a PSF from an image or input parameters.

    Attributes
    ----------

    keys : What properties are returned by PSF_Evaluator

    Methods
    -------

    Evaluate
        Return relevant PSF statistics.

    """

    def __init__(self, **kwargs):
        pass

    def evaluate(self, psfs, **kwargs):
        # by default just return whatever came in
        return psfs

    def __call__(self, psfs, **kwargs):
        return self.evaluate(psfs, **kwargs)

class Moment_Evaluator(PSF_Evaluator):
    """Class that takes a PSF image and evaluates its second and third moments.
    """

    def __init__(self, **kwargs):
        self.adaptive_moments_kwargs = dict(
            epsilon = 1e-6,
            convergence_factor = 1.0,
            guess_sig = 3.0,
            bound_correct_wt = 0.25,  # Maximum shift in centroids and sigma
                                      # between iterations for adaptive moments.
            num_iter = 0,
            num_iter_max = 100)
        self.adaptive_moments_kwargs.update(kwargs)

        self.keys = ['e0', 'e1', 'e2',
                     'delta1', 'delta2', 'zeta1', 'zeta2',
                     'a4', 'flux', 'Mx', 'My']

        super(Moment_Evaluator, self).__init__(**kwargs)

    def evaluate_one_psf(self, psf, background=0, threshold=-1e9, **kwargs):
        if background != 0:
            stamp = psf - background
        else:
            stamp = psf
        if threshold != -1e9:
            stamp = np.where(stamp > threshold, stamp, 0)

        # get moment matrix
        Mx, My, Mxx, Mxy, Myy, A, rho4, \
            x2, xy, y2, x3, x2y, xy2, y3 \
            = adaptive_moments(stamp, **self.adaptive_moments_kwargs)

        fwhm = np.sqrt(np.sqrt(Mxx * Myy - Mxy * Mxy))
        whisker = np.sqrt(np.sqrt(Mxy * Mxy + 0.25 * np.square(Mxx - Myy)))
        # 2 (1 + a4) = rho4
        a4 = 0.5 * rho4 - 1
        # update return_dict
        return_dict = {
                'Mx': Mx, 'My': My,
                'Mxx': Mxx, 'Mxy': Mxy, 'Myy': Myy,
                'fwhm': fwhm, 'flux': A, 'a4': a4, 'whisker': whisker,
                'x2': x2, 'xy': xy, 'y2': y2,
                'x3': x3, 'x2y': x2y, 'xy2': xy2, 'y3': y3,
                }
        # now get the poles etc
        return_dict = convert_moments(return_dict)
        # this gives us: e0, e1, e2, delta1, delta2, zeta1, zeta2

        return return_dict

    def evaluate(self, psfs, **kwargs):
        shape_psfs = np.shape(psfs)
        if len(shape_psfs) == 2:
            if shape_psfs[0] != shape_psfs[1]:
                raise TypeError('PSF needs to be an image or set of images where first index points to the specific image!')
            # boldly moving forth
            psfs = [psfs]
        # now we iterate through psfs
        evaluated_psfs = []
        for psf_i, psf in enumerate(psfs):
            evaluated_psfs.append(self.evaluate_one_psf(psf, **kwargs))
        # convert to DataFrame but only return the keys
        evaluated_psfs = DataFrame(evaluated_psfs)#[self.keys]
        return evaluated_psfs

