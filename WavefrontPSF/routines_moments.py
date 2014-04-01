#!/usr/bin/env python
"""
File: routines_moments.py
Author: Chris Davis
Description: set of useful moment routines for focal plane objects.
"""

from __future__ import print_function, division
import numpy as np
from numpy.lib.recfunctions import merge_arrays
from routines import convert_dictionary

def second_moment_to_ellipticity(x2, y2, xy, **args):

    """Take moments and convert to unnormalized ellipticity basis

    Parameters
    ----------
    x2, y2, xy : array
        Array of second moments, e.g. fits_data['X2WIN_IMAGE'] .

    Returns
    -------
    e0, e0prime, e1, e2 : array
        Arrays converted to unnormalized ellipticity basis.
        e0prime is an alternative definition for the spin 0 second moment.

    Notes
    -----
    If you want the normalized (and unitless) ellipticity:
        e1 -> e1 / e0
        e2 -> e2 / e0

    Units are arcsec ** 2 provided the moments are in pixel ** 2.

    Roughly, FWHM ~ sqrt(8 ln 2 e0).

    References
    ----------
    see http://des-docdb.fnal.gov:8080/cgi-bin/RetrieveFile?docid=353
    though my convention is different.

    """

    e0 = (x2 + y2) * 0.27 ** 2
    e0prime = (x2 + y2 + 2 * np.sqrt(x2 * y2 - xy ** 2)) * 0.27 ** 2
    e1 = (x2 - y2) * 0.27 ** 2
    e2 = (2 * xy) * 0.27 ** 2

    return e0, e0prime, e1, e2


def third_moments_to_octupoles(x3, x2y, xy2, y3, **args):

    """Take moments and convert to unnormalized octupole basis

    Parameters
    ----------
    x3, x2y, xy2, y3 : array
        Array of third moments.

    Returns
    -------
    zeta1, zeta2, delta1, delta2 : array
        Arrays converted to unnormalized octupole basis.
        zeta is spin-1, delta is spin-3 (See Okura 2008)
        these are F and G (roughly and modulo normalization factors)

    Notes
    -----

    Units are arcsec ** 3 provided the moments are in pixel ** 3.

    """

    zeta1 = (x3 + xy2) * 0.27 ** 3
    zeta2 = (y3 + x2y) * 0.27 ** 3
    delta1 = (x3 - 3 * xy2) * 0.27 ** 3
    delta2 = (-y3 + 3 * x2y) * 0.27 ** 3

    return zeta1, zeta2, delta1, delta2


def ellipticity_to_whisker(e1, e2, spin=2., power=2., **args):

    """Take unnormalized ellipticities and convert to relevant whisker
    parameters

    Parameters
    ----------
    e1, e2 : array
        Array of unnormalized ellipticity vectors.

    Returns
    -------
    u, v : array
        Whisker parameters in cartesian basis (for plotting with matplotlib
        quiver).

    w, phi : array
        Whisker length and position angle. (See reference below.)

    Notes
    -----
    If your unnormalized ellipticities are in quantity x$^{2}$ (e.g.
    pixels$^{2}$), then all these return quantities are in x (except for phi,
    which is in radians).

    References
    ----------
    see http://des-docdb.fnal.gov:8080/cgi-bin/RetrieveFile?docid=353

    """

    # get magnitude and direction
    w = np.sqrt(np.square(e1) + np.square(e2)) ** (1 / power)
    phi = np.arctan2(e2, e1) / spin
    # convert to cartesian
    u = w * np.cos(phi)
    v = w * np.sin(phi)

    return u, v, w, phi


def second_moment_variance_to_ellipticity_variance(var_x2, var_y2, var_xy,
                                                   **args):

    """Convert variance in moments to ellipticity variances

    Parameters
    ----------
    var_x2, var_y2, var_xy : array
        Array of second moments variances

    Returns
    -------
    var_e0, var_e1, var_e2 : array
        Arrays converted to unnormalized ellipticity basis.

    """

    alpha = 0.27 ** 2  # pixel to arcsec
    var_e0 = alpha ** 2 * (var_x2 + var_y2)
    var_e1 = alpha ** 2 * (var_x2 + var_y2)
    var_e2 = alpha ** 2 * 4 * var_xy

    return var_e0, var_e1, var_e2

def third_moment_variance_to_octupole_variance(
        var_x3, var_x2y, var_xy2, var_y3, **args):

    """Convert variance in moments to ellipticity variances assuming no
    covariance.

    Parameters
    ----------
    var_x3, etc : array
        Array of third moments variances

    Returns
    -------
    var_delta1 etc : array
        Arrays converted to unnormalized octupole basis.

    """

    alpha = 0.27 ** 3  # pixel to arcsec
    var_zeta1 = alpha ** 2 * (var_x3 + var_xy2)
    var_zeta2 = alpha ** 2 * (var_y3 + var_x2y)

    var_delta1 = alpha ** 2 * (var_x3 + 9 * var_xy2)
    var_delta2 = alpha ** 2 * (var_y3 + 9 * var_x2y)

    return var_zeta1, var_zeta2, var_delta1, var_delta2


def ellipticity_variance_to_whisker_variance(e1, e2, var_e1, var_e2, **args):

    """Convert error in ellipticity to error in cartesian whisker parameters
    (for later wedge plotting).

    Parameters
    ----------
    e1, e2 : array
        Array of unnormalized ellipticity vectors.

    var_e1, var_e2 : array or float
        Variance in measurement of e1 or e2, either one value for all, or one
        value per object.

    Returns
    -------
    var_u, var_v : array
        Variance in the u and v components of the whisker.

    """

    # get relevant whisker parameters
    u, v, w, phi = ellipticity_to_whisker(e1, e2)

    # thanks to the power of WolframAlpha:
    dude1 = 0.5 * w ** -4 * (e2 * v + e1 * u)
    dude2 = 0.5 * w ** -4 * (e2 * u - e1 * v)
    dvde1 = 0.5 * w ** -4 * (-e2 * u + e1 * v)
    dvde2 = 0.5 * w ** -4 * (e2 * v + e1 * u)

    # errors are added in quadrature
    var_u = dude1 ** 2 * var_e1 + dude2 ** 2 * var_e2
    var_v = dvde1 ** 2 * var_e1 + dvde2 ** 2 * var_e2

    return var_u, var_v

def convert_moments(data, **args):

    """Looks through data and converts all relevant parameters and updates into
    the data object.

    Parameters
    ----------
    data : recarray or dictionary
        Set of data with the moments

    Returns
    -------
    poles : dictionary
        Dictionary of results

    """

    poles = {}
    if ('x2' in data) * ('y2' in data) * ('xy' in data):
        e0, e0prime, e1, e2 = second_moment_to_ellipticity(**data)
        poles.update(dict(e0=e0, e0prime=e0prime, e1=e1, e2=e2))

    if ('e1' in poles) * ('e2' in poles):
        w1, w2, w, phi = ellipticity_to_whisker(**poles)
        poles.update(dict(w1=w1, w2=w2, w=w, phi=phi))

    if ('x3' in data) * ('x2y' in data) * ('xy2' in data) * ('y3' in data):
        zeta1, zeta2, delta1, delta2 = third_moments_to_octupoles(**data)
        wd1, wd2 = ellipticity_to_whisker(delta1, delta2, spin=3, power=3)[:2]
        poles.update(dict(zeta1=zeta1, zeta2=zeta2,
                          delta1=delta1, delta2=delta2))

    if ('delta1' in poles) * ('delta2' in poles):
        wd1, wd2 = ellipticity_to_whisker(poles['delta1'], poles['delta2'],
                                          spin=3, power=3)[:2]
        poles.update(dict(wd1=wd1, wd2=wd2))



    if ('x4' in data) * ('x2y2' in data) * ('y4' in data):
        xi = data['x4'] + 2 * data['x2y2'] + data['y4']
        poles.update(dict(xi=xi))

    # other parameters
    if ('x' in data) * ('y' in data):
        poles.update(dict(x=data['x'], y=data['y']))

    if ('x_box' in data) * ('y_box' in data):
        poles.update(dict(x_box=data['x_box'], y_box=data['y_box']))

    if 'n' in data:
        poles.update(dict(n=data['n']))


    # Variances
    if (('var_x2' in data) * ('var_y2' in data) * ('var_xy' in data) *
            ('var_e0' not in data) * ('var_e1' not in data) *
            ('var_e2' not in data)):
        var_e0, var_e1, var_e2 = \
            second_moment_variance_to_ellipticity_variance(**data)
        poles.update(dict(var_e0=var_e0, var_e1=var_e1, var_e2=var_e2))

    if (('e1' in poles) * ('e2' in poles) *
            ('var_e1' in poles) * ('var_e2' in poles) *
            ('var_w1' not in poles) * ('var_w2' not in poles)):
        var_w1, var_w2 = ellipticity_variance_to_whisker_variance(**poles)
        poles.update(dict(var_w1=var_w1, var_w2=var_w2))

    if (('var_x3' in data) * ('var_x2y' in data) *
            ('var_xy2' in data) * ('var_y3' in data) *
            ('var_zeta1' not in data) * ('var_zeta2' not in data) *
            ('var_delta1' not in data) * ('var_delta2' not in data)):
        var_zeta1, var_zeta2, var_delta1, var_delta2 = \
            third_moment_variance_to_octupole_variance(**data)
        poles.update(dict(var_zeta1=var_zeta1, var_zeta2=var_zeta2,
                          var_delta1=var_delta1, var_delta2=var_delta2,))

    return poles

