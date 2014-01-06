#!/usr/bin/env python
"""
File: focal_plane_routines.py
Author: Chris Davis
Description: set of useful routines for focal plane objects.
"""

from __future__ import print_function, division
import numpy as np
from decamutil_cpd import decaminfo

# TODO: docs!

def print_command(command):
    string = ''
    for i in command:
        string += str(i)
        string += ' '
    print(string)
    return string

def rzero_to_fwhm(rzero):
    # from fitting tests...
    fwhm = 0.13425711 / rzero + 0.25449468
    return fwhm


def fwhm_to_rzero(fwhm):
    rzero = 0.13425711 / (fwhm - 0.25449468)
    return rzero


def e0_to_fwhm(e0, windowed=True):
    '''
    roughly speaking, should be sqrt(8 ln 2 e0) but this is from fitting
    data
    '''
    if windowed:
        fwhm = 2.2122823 * np.sqrt(e0) + 0.03406004
    else:
        fwhm = 3.56246781 * np.sqrt(e0) - 2.72639134
    return fwhm


def fwhm_to_e0(fwhm, windowed=True):
    if windowed:
        e0 = np.square((fwhm - 0.03406004) / 2.2122823)
    else:
        e0 = np.square((fwhm + 2.72639134) / 3.56246781)
    return e0


def second_moment_to_ellipticity(x2, y2, xy):

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


def third_moments_to_octupoles(x3, x2y, xy2, y3):

    """Take moments and convert to unnormalized octupole basis

    Parameters
    ----------
    x3, x2y, xy2, y3 : array
        Array of third moments.

    Returns
    -------
    zeta1, zeta2, delta1, delta2 : array
        Arrays converted to unnormalized octupole basis.

    Notes
    -----

    Units are arcsec ** 3 provided the moments are in pixel ** 3.

    """

    zeta1 = (x3 + xy2) * 0.27 ** 3
    zeta2 = (y3 + x2y) * 0.27 ** 3
    delta1 = (x3 - 3 * xy2) * 0.27 ** 3
    delta2 = (-y3 + 3 * x2y) * 0.27 ** 3

    return zeta1, zeta2, delta1, delta2


def ellipticity_to_whisker(e1, e2):

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
    w = np.sqrt(np.sqrt(np.square(e1) + np.square(e2)))
    phi = 0.5 * np.arctan2(e2, e1)
    # convert to cartesian
    u = w * np.cos(phi)
    v = w * np.sin(phi)

    return u, v, w, phi


def second_moment_variance_to_ellipticity_variance(x2_var, y2_var, xy_var):

    """Convert variance in moments to ellipticity variances

    Parameters
    ----------
    x2_var, y2_var, xy_var : array
        Array of second moments variances

    Returns
    -------
    e0_var, e1_var, e2_var : array
        Arrays converted to unnormalized ellipticity basis.

    """
    alpha = 0.27 ** 2  # pixel to arcsec
    e0_var = alpha ** 2 * (x2_var + y2_var)
    e1_var = alpha ** 2 * (x2_var + y2_var)
    e2_var = alpha ** 2 * 4 * xy_var

    return e0_var, e1_var, e2_var


def ellipticity_variance_to_whisker_variance(e1, e2, e1_var, e2_var):

    """Convert error in ellipticity to error in cartesian whisker parameters
    (for later wedge plotting).

    Parameters
    ----------
    e1, e2 : array
        Array of unnormalized ellipticity vectors.

    e1_var, e2_var : array or float
        Variance in measurement of e1 or e2, either one value for all, or one
        value per object.

    Returns
    -------
    u_var, v_var : array
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
    u_var = dude1 ** 2 * e1_var + dude2 ** 2 * e2_var
    v_var = dvde1 ** 2 * e1_var + dvde2 ** 2 * e2_var

    return u_var, v_var


def average_dictionary(
        data, average,
        boxdiv=1, subav=False):
    returndict = {}
    x = data['x']
    y = data['y']
    for name in data.keys():
        zin = data[name]
        # subtract average if desired
        if subav:
            zin = zin - average(zin)
        z, z2 = decaminfo().average_boxdiv(x, y, zin, average,
                                           boxdiv=boxdiv)
        returndict.update({name: z, '{0}_2'.format(name): z2})
    # do x and y and get number, too
    x_av, x_av2, N, bounds = decaminfo().average_boxdiv(x, y, x, average,
                                                        boxdiv=boxdiv,
                                                        Ntrue=True)
    y_av, y_av2, = decaminfo().average_boxdiv(x, y, y, average,
                                              boxdiv=boxdiv,
                                              Ntrue=False)

    returndict.update({'x': x_av, 'x_2': x_av2,
                       'y': y_av, 'y_2': y_av2,
                       'n': N})

    # get the midpoints of each box
    x_box = []
    y_box = []
    for box in bounds:
        for x in range(len(box[0]) - 1):
            for y in range(len(box[1]) - 1):
                x_box.append((box[0][x] + box[0][x + 1]) / 2.)
                y_box.append((box[1][y] + box[1][y + 1]) / 2.)
    x_box = np.array(x_box)
    y_box = np.array(y_box)
    returndict.update({'x_box': x_box, 'y_box': y_box})

    return returndict


def variance_dictionary(data, keys, var_type=0):
    var_dict = {}
    for key in keys:
        if var_type == 1:
            # just set var to 1
            var = 1
        elif var_type == -1:
            # I have no idea why I did this; this is for historical record
            N = data['n']
            val = data[key]
            val_2 = data['{0}_2'.format(key)]
            var = (val_2) / N
        else:
            # calculate variance from usual statistics
            N = data['n']
            val = data[key]
            val_2 = data['{0}_2'.format(key)]
            var = (val_2 - val ** 2) / N

        var_dict.update({key: var})

    return var_dict


def chi2(data_a, data_b, chi_weights, var_dict):
    '''
    make the chi2

    here I choose to consider the overall differences between ellipticity
    vectors over the plane: chi2 = square(e1_a - e1_b) + square(e2_a -
    e2_b)

    choose the error to be in the observed data (assumed to be data_b),
    sigma^{2}_{i,b} = 1 - N_{b} (<e_{i,b}^{2}> - <e_{i,b}>^2)
    where the averaging is over the chip. ie (ei_2_b - ei_b**2) / N_B

    '''

    chi_dict = {'chi2': 0}
    for key in chi_weights.keys():
        val_a = data_a[key]
        val_b = data_b[key]
        weight = chi_weights[key]
        var = var_dict[key]

        chi2 = np.square(val_a - val_b) / var

        # update chi_dict
        chi_dict.update({key: chi2,
                         'var_{0}'.format(key): var})
        chi_dict['chi2'] = chi_dict['chi2'] + np.sum(weight * chi2)

    # check whether chi_dict['chi2'] is an allowable number (finite
    # positive)
    if (chi_dict['chi2'] < 0) + (np.isnan(chi_dict['chi2'])):
        # if it isn't, make it really huge
        chi_dict['chi2'] = 1e20

    return chi_dict


def average_function(data, average, average_type):
    if average_type == 'scalar_whisker':
        a_i = average(np.sqrt(np.sqrt(data['e1'] ** 2 +
                                      data['e2'] ** 2)))
    elif average_type == 'vector_whisker':
        a_i = np.sqrt(np.sqrt(average(data['e1']) ** 2 +
                              average(data['e2']) ** 2))
    else:
        a_i = average(data[average_type])
    return a_i


def minuit_dictionary(keys):
    # based on what params you want to fit to, create a minuit dictionary
    # with errors and initial values and everything
    minuit_dict = {}
    for key in keys:
        minuit_dict.update({'fix_{0}'.format(key): False})
        if key == 'rzero':
            minuit_dict.update({'rzero': 0.125})
            minuit_dict.update({'error_rzero': 0.005})
            minuit_dict.update({'limit_rzero': (0.07, 0.4)})
            #minuit_dict.update({'fix_rzero':False})
        elif key == 'fwhm':
            #minuit_dict.update({'fwhm':0.14 / self.rzero})
            minuit_dict.update({'fwhm': 1.0})
            minuit_dict.update({'error_fwhm': 0.1})
            minuit_dict.update({'limit_fwhm': (0.6, 2.0)})
        elif key == 'dz':
            minuit_dict.update({'dz': 0})
            minuit_dict.update({'error_dz': 40})
            minuit_dict.update({'limit_dz': (-300, 300)})
        elif key == 'z04d':
            minuit_dict.update({'z04d': 0})
            minuit_dict.update({'error_z04d': 40})
            minuit_dict.update({'limit_z04d': (-300, 300)})
        elif key == 'z04x':
            minuit_dict.update({'z04x': 0})
            minuit_dict.update({'error_z04x': 20})
            minuit_dict.update({'limit_z04x': (-100, 100)})
        elif key == 'z04y':
            minuit_dict.update({'z04y': 0})
            minuit_dict.update({'error_z04y': 20})
            minuit_dict.update({'limit_z04y': (-100, 100)})
        elif (key[0] == 'z') * (len(key) == 4):
            # all zernikes
            # guess from image_correction
            minuit_dict.update({key: 0})
            ## znum = int(key[1:3]) - 1
            ## ztype = key[-1]
            ## ztype_num = ztype_dict[ztype]
            ## minuit_dict.update({key:self.image_correction[znum][ztype_num]})
            if key[3] == 'd':
                minuit_dict.update({'error_{0}'.format(key): 0.1})
                minuit_dict.update({'limit_{0}'.format(key): (-0.75, 0.75)})
            else:
                # x and y
                minuit_dict.update({'error_{0}'.format(key): 5e-4})
                minuit_dict.update({
                    'limit_{0}'.format(key): (-0.0075, 0.0075)})
        elif (key[0] == 'e') * (len(key) == 2):
            # looking at the common mode terms
            minuit_dict.update({key: 0})
            minuit_dict.update({'error_{0}'.format(key): 0.005})
            minuit_dict.update({'limit_{0}'.format(key): (-0.05, 0.05)})
        else:
            # hexapod:
            minuit_dict.update({key: 0})
            minuit_dict.update({'error_{0}'.format(key): 100})
            minuit_dict.update({'limit_{0}'.format(key): (-4500, 4500)})

    return minuit_dict


def in_dict_from_minuit_dict(minuit_dict):
    # determine par_names
    par_names = []
    for key in minuit_dict:
        if 'error_' in key:
            par_names.append(key[6:])
    in_dict = {}
    for pari in par_names:
        in_dict.update({pari: minuit_dict[pari]})
    return in_dict


def image_zernike_corrections(image_data):
    """create image_correction from image data

    Parameters
    ----------
    image_data : recarray
        contains all the telescope information for the specific image.

    Returns
    -------
    image_dictionary : dictionary
        dictionary with all the donut corrections

    Notes
    -----
    tries to fit the "do" ones first (which are offline), but if those are
    not there, then try the online processing. If that is /also/ not
    present, then nothing gets added!

    """

    # get image_correction
    # get corrections for an image
    image_dictionary = {}

    image_correction_keys = ['', '5', '6', '7', '8', '9', '10', '11']
    row_keys = ['delta', 'thetax', 'thetay']
    row_keys_alt = ['d', 'x', 'y']
    for key_i in range(len(image_correction_keys)):
        key = image_correction_keys[key_i]
        for row_key_i in range(len(row_keys)):
            row_key = row_keys[row_key_i]
            # first try do; the offline processing
            entry = 'doz' + key + row_key
            if entry in image_data.dtype.names:
                if not image_data[entry].mask:
                    if image_data[entry].data > -2000:
                        image_dictionary.update({
                            'z{0:02d}{1}'.format(key_i + 4,
                                                 row_keys_alt[row_key_i]):
                            image_data[entry].data})
                # try online processing
                else:
                    entry = 'z' + key + row_key
                    if entry in image_data.dtype.names:
                        if not image_data[entry].mask:
                            image_dictionary.update({
                                'z{0:02d}{1}'.format(
                                    key_i + 4,
                                    row_keys_alt[row_key_i]):
                                image_data[entry].data})
    # now do the hexapod parameters
    hexapod_keys = ['dz', 'dx', 'dy', 'xt', 'yt']
    # there is a minus sign in the correction because of aaron's
    # conventions
    for hexapod_key in hexapod_keys:
        # first try online
        entry = 'dodo' + hexapod_key
        if entry in image_data.dtype.names:
            if not image_data[entry].mask:
                if image_data[entry].data > -2000:
                    image_dictionary.update({
                        hexapod_key:
                        -image_data[entry].data})
            else:
                if not image_data[entry].mask:
                    image_dictionary.update({
                        hexapod_key:
                        -image_data[entry].data})

    return image_dictionary
