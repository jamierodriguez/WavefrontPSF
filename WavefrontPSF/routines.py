#!/usr/bin/env python
"""
File: routines.py
Author: Chris Davis
Description: set of useful routines.
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

def convert_dictionary(dictionary):
    # convert dictionary into rec array
    names = ''
    formats = ''
    for i in xrange(len(dictionary.keys())):
        names += sorted(dictionary.keys())[i]
        shape_i = np.shape(dictionary[sorted(dictionary.keys())[i]])
        if len(shape_i) > 1:
            strshape = str(shape_i[1:])
        else:
            strshape = ''

        dtype_i = str(np.array(dictionary[sorted(dictionary.keys())[i]]).dtype)
        formats += strshape + dtype_i
        if i != len(dictionary.keys()):
            names += ','
            formats += ','
    test = []
    # this has to be able to be sped up!
    for j in xrange(len(dictionary[sorted(dictionary.keys())[0]])):
        inner = []
        for i in sorted(dictionary.keys()):
            inner.append(dictionary[i][j])
        test.append(tuple(inner))
    test = np.rec.array(test, formats=formats, names=names)

    return test

def convert_recarray(recarray):
    # inverse of convert_dictionary
    return_dict = {}

    for name in recarray.dtype.names:
        return_dict.update({name: recarray[name]})

    return return_dict

def average_dictionary(
        data, average,
        boxdiv=1, subav=False,
        xcoord='x', ycoord='y',
        keys=[]):
    returndict = {}
    if len(keys) == 0:
        if type(data) == dict:
            keys = data.keys()
        elif type(data) == np.core.records.recarray:
            keys = data.dtype.names
    if type(xcoord) == str:
        x = data[xcoord]
    else:
        x = xcoord
    if type(ycoord) == str:
        y = data[ycoord]
    else:
        y = ycoord
    for name in keys:
        zin = data[name]
        # subtract average if desired
        if subav:
            zin = zin - average(zin)
        z, var_z = decaminfo().average_boxdiv(x, y, zin, average,
                                           boxdiv=boxdiv)
        returndict.update({name: z, 'var_{0}'.format(name): var_z})
    # do x and y and get number, too
    x_av, var_x, N, bounds = decaminfo().average_boxdiv(x, y, x, average,
                                                        boxdiv=boxdiv,
                                                        Ntrue=True)
    y_av, var_y, = decaminfo().average_boxdiv(x, y, y, average,
                                              boxdiv=boxdiv,
                                              Ntrue=False)

    returndict.update({'x': x_av, 'var_x': var_x,
                       'y': y_av, 'var_y': var_y,
                       'n': N})

    # get the midpoints of each box
    boxes = decaminfo().average_boxdiv(x, y, y, average,
                                       boxdiv=boxdiv,
                                       boxes=True)
    x_box = boxes[:,0]
    y_box = boxes[:,1]
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


def average_function(data, average=np.mean, average_type='default'):
    if average_type == 'scalar_whisker':
        a_i = average(np.sqrt(np.sqrt(data['e1'] ** 2 +
                                      data['e2'] ** 2)))
    elif average_type == 'vector_whisker':
        a_i = np.sqrt(np.sqrt(average(data['e1']) ** 2 +
                              average(data['e2']) ** 2))
    else:
        a_i = average(data[average_type])
    return a_i


def minuit_dictionary(keys, h_base=1e-3):
    # based on what params you want to fit to, create a minuit dictionary
    # with errors and initial values and everything
    minuit_dict = {}
    h_dict = {}
    for key in keys:
        minuit_dict.update({'fix_{0}'.format(key): False})
        if key == 'rzero':
            minuit_dict.update({key: 0.125})
            minuit_dict.update({'error_{0}'.format(key): 0.005})
            minuit_dict.update({'limit_{0}'.format(key): (0.07, 0.24)})
            h_dict.update({key: 0.005 * h_base})
            #minuit_dict.update({'fix_rzero':False})
        elif key == 'fwhm':
            #minuit_dict.update({'fwhm':0.14 / self.rzero})
            minuit_dict.update({key: 1.0})
            minuit_dict.update({'error_{0}'.format(key): 0.1})
            minuit_dict.update({'limit_{0}'.format(key): (0.6, 2.0)})
            h_dict.update({key: 0.1 * h_base})
        elif (key == 'dz') + (key == 'z04d'):
            minuit_dict.update({key: 0})
            minuit_dict.update({'error_{0}'.format(key): 40})
            minuit_dict.update({'limit_{0}'.format(key): (-400, 400)})
            h_dict.update({key: 40 * h_base})
        elif (key == 'z04x') + (key == 'z04y'):
            minuit_dict.update({key: 0})
            minuit_dict.update({'error_{0}'.format(key): 20})
            minuit_dict.update({'limit_{0}'.format(key): (-100, 100)})
            h_dict.update({key: 20 * h_base})
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
                minuit_dict.update({'limit_{0}'.format(key): (-1.25, 1.25)})
                h_dict.update({key: 0.1 * h_base})
            else:
                # x and y
                minuit_dict.update({'error_{0}'.format(key): 5e-4})
                minuit_dict.update({
                    'limit_{0}'.format(key): (-0.0075, 0.0075)})
                h_dict.update({key: 5e-4 * h_base})
        elif ((key == 'e1') + (key == 'e2') +
              (key == 'delta1') + (key == 'delta2') +
              (key == 'zeta1') + (key == 'zeta2')):
            # looking at the common mode terms
            minuit_dict.update({key: 0})
            minuit_dict.update({'error_{0}'.format(key): 0.005})
            minuit_dict.update({'limit_{0}'.format(key): (-0.5, 0.5)})
            h_dict.update({key: 0.005 * h_base})
        elif (key == 'dx') + (key == 'dy'):
            # hexapod:
            minuit_dict.update({key: 0})
            minuit_dict.update({'error_{0}'.format(key): 100})
            minuit_dict.update({'limit_{0}'.format(key): (-4500, 4500)})
            h_dict.update({key: 100 * h_base})
        elif (key == 'xt') + (key == 'yt'):
            # hexapod:
            minuit_dict.update({key: 0})
            minuit_dict.update({'error_{0}'.format(key): 50})
            minuit_dict.update({'limit_{0}'.format(key): (-1000, 1000)})
            h_dict.update({key: 50 * h_base})

    return minuit_dict, h_dict


def in_dict_from_minuit_dict(minuit_dict):
    # determine par_names
    par_names = []
    for key in minuit_dict:
        if '_' not in key:
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

def MAD(data, sigma=3):
    """Take your data and give conditions that cut out MAD outliers

    Parameters
    ----------
    data : array
        the data over which we want to filter

    sigma : float
        the number of sigma greater than which we want to fitler out

    Returns
    -------
    conds_mad : bool array
        boolean conditions for True being within sigma * mad

    mad : float
        The value of mad
    """
    a = data
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + sigma * mad) * (a > d - sigma * mad)

    return conds_mad, mad

def mean_trim(data, sigma=3, axis=0):
    conds, mad = MAD(data, sigma=sigma)
    return np.mean(data[conds])

def variance_trim(data, sigma=3, axis=0):
    conds, mad = MAD(data, sigma=sigma)
    return np.var(data[conds])

def get_data(data, i):
    ret = {}
    for key in data.keys():
        ret.update({key: data[key][i]})
    return ret


def vary_one_parameter(parameter,
                       FitFunc,
                       minuit_dict,
                       N=11):

    if minuit_dict['error_{0}'.format(parameter)] > 0:
        err = minuit_dict['error_{0}'.format(parameter)]
        mid = minuit_dict[parameter]
        parameters = np.linspace(mid - 5 * err,
                                 mid + 5 * err,
                                 N)
    else:
        parameters = np.linspace(
            minuit_dict['limit_{0}'.format(parameter)][0],
            minuit_dict['limit_{0}'.format(parameter)][1],
            N)
    chi2 = []
    for par in parameters:
        in_dict = {key: minuit_dict[key] for key in minuit_dict}
        in_dict.update({parameter: par})
        chi2_i = FitFunc(in_dict)
        chi2.append(chi2_i)
    return chi2, parameters


def vary_two_parameters(parameter,
                        parameter2,
                        FitFunc,
                        minuit_dict,
                        N=11):

    if minuit_dict['error_{0}'.format(parameter)] > 0:
        err = minuit_dict['error_{0}'.format(parameter)]
        mid = minuit_dict[parameter]
        parameters = np.linspace(mid - 5 * err,
                                 mid + 5 * err,
                                 N)
    else:
        parameters = np.linspace(
            minuit_dict['limit_{0}'.format(parameter)][0],
            minuit_dict['limit_{0}'.format(parameter)][1],
            N)

    # do parameter2 now
    if minuit_dict['error_{0}'.format(parameter2)] > 0:
        err = minuit_dict['error_{0}'.format(parameter2)]
        mid = minuit_dict[parameter]
        parameters2 = np.linspace(mid - 5 * err,
                                  mid + 5 * err,
                                  N)
    else:
        parameters2 = np.linspace(
            minuit_dict['limit_{0}'.format(parameter2)][0],
            minuit_dict['limit_{0}'.format(parameter2)][1],
            N)

    chi2 = []
    parameters_1 = []
    parameters_2 = []
    for par in parameters:
        for par2 in parameters2:
            in_dict = {key: minuit_dict[key] for key in minuit_dict}
            in_dict.update({parameter: par,
                            parameter2: par2})
            chi2_i = FitFunc(in_dict)
            chi2.append(chi2_i)
            parameters_1.append(par)
            parameters_2.append(par2)
    return parameters_1, parameters_2, chi2, parameters, parameters2
