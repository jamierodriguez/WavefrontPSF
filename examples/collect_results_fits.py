#!/usr/bin/env python
"""
File: collect_results.py
Author: Chris Davis
Description: Include the locations of the moments (both fitted and comparison)

This file will take the results and plot them as well as collate everything
together into csv files. Those can also be plotted?

TODO: using image_zernike_corrections add the priors used
TODO: add as an args_parse the size of the planes used
TODO: for the values that are ints, make them ints instead of floats etc.
TODO: for the ints that are also only positive (expid, etc), use unsigned int
TODO: why is my expid negative?
TODO: add thing in front for plane parameter? say plane_
TODO: worth it to sort my columns instead of whatever craziness python does?
TODO: my fitted and image parameters are the same?!
"""

from __future__ import print_function, division
import argparse
import numpy as np
from focal_plane_routines import average_function
from time import asctime
import pyfits

##############################################################################
# argparse
##############################################################################

parser = argparse. \
    ArgumentParser(description=
                   'Fit image and dump results.')
parser.add_argument("-e",
                    dest="expid",
                    help="what image numbers will we fit now? Format is " +
                    "[12,34] for looking at expids 12 and 34.")
parser.add_argument("-i",
                    dest="input_directory",
                    default="['/nfs/slac/g/ki/ki18/cpd/focus/november_8/']",
                    help="in what directories are my results located?" +
                    " Format is ['/path/to/directory1/', " +
                    "'/path/to/directory2/']")
parser.add_argument("-o",
                    dest="output_directory",
                    default="/nfs/slac/g/ki/ki18/cpd/focus/november_8/",
                    help="where will the outputs go (modulo image number)")
parser.add_argument("-b",
                    dest="boxdiv",
                    default=0,
                    type=int,
                    help="Division of chips. 0 is full, 1 is one division...")
options = parser.parse_args()

args_dict = vars(options)
args_dict['expid'] = eval(args_dict['expid'])
args_dict['input_directory'] = eval(args_dict['input_directory'])

if args_dict['boxdiv'] == 0:
    number_entries = 61
else:
    number_entries = 2 ** (2 * args_dict['boxdiv'] - 1)

time = asctime()

# go through all the inputs

plane_keys = ['x', 'y', 'x_box', 'y_box', 'n', 'fwhm',
              'e0', 'e1', 'e2',
              'var_e0', 'var_e1', 'var_e2']
minuit_sub_dictionary=dict(
    args=['rzero', 'dz', 'dx', 'dy', 'xt', 'yt',
          'e1', 'e2', 'z05d', 'z06d',
          'z07x', 'z07y', 'z08x', 'z08y'],
    errors=['rzero', 'dz', 'dx', 'dy', 'xt', 'yt',
            'e1', 'e2', 'z05d', 'z06d',
            'z07x', 'z07y', 'z08x', 'z08y'],
    mnstat=['amin', 'edm', 'errdef', 'nvpar',
            'nparx', 'icstat'],
    status=['migrad_ierflg',
            'deltatime', 'force_derivatives',
            'strategy', 'max_iterations',
            'tolerance', 'verbosity',
            'nCalls', 'nCallsDerivative'])

# generate dictionary of columns
pyfits_dict = {}
for plane_key in plane_keys:
    pyfits_dict.update({plane_key + '_image': dict(
        name=plane_key + '_image',
        array=[],
        format='{0}D'.format(number_entries)
        )})
    pyfits_dict.update({plane_key + '_fitted': dict(
        name=plane_key + '_fitted',
        array=[],
        format='{0}D'.format(number_entries)
        )})
for minuit_key in minuit_sub_dictionary:
    for minuit_sub_key in minuit_sub_dictionary[minuit_key]:
        pyfits_dict.update({minuit_key + '_' + minuit_sub_key: dict(
            name=minuit_key + '_' + minuit_sub_key,
            array=[],
            format='E',
            )})
pyfits_dict.update({
    'expid': dict(
        name='expid', array=[], format='D'),
    'time': dict(
        name='time', array=[], format='24A'),
    'vector_whisker_image': dict(
        name='vector_whisker_image', array=[], format='D'),
    'vector_whisker_fitted': dict(
        name='vector_whisker_fitted', array=[], format='D'),
    'scalar_whisker_image': dict(
        name='scalar_whisker_image', array=[], format='D'),
    'scalar_whisker_fitted': dict(
        name='scalar_whisker_fitted', array=[], format='D'),
    'average_fwhm_image': dict(
        name='average_fwhm_image', array=[], format='D'),
    'average_fwhm_fitted': dict(
        name='average_fwhm_fitted', array=[], format='D'),
    })


print(len(args_dict['expid']))
for iterator in xrange(len(args_dict['expid'])):
    directory = args_dict['input_directory'][iterator]
    expid = args_dict['expid'][iterator]
    print(expid, iterator)

    # comparison
    path_comparison = directory + '{0:08d}_image_plane.npy'.format(expid)
    # get the items from the dictionary
    comparison_dict = np.load(path_comparison).item()
    comparison_dict_use = {}
    for plane_key in plane_keys:
        pyfits_dict[plane_key + '_image']['array'].append(
            comparison_dict[plane_key])

    vector_whisker_image = average_function(
        comparison_dict, np.mean, 'vector_whisker')
    scalar_whisker_image = average_function(
        comparison_dict, np.mean, 'scalar_whisker')
    fwhm_image = average_function(
        comparison_dict, np.mean, 'fwhm')

    # fitted
    path_fitted = directory + '{0:08d}_fitted_plane.npy'.format(expid)
    # get the items from the dictionary
    fitted_dict = np.load(path_fitted).item()
    fitted_dict_use = {}
    for plane_key in plane_keys:
        pyfits_dict[plane_key + '_fitted']['array'].append(
            fitted_dict[plane_key])

    vector_whisker_fitted = average_function(
        fitted_dict, np.mean, 'vector_whisker')
    scalar_whisker_fitted = average_function(
        fitted_dict, np.mean, 'scalar_whisker')
    fwhm_fitted = average_function(
        fitted_dict, np.mean, 'fwhm')

    # minuit results
    path_minuit = directory + '{0:08d}_minuit_results.npy'.format(expid)
    minuit_dict = np.load(path_minuit).item()
    for minuit_key in minuit_sub_dictionary:
        for minuit_sub_key in minuit_sub_dictionary[minuit_key]:
            key = minuit_key + '_' + minuit_sub_key
            pyfits_dict[key]['array'].append(
                minuit_dict[minuit_key][minuit_sub_key])

    user_dict = {'expid': expid,
                 'time': time,
                 'vector_whisker_image': vector_whisker_image,
                 'vector_whisker_fitted': vector_whisker_fitted,
                 'scalar_whisker_image': scalar_whisker_image,
                 'scalar_whisker_fitted': scalar_whisker_fitted,
                 'average_fwhm_image': fwhm_image,
                 'average_fwhm_fitted': fwhm_fitted}
    for user_key in user_dict:
        pyfits_dict[user_key]['array'].append(
            user_dict[user_key])

# create the columns
columns = []
for key in pyfits_dict:
    columns.append(pyfits.Column(**pyfits_dict[key]))
tbhdu = pyfits.new_table(columns)

tbhdu.writeto(args_dict['output_directory'] + 'results.fits')
