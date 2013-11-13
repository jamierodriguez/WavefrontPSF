#!/usr/bin/env python
# batch_fit.py
from __future__ import print_function, division
import argparse
import numpy as np
from minuit_fit import Minuit_Fit
from iminuit import describe
from os import path, makedirs
from focal_plane_routines import average_dictionary, variance_dictionary, \
    chi2, minuit_dictionary, fwhm_to_rzero
from decam_csv_routines import generate_hdu_lists
from focal_plane import FocalPlane

"""
If you submit this script to the batch queue, you will get fits for a given
image.
"""

#TODO: output the args parse commands, too

# argparse
parser = argparse. \
    ArgumentParser(description=
                   'Fit image and dump results.')
parser.add_argument("-e",
                    dest="expid",
                    type=int,
                    help="what image number will we fit now?")
parser.add_argument("-c",
                    dest="csv",
                    default='/afs/slac.stanford.edu/u/ki/cpd/makedonuts/' +
                        'db20120914to20130923.csv',
                    help="where is the csv of the image data located")
parser.add_argument("-m",
                    dest="mesh",
                    default='/u/ec/roodman/Astrophysics/Donuts/Meshes/',
                    help="where is the meshes are located")
parser.add_argument("-o",
                    dest="output_directory",
                    default="/nfs/slac/g/ki/ki18/cpd/focus/november_8/",
                    help="where will the outputs go (modulo image number)")
parser.add_argument("-s",
                    dest="max_stars",
                    default=500,
                    type=int,
                    help="How many stars total?")
parser.add_argument("-a",
                    dest="subav",
                    default=0,
                    type=int,
                    help="subtract the average off?")
parser.add_argument("-t",
                    dest="catalogs",
                    default='/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',
                    help='directory containing the catalogs')
parser.add_argument("-b",
                    dest="boxdiv",
                    default=0,
                    type=int,
                    help="Division of chips. 0 is full, 1 is one division...")
options = parser.parse_args()

args_dict = vars(options)

# initialize all the values

subav = args_dict['subav']
boxdiv = args_dict['boxdiv']
max_stars = args_dict['max_stars']


# load up image data
csv = np.recfromcsv(args_dict['csv'], usemask=True)
image_data = csv[csv['expid'] == args_dict['expid']]

# find the locations of the catalog files
path_catalogs = args_dict['catalogs']
list_catalogs, list_fits_extension, list_chip = \
    generate_hdu_lists(args_dict['expid'], path_catalogs)

path_mesh = args_dict['mesh']

# create the focalplane
FP = FocalPlane(image_data=image_data,
                list_catalogs=list_catalogs,
                list_fits_extension=list_fits_extension,
                list_chip=list_chip,
                path_mesh=path_mesh,
                max_stars=max_stars)

comparison_average = average_dictionary(FP.comparison_dict_all, FP.average,
    boxdiv=boxdiv, subav=subav)

chi_weights = {
    'e0': 1.,
    'e1': 1.,
    'e2': 1.,
    }
order_dict = {
    'x2': {'p': 2, 'q': 0},
    'y2': {'p': 0, 'q': 2},
    'xy': {'p': 1, 'q': 1},
    }

def FP_func(dz, e1, e2, rzero, dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y):
    in_dict = locals().copy()

    # assumes FP, order_dict, chi_weights, boxdiv, subav, comparison_average
    # are defined before this!

    # go through the keys and make sure there are no nans
    for key in in_dict.keys():
        if np.isnan(in_dict[key]).any():
            # if there is a nan, don't even bother calling, just return a
            # bigass chi2
            return 1e20

    # get current iteration
    moments = FP.plane(in_dict, coords=FP.coords, order_dict=order_dict)

    # convert to average
    moments_average = average_dictionary(moments, FP.average, boxdiv=boxdiv,
        subav=subav)

    # create var_dict
    var_dict = variance_dictionary(
        data=comparison_average,
        keys=chi_weights.keys(),
        var_type=0)

    # get chi
    chi_dict = chi2(data_a=moments_average,
                    data_b=comparison_average,
                    chi_weights=chi_weights,
                    var_dict=var_dict)
    chi2_val = chi_dict['chi2']

    # update the chi2 by *= 1. / (Nobs - Nparam)
    chi2_val *= 1. / (len(moments_average['fwhm']) - len(in_dict.keys()))
    # divide another bit by the sum of the chi_weights
    # not so sure about this one...
    chi2_val *= 1. / sum([chi_weights[i] for i in chi_weights])

    return chi2_val

# set up minuit
par_names = describe(FP_func)
verbosity = 3
force_derivatives = 1
grad_dict = dict(h_base=1e-1)
strategy = 1
tolerance = 40
max_iterations = len(par_names) * 100

minuit_dict = minuit_dictionary(par_names)
# set up initial guesses
image_dictionary = FP.image_zernike_corrections(FP.image_data)
for key in par_names:
    if (key == 'e1') + (key == 'e2'):
        continue
    elif (key == 'rzero'):
        rzero_estimate = fwhm_to_rzero(FP.image_data['qrfwhm'])
        minuit_dict[key] = rzero_estimate.data[0]
    elif key == 'fwhm':
        minuit_dict[key] = FP.image_data['qrfwhm'].data[0]
    else:
        minuit_dict[key] = image_dictionary[key][0]

minuit_fit = Minuit_Fit(FP_func, minuit_dict, par_names, grad_dict=grad_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)
minuit_fit.setupFit()
minuit_fit.doFit()
minuit_results = minuit_fit.outFit()

# print results
error_dict = minuit_results['errors']
in_dict = minuit_results['args']
print('key\tfit\t\terror\t\tminuit\t\tfirst')
for key in par_names:
    print(key, '\t{0: .4e}'.format(in_dict[key]),
          '\t{0: .4e}'.format(error_dict[key]),
          '\t{0: .4e}'.format(minuit_dict[key]))

# print the sigma stuff here, too
for key in par_names:
    if abs(in_dict[key]) > 3 * abs(error_dict[key]):
        print(key, '\t', '{0:.4e}'.format(in_dict[key]), '\t',
              '{0:.4e}'.format(error_dict[key]))
    else:
        print(key, '\t', '{0:.4e}'.format(in_dict[key]), '\t',
              '{0:.4e}'.format(error_dict[key]), '\t', 'NOT')

# save FP and the results
output_directory = args_dict['output_directory']
# check the output_directory exists
if not path.exists(output_directory):
    makedirs(output_directory)

# save the minuit_results
np.save(output_directory + '{0:08d}_minuit_results'.format(
    args_dict['expid']), minuit_results)
# save the coords
np.save(output_directory + '{0:08d}_coords'.format(args_dict['expid']),
    FP.coords)
# save the comparison dictionary (averaged)
np.save(output_directory + '{0:08d}_comparison_plane'.format(
        args_dict['expid']),
    comparison_average)
# save the outputted focal plane
moments = FP.plane(in_dict, coords=FP.coords, order_dict=order_dict)
# convert to average
moments_average = average_dictionary(moments, FP.average, boxdiv=boxdiv,
    subav=subav)
np.save(output_directory + '{0:08d}_fitted_plane'.format(
        args_dict['expid']), moments_average)
## # save the FP object
## FP.save(output_directory + '{0:08d}_FP.pkl'.format(args_dict['expid']))
