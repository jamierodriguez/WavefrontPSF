#!/usr/bin/env python
"""
File: batch_fit.py
Author: Chris Davis
Description: If you submit this script to the batch queue, you will get fits
for a given image.

TODO: fix nan's!
"""

from __future__ import print_function, division
import argparse
import numpy as np
from minuit_fit import Minuit_Fit
#from iminuit import describe
from os import path, makedirs
from focal_plane_routines import average_dictionary, variance_dictionary, \
    chi2, minuit_dictionary, fwhm_to_rzero, second_moment_to_ellipticity, \
    second_moment_variance_to_ellipticity_variance, image_zernike_corrections
from decam_csv_routines import generate_hdu_lists, generate_hdu_lists_cpd
from focal_plane import FocalPlane

##############################################################################
# argparse
##############################################################################

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
                    dest="path_mesh",
                    default='/u/ec/roodman/Astrophysics/Donuts/Meshes/',
                    help="where is the meshes are located")
parser.add_argument("-n",
                    dest="mesh_name",
                    default="Science20120915s1v3_134239",
                    help="Name of mesh used.")
parser.add_argument("-o",
                    dest="output_directory",
                    default="/nfs/slac/g/ki/ki18/cpd/focus/november_8/",
                    help="where will the outputs go (modulo image number)")
parser.add_argument("-s",
                    dest="max_samples_box",
                    default=500,
                    type=int,
                    help="How many stars per box?")
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
## parser.add_argument("-r",
##                     dest="random",
##                     default=0,
##                     type=int,
##                     help="Do we take random locations on the chip or match to"
##                     + "stars? If the former, then the number of stars per"
##                     + "chip is taken to be int(max_samples / len(list_chip))")
parser.add_argument("-d",
                    dest="seed",
                    default=0,#np.random.randint(1e9),
                    type=int,
                    help="Set the seed we will use for random. This is "
                    + "apparently not thread safe. oh well.")
parser.add_argument("-f",
                    dest="conds",
                    default='eli',  # eli's filterings
                    help="String for filter conditions")
parser.add_argument("-cpd",
                    dest="cpd",
                    default=1,  # yes use my catalogs
                    type=int,
                    help="Use my catalogs or sextractor's?")
options = parser.parse_args()

args_dict = vars(options)

# seed the random number generator. the goal here is for me to be able to
# /exactly/ reproduce any run of the program and since I'm using random
# numbers, I need to save the seed.
np.random.seed(args_dict['seed'])

##############################################################################
# load up image data
##############################################################################

csv = np.recfromcsv(args_dict['csv'], usemask=True)
image_data = csv[csv['expid'] == args_dict['expid']]

# find the locations of the catalog files
path_catalogs = args_dict['catalogs']
if args_dict['cpd']:
    list_catalogs, list_fits_extension, list_chip = \
        generate_hdu_lists_cpd(args_dict['expid'], path_catalogs)
else:
    list_catalogs, list_fits_extension, list_chip = \
        generate_hdu_lists(args_dict['expid'], path_catalogs)

##############################################################################
# create the focalplane
##############################################################################

FP = FocalPlane(image_data=image_data,
                list_catalogs=list_catalogs,
                list_fits_extension=list_fits_extension,
                list_chip=list_chip,
                path_mesh=args_dict['path_mesh'],
                mesh_name=args_dict['mesh_name'],
                boxdiv=args_dict['boxdiv'],
                max_samples_box=args_dict['max_samples_box'],
                conds=args_dict['conds'],
                )

# convert comparison dict to ellipticities
e0_comparison, e0prime_comparison, e1_comparison, e2_comparison = \
    second_moment_to_ellipticity(FP.data['x2'],
                                 FP.data['y2'],
                                 FP.data['xy'])
FP.data.update(dict(
    e0=e0_comparison, e0prime=e0prime_comparison,
    e1=e1_comparison, e2=e2_comparison))
comparison_average = average_dictionary(FP.data, FP.average,
    boxdiv=args_dict['boxdiv'], subav=args_dict['subav'])

coords = FP.coords

## # check our variables are the right length
## check_average = average_dictionary(FP.comparison_dict, FP.average,
##     boxdiv=args_dict['boxdiv'], subav=args_dict['subav'])
## if len(check_average['x']) != len(comparison_average['x']):
##     # if they are not, use the random coords
##     coords = FP.coords_random

# create var_dict from comparison_average
var_dict = variance_dictionary(
    data=comparison_average,
    keys=['x', 'y', 'x2', 'y2', 'xy', 'e0', 'e1', 'e2'],
    var_type=0)
# also update comparison_average to include these vars
for key_var_dict in var_dict.keys():
    comparison_average.update(
        {'var_{0}'.format(key_var_dict):var_dict[key_var_dict]})

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
    in_dict_FP_func = locals().copy()

    # assumes FP, order_dict, chi_weights, boxdiv, subav, comparison_average
    # are defined before this!

    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict_FP_func.keys():
        if np.isnan(in_dict_FP_func[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FP.remakedonut()
            return 1e20

    # get current iteration
    moments_FP_func = FP.plane(in_dict_FP_func,
        coords=coords, order_dict=order_dict)

    # convert to ellipticities
    e0_moment, e0prime_moment, e1_moment, e2_moment = \
        second_moment_to_ellipticity(
            moments_FP_func['x2'],
            moments_FP_func['y2'],
            moments_FP_func['xy'])

    # add ellipticity corrections
    e1_moment += e1
    e2_moment += e2

    ellipticities = {'x': moments_FP_func['x'], 'y': moments_FP_func['y'],
                     'e0': e0_moment, 'e1': e1_moment, 'e2': e2_moment}

    # convert to average
    ellipticities_average = average_dictionary(ellipticities, FP.average,
        boxdiv=args_dict['boxdiv'], subav=args_dict['subav'])

    # get chi
    chi_dict = chi2(data_a=ellipticities_average,
                    data_b=comparison_average,
                    chi_weights=chi_weights,
                    var_dict=var_dict)
    chi2_val = chi_dict['chi2']

    # update the chi2 by *= 1. / (Nobs - Nparam)
    chi2_val *= 1. / (len(ellipticities_average['x']) -
                      len(in_dict_FP_func.keys()))
    # divide another bit by the sum of the chi_weights
    # not so sure about this one...
    chi2_val *= 1. / sum([chi_weights[i] for i in chi_weights])

    return chi2_val

##############################################################################
# set up minuit fit
##############################################################################
par_names = ['dz', 'e1', 'e2', 'rzero', 'dx', 'dy', 'xt', 'yt', 'z05d', 'z06d', 'z07x', 'z07y', 'z08x', 'z08y']
verbosity = 3
force_derivatives = 1
grad_dict = dict(h_base=1e-1)
strategy = 1
tolerance = 40
max_iterations = len(par_names) * 1000

# set up initial guesses
minuit_dict = minuit_dictionary(par_names)
image_dictionary = image_zernike_corrections(FP.image_data)
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

##############################################################################
# print results
##############################################################################

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

##############################################################################
# save the results
##############################################################################

output_directory = args_dict['output_directory']
# check the output_directory exists
if not path.exists(output_directory):
    makedirs(output_directory)

# save the minuit_results
np.save(
    output_directory + '{0:08d}_minuit_results'.format(
    args_dict['expid']), minuit_results)

# save the coords
np.save(
    output_directory + '{0:08d}_coords'.format(args_dict['expid']),
    coords)

# save the comparison dictionary
np.save(
    output_directory + '{0:08d}_image_plane'.format(args_dict['expid']),
    comparison_average)

# save the outputted focal plane
moments_results = FP.plane(in_dict, coords=coords, order_dict=order_dict)
# convert moments dict to ellipticities
e0_moments, e0prime_moments, e1_moments, e2_moments = \
    second_moment_to_ellipticity(moments_results['x2'],
                                 moments_results['y2'],
                                 moments_results['xy'])
moments_results.update(dict(
    e0=e0_moments, e0prime=e0prime_moments,
    e1=e1_moments, e2=e2_moments))
# and then average
moments_average = average_dictionary(moments_results, FP.average,
    boxdiv=args_dict['boxdiv'], subav=args_dict['subav'])
# create var_dict from moments_average
var_dict_results = variance_dictionary(
    data=moments_average,
    keys=['x', 'y', 'x2', 'y2', 'xy', 'e0', 'e1', 'e2'],
    var_type=0)

# also update moments_average to include these vars
for key_var_dict_results in var_dict_results.keys():
    moments_average.update(
        {'var_{0}'.format(key_var_dict_results):
         var_dict_results[key_var_dict_results]})
np.save(
    output_directory + '{0:08d}_fitted_plane'.format(
    args_dict['expid']), moments_average)

# save the FP history
np.save(
    output_directory + '{0:08d}_history'.format(
    args_dict['expid']), FP.history)

## # save the FP object
## FP.save(output_directory + '{0:08d}_FP.pkl'.format(args_dict['expid']))

# save the argsparse commands too
np.save(output_directory + '{0:08d}_args_dict'.format(
        args_dict['expid']), args_dict)
