#!/usr/bin/env python
"""
File: batch_fit.py
Author: Chris Davis
Description: If you submit this script to the batch queue, you will get fits
for a given image.

TODO: fix nan's!
TODO: take out limits to my fits? it slows things down and also puts roundoff problems in?
TODO: update code here from my notebook code
"""

from __future__ import print_function, division
import argparse
import numpy as np
from minuit_fit import Minuit_Fit
#from iminuit import describe
from os import path, makedirs
from routines import minuit_dictionary, fwhm_to_rzero, \
    image_zernike_corrections
from routines_files import generate_hdu_lists
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
                    help="where will the outputs go")
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
                    default='default',
                    help="String for filter conditions")
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
list_catalogs, list_fits_extension, list_chip = \
        generate_hdu_lists(args_dict['expid'], path_catalogs)

##############################################################################
# create the focalplane
##############################################################################

average = np.mean

FP = FocalPlane(list_catalogs=list_catalogs,
                list_fits_extension=list_fits_extension,
                list_chip=list_chip,
                path_mesh=args_dict['path_mesh'],
                mesh_name=args_dict['mesh_name'],
                boxdiv=args_dict['boxdiv'],
                max_samples_box=args_dict['max_samples_box'],
                conds=args_dict['conds'],
                subav=args_dict['subav'],
                average=average,
                )

comparison = FP.data

coords = FP.coords

chi_weights = {
    'e0': 1.,
    'e1': 1.,
    'e2': 1.,
    }

def FP_func(dz, e1, e2, rzero, dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y, z09d, z10d):
    in_dict_FP_func = locals().copy()

    # assumes FP, chi_weights, boxdiv, subav, comparison_average
    # are defined before this!

    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict_FP_func.keys():
        if np.isnan(in_dict_FP_func[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FP.remakedonut()
            return 1e20

    # get current iteration
    poles_i = FP.plane_averaged(
            in_dict_FP_func, coords=coords,
            average=average, boxdiv=args_dict['boxdiv'],
            )
    poles_i['e1'] += e1
    poles_i['e2'] += e2

    # get chi
    chi2 = 0
    for key in chi_weights:
        val_a = comparison[key]
        val_b = poles_i[key]
        var = comparison['var_{0}'.format(key)]
        weight = chi_weights[key]

        chi2_i = np.square(val_a - val_b) / var
        chi2 += np.sum(weight * chi2_i)

    if (chi2 < 0) + (np.isnan(chi2)):
        chi2 = 1e20

    # update the chi2 by *= 1. / (Nobs - Nparam)
    chi2 *= 1. / (len(poles_i['e1']) -
                  len(in_dict_FP_func.keys()))
    # divide another bit by the sum of the chi_weights
    # not so sure about this one...
    chi2 *= 1. / sum([chi_weights[i] for i in chi_weights])

    return chi2

##############################################################################
# set up minuit fit
##############################################################################

par_names = ['dz', 'e1', 'e2', 'rzero', 'dx', 'dy', 'xt', 'yt', 'z05d', 'z06d',
             'z07x', 'z07y', 'z08x', 'z08y', 'z09d', 'z10d']
verbosity = 3
force_derivatives = 1
strategy = 1
tolerance = 40
h_base = 1e-3
max_iterations = len(par_names) * 1000


# set up initial guesses
minuit_dict, h_dict = minuit_dictionary(par_names, h_base=h_base)
image_dictionary = image_zernike_corrections(image_data)
for key in par_names:
    if (key == 'e1') + (key == 'e2'):
        continue
    elif (key == 'rzero'):
        rzero_estimate = fwhm_to_rzero(image_data['qrfwhm'])
        minuit_dict[key] = rzero_estimate.data[0]
    elif key == 'fwhm':
        minuit_dict[key] = image_data['qrfwhm'].data[0]
    else:
        minuit_dict[key] = image_dictionary[key][0]

minuit_fit = Minuit_Fit(FP_func, minuit_dict, par_names=par_names,
                        h_dict=h_dict,
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
    comparison)

# save the outputted focal plane
moments = FP.plane_averaged(
    minuit_results['args'], coords=coords,
    average=average, boxdiv=args_dict['boxdiv'])
moments['e1'] += minuit_results['args']['e1']
moments['e2'] += minuit_results['args']['e2']

np.save(
    output_directory + '{0:08d}_fitted_plane'.format(
    args_dict['expid']), moments)

# save the FP history
np.save(
    output_directory + '{0:08d}_history'.format(
    args_dict['expid']), FP.history)

## # save the FP object
## FP.save(output_directory + '{0:08d}_FP.pkl'.format(args_dict['expid']))

# save the argsparse commands too
np.save(output_directory + '{0:08d}_args_dict'.format(
        args_dict['expid']), args_dict)
