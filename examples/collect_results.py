#!/usr/bin/env python
# collect_results.py
from __future__ import print_function, division
import argparse
import numpy as np
from decam_csv_routines import collect_dictionary_results, collect_fit_results
from focal_plane_routines import average_function
from plot_wavefront import collect_images
from time import asctime

"""
Include the locations of the moments (both fitted and comparison)

This file will take the results and plot them as well as collate everything
together into csv files. Those can also be plotted?
"""

# take as give that we have list_fitted_plane list_comparison_plane,
# list_minuit_results

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
options = parser.parse_args()

args_dict = vars(options)
args_dict['expid'] = eval(args_dict['expid'])
args_dict['input_directory'] = eval(args_dict['input_directory'])

time = asctime()

# go through all the inputs

file_list = []
plane_keys = ['x', 'y', 'x_box', 'y_box', 'n', 'fwhm',
              'e0', 'e1', 'e2',
              'var_e0', 'var_e1', 'var_e2']

print(len(args_dict['expid']))
for iterator in xrange(len(args_dict['expid'])):
    directory = args_dict['input_directory'][iterator]
    expid = args_dict['expid'][iterator]
    print(expid, iterator)

    # comparison
    path_comparison = directory + '{0:08d}_image_plane.npy'.format(expid)
    path_comparison_out = args_dict['output_directory'] + 'image_plane.csv'
    # get the items from the dictionary
    comparison_dict = np.load(path_comparison).item()
    comparison_dict_use = {}
    for plane_key in plane_keys:
        comparison_dict_use.update({plane_key: comparison_dict[plane_key]})
    comparison_dict_use.update({'expid': [expid] * len(comparison_dict['n'])})
    collect_dictionary_results(path_comparison_out,
                               item_dict=comparison_dict_use)

    vector_whisker_image = average_function(
        comparison_dict, np.mean, 'vector_whisker')
    scalar_whisker_image = average_function(
        comparison_dict, np.mean, 'scalar_whisker')
    fwhm_image = average_function(
        comparison_dict, np.mean, 'fwhm')

    # fitted
    path_fitted = directory + '{0:08d}_fitted_plane.npy'.format(expid)
    path_fitted_out = args_dict['output_directory'] + 'fitted_plane.csv'
    # get the items from the dictionary
    fitted_dict = np.load(path_fitted).item()
    fitted_dict_use = {}
    for plane_key in plane_keys:
        fitted_dict_use.update({plane_key: fitted_dict[plane_key]})
    fitted_dict_use.update({'expid': [expid] * len(fitted_dict['n'])})
    collect_dictionary_results(path_fitted_out,
                               item_dict=fitted_dict_use)

    vector_whisker_fitted = average_function(
        fitted_dict, np.mean, 'vector_whisker')
    scalar_whisker_fitted = average_function(
        fitted_dict, np.mean, 'scalar_whisker')
    fwhm_fitted = average_function(
        fitted_dict, np.mean, 'fwhm')

    # argparse
    path_args = directory + '{0:08d}_args_dict.npy'.format(expid)
    path_args_out = args_dict['output_directory'] + 'args_dict.csv'
    # get the items from the dictionary
    argparse_dict = np.load(path_args).item()
    argparse_dict = {args_key: [argparse_dict[args_key]]
        for args_key in argparse_dict}
    argparse_dict.update({'expid': [expid]})
    collect_dictionary_results(path_args_out, item_dict=argparse_dict)

    # minuit results
    path_minuit = [directory + '{0:08d}_minuit_results.npy'.format(expid)]
    path_minuit_out = args_dict['output_directory'] + 'minuit_results.csv'
    user_dict = {'expid': [expid],
                 'time': [time],
                 'vector_whisker_image': [vector_whisker_image],
                 'vector_whisker_fitted': [vector_whisker_fitted],
                 'scalar_whisker_image': [scalar_whisker_image],
                 'scalar_whisker_fitted': [scalar_whisker_fitted],
                 'fwhm_image': [fwhm_image],
                 'fwhm_fitted': [fwhm_fitted]}

    collect_fit_results(path_minuit, path_minuit_out,
                        user_dict)

    # plots
    out_plots = args_dict['output_directory'] + '{0:08d}/'.format(expid)
    file_list.append(out_plots)


# file_list.append(args_dict['output_directory'] + '{0:08d}/'.format(expid))
# combine all the graphs
print(file_list)
collect_images(file_list,
               args_dict['output_directory'],
               graphs_list=['e0', 'ellipticity', 'whisker', 'whisker_rotated'],
               rate=0)
