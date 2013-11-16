#!/usr/bin/env python
# batch_plot.py
from __future__ import print_function, division
import argparse
import numpy as np
from matplotlib import pyplot as plt
from plot_wavefront import focal_plane_plot, collect_images
from os import makedirs, path, remove

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
parser.add_argument("-c",
                    dest="csv",
                    default='/afs/slac.stanford.edu/u/ki/cpd/makedonuts/' +
                        'db20120914to20130923.csv',
                    help="where is the csv of the image data located")
parser.add_argument("-o",
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

# go through all the inputs
# collect into one big csv, and also collect averages
for iterator in xrange(len(args_dict['expid'])):
    directory = args_dict['input_directory'][iterator]
    expid = args_dict['expid'][iterator]

    # comparison
    path_comparison = [directory + '{0:08d}_image_plane.npy']
    path_comparison_out = directory + 'image_plane.csv'
    user_dict = {'expid': [expid]}
    collect_dictionary_results(
        path_comparison, path_comparison_out, user_dict=user_dict)
    # get the items from the dictionary
    comparison_values = np.load(path_comparison).item()

    # fitted

    # argparse

    # history

    # minuit results
    path_minuit = [directory + '{0:08d}_minuit_results.npy'.format(expid)]
    path_minuit_out = directory + 'minuit_results.csv'
    user_dict = {'expid': [expid],
                 'vector_whisker_image': [vector_whisker_image],
                 'vector_whisker_fitted': [vector_whisker_fitted],
                 'scalar_whisker_image': [scalar_whisker_image],
                 'scalar_whisker_fitted': [scalar_whisker_fitted],
                 'fwhm_image': [fwhm_image],
                 'fwhm_fitted': [fwhm_fitted]}

    # plots
# average plot

