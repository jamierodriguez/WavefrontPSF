#!/usr/bin/env python
"""
File: batch_plot.py
Author: Chris Davis
Description: Include the locations of the moments (both fitted and comparison)

This file will take the results and plot them as well as collate everything
together into csv files. Those can also be plotted?

TODO: redo construction to use routines_plot.data_focal_plot
TODO: plot residual histograms and such

"""

from __future__ import print_function, division
import matplotlib
# the agg is so I can submit for batch jobs.
matplotlib.use('Agg')
matplotlib.rc('image', interpolation='none', origin='lower', cmap = 'gray_r')
import argparse
import numpy as np
from routines_plot import 
from routines_files import collect_dictionary_results, collect_fit_results
from routines import \
    ellipticity_variance_to_whisker_variance, ellipticity_to_whisker
from matplotlib.pyplot import close
from os import path, makedirs
from decamutil_cpd import decaminfo

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
                    default="['/nfs/slac/g/ki/ki18/cpd/focus/january_06/']",
                    help="in what directories are my results located?" +
                    " Format is ['/path/to/directory1/', " +
                    "'/path/to/directory2/']")
parser.add_argument("-m",
                    dest="minuit_results",
                    help="Where are the minuit results located?")
parser.add_argument("-o",
                    dest="output_directory",
                    default="/nfs/slac/g/ki/ki18/cpd/focus/january_06/",
                    help="where will the outputs go (modulo image number)")
options = parser.parse_args()

args_dict = vars(options)
args_dict['expid'] = eval(args_dict['expid'])
args_dict['input_directory'] = eval(args_dict['input_directory'])

minuit_dict = np.recfromcsv(args_dict['minuit_results'])

# go through all the inputs

print(len(args_dict['expid']))
for iterator in xrange(len(args_dict['expid'])):
    directory = args_dict['input_directory'][iterator]
    expid = args_dict['expid'][iterator]
    minuit_dict_i = minuit_dict[minuit_dict['user_expid'] == expid]
    ierflg = np.float64(minuit_dict_i['status_migrad_ierflg'])
    amin = np.float64(minuit_dict_i['mnstat_amin'])
    print(expid, iterator, ierflg, amin)

    # comparison
    path_comparison = directory + '{0:08d}_image_plane.npy'.format(expid)
    path_comparison_out = args_dict['output_directory'] + 'image_plane.csv'
    # get the items from the dictionary
    comparison_dict = np.load(path_comparison).item()

    # fitted
    path_fitted = directory + '{0:08d}_fitted_plane.npy'.format(expid)
    path_fitted_out = args_dict['output_directory'] + 'fitted_plane.csv'
    # get the items from the dictionary
    fitted_dict = np.load(path_fitted).item()

    # plots
    out_plots = args_dict['output_directory']
    if not path.exists(out_plots):
        makedirs(out_plots)

    figures, axes, scales = data_focal_plot(comparison_dict, color='r')
    figures, axes, scales = data_focal_plot(fitted_dict, color='b',
                                            figures=figures,
                                            axes=axes,
                                            scales=scales)
    for fig_key in figures:
        figures[fig_key].savefig(out_plots + '{0:08d}_'.format(expid) +
                                 fig_key + '-focal.png')
    close('all')

    xedges = np.unique(fitted_dict['x_box'])
    dx = xedges[1] - xedges[0]
    xedges = np.append(xedges - dx, xedges[-1] + dx)
    yedges = np.unique(fitted_dict['y_box'])
    dy = yedges[1] - yedges[0]
    yedges = np.append(yedges - dy, yedges[-1] + dy)
    edges = [xedges, yedges]
    figures, axes, scales = data_hist_plot(fitted_dict, edges)
    for fig_key in figures:
        figures[fig_key].savefig(out_plots + '{0:08d}_'.format(expid) +
                                 fig_key + '-hist.png')
    close('all')
    figures, axes, scales = data_contour_plot(fitted_dict, edges)
    for fig_key in figures:
        figures[fig_key].savefig(out_plots + '{0:08d}_'.format(expid) +
                                 fig_key + '-contours.png')
    close('all')
