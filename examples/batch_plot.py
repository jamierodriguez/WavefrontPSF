#!/usr/bin/env python
# batch_plot.py
from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
from plot_wavefront import focal_plane_plot, collect_images
from decam_csv_routines import collect_dictionary_results, collect_fit_results
from focal_plane_routines import \
    ellipticity_variance_to_whisker_variance, ellipticity_to_whisker
from matplotlib.pyplot import close
from os import path, makedirs

"""
Include the locations of the moments (both fitted and comparison)

This file will take the results and plot them as well as collate everything
together into csv files. Those can also be plotted?

TODO: split the plotting and collating
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

# go through all the inputs

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

    # fitted
    path_fitted = directory + '{0:08d}_fitted_plane.npy'.format(expid)
    path_fitted_out = args_dict['output_directory'] + 'fitted_plane.csv'
    # get the items from the dictionary
    fitted_dict = np.load(path_fitted).item()

    # plots
    out_plots = args_dict['output_directory'] + '{0:08d}/'.format(expid)
    if not path.exists(out_plots):
        makedirs(out_plots)
    # e0
    path_e0_plot = out_plots + 'e0.pdf'
    # do the comparison first
    x_comparison = comparison_dict['x_box'] - 5
    y_comparison = comparison_dict['y_box']
    u_comparison = np.array([0] * len(x_comparison))
    v_comparison = comparison_dict['e0']
    u_var_comparison = comparison_dict['var_e0']  # this is hackish
    v_var_comparison = comparison_dict['var_e0']
    u_ave_comparison = 0
    v_ave_comparison = np.mean(v_comparison)
    figure_e0, axis_e0 = focal_plane_plot(
        x=x_comparison, y=y_comparison,
        u=u_comparison, v=v_comparison,
        u_ave=u_ave_comparison, v_ave=v_ave_comparison,
        u_var=u_var_comparison, v_var=v_var_comparison,
        color='r',
        scale=10 / 1e-1,
        quiverkey_dict={'title': r'$1 \times 10^{-1}$ arcsec$^{2}$',
                        'value': 2 * 10 / 1e-1},
        artpatch=2)
    # do the fitted
    x_fitted = fitted_dict['x_box'] + 5
    y_fitted = fitted_dict['y_box']
    u_fitted = np.array([0] * len(x_fitted))
    v_fitted = fitted_dict['e0']
    u_var_fitted = fitted_dict['var_e0']  # this is hackish
    v_var_fitted = fitted_dict['var_e0']
    u_ave_fitted = 0
    v_ave_fitted = np.mean(v_fitted)
    figure_e0, axis_e0 = focal_plane_plot(
        x=x_fitted, y=y_fitted,
        u=u_fitted, v=v_fitted,
        u_ave=u_ave_fitted, v_ave=v_ave_fitted,
        u_var=u_var_fitted, v_var=v_var_fitted,
        focal_figure=figure_e0,
        focal_axis=axis_e0,
        color='k',
        scale=10 / 1e-1,
        quiverkey_dict={'title': r'$1 \times 10^{-1}$ arcsec$^{2}$',
                        'value': 2 * 10 / 1e-1},
        artpatch=2)
    figure_e0.savefig(path_e0_plot)
    close()

    # ellipticity
    path_ellipticity_plot = out_plots + 'ellipticity.pdf'
    # do the comparison first
    x_comparison = comparison_dict['x_box']
    y_comparison = comparison_dict['y_box']
    u_comparison = comparison_dict['e1']
    v_comparison = comparison_dict['e2']
    u_var_comparison = comparison_dict['var_e1']
    v_var_comparison = comparison_dict['var_e2']
    u_ave_comparison = np.mean(u_comparison)
    v_ave_comparison = np.mean(v_comparison)
    figure_ellipticity, axis_ellipticity = focal_plane_plot(
        x=x_comparison, y=y_comparison,
        u=u_comparison, v=v_comparison,
        u_ave=u_ave_comparison, v_ave=v_ave_comparison,
        u_var=u_var_comparison, v_var=v_var_comparison,
        color='r',
        scale=10 / 5e-3,
        quiverkey_dict={'title': r'$5 \times 10^{-3}$ arcsec$^{2}$',
                        'value': 10 / 5e-3},
        artpatch=1)
    # do the fitted
    x_fitted = fitted_dict['x_box']
    y_fitted = fitted_dict['y_box']
    u_fitted = fitted_dict['e1']
    v_fitted = fitted_dict['e2']
    u_var_fitted = fitted_dict['var_e1']
    v_var_fitted = fitted_dict['var_e2']
    u_ave_fitted = np.mean(u_fitted)
    v_ave_fitted = np.mean(v_fitted)
    figure_ellipticity, axis_ellipticity = focal_plane_plot(
        x=x_fitted, y=y_fitted,
        u=u_fitted, v=v_fitted,
        u_ave=u_ave_fitted, v_ave=v_ave_fitted,
        u_var=u_var_fitted, v_var=v_var_fitted,
        focal_figure=figure_ellipticity,
        focal_axis=axis_ellipticity,
        color='k',
        scale=10 / 5e-3,
        quiverkey_dict={'title': r'$5 \times 10^{-3}$ arcsec$^{2}$',
                        'value': 10 / 5e-3},
        artpatch=1)
    figure_ellipticity.savefig(path_ellipticity_plot)
    close()

    # whisker
    path_whisker_plot = out_plots + 'whisker.pdf'
    # convert ellipticity to whisker
    u_var_comparison, v_var_comparison = \
        ellipticity_variance_to_whisker_variance(
            u_comparison, v_comparison, u_var_comparison, v_var_comparison)
    u_comparison, v_comparison, w, phi = ellipticity_to_whisker(
        u_comparison, v_comparison)
    # vector whisker
    u_ave_comparison, v_ave_comparison, w, phi = ellipticity_to_whisker(
        u_ave_comparison, v_ave_comparison)
    figure_whisker, axis_whisker = focal_plane_plot(
        x=x_comparison, y=y_comparison,
        u=u_comparison, v=v_comparison,
        u_ave=u_ave_comparison, v_ave=v_ave_comparison,
        u_var=u_var_comparison, v_var=v_var_comparison,
        color='r')
    # do fitted
    u_var_fitted, v_var_fitted = \
        ellipticity_variance_to_whisker_variance(
            u_fitted, v_fitted, u_var_fitted, v_var_fitted)
    u_fitted, v_fitted, w, phi = ellipticity_to_whisker(
        u_fitted, v_fitted)
    # vector whisker
    u_ave_fitted, v_ave_fitted, w, phi = ellipticity_to_whisker(
        u_ave_fitted, v_ave_fitted)
    figure_whisker, axis_whisker = focal_plane_plot(
        x=x_fitted, y=y_fitted,
        u=u_fitted, v=v_fitted,
        u_ave=u_ave_fitted, v_ave=v_ave_fitted,
        u_var=u_var_fitted, v_var=v_var_fitted,
        focal_figure=figure_whisker,
        focal_axis=axis_whisker,
        color='k')
    figure_whisker.savefig(path_whisker_plot)
    close()

    # whisker rotated
    path_whisker_rotated_plot = out_plots + 'whisker_rotated.pdf'
    # do the comparison first
    x_comparison = comparison_dict['x_box'] - 5
    y_comparison = comparison_dict['y_box']
    figure_whisker_rotated, axis_whisker_rotated = focal_plane_plot(
        x=x_comparison, y=y_comparison,
        u=np.array([0] * len(x_comparison)),
        v=np.array([1] * len(x_comparison)),
        u_ave=0, v_ave=1,
        u_var=[], v_var=[],
        color='r',
        scale=10 / 1,
        quiverkey_dict={'title': r'',
                        'value': 0},
        artpatch=2)

    # do the fitted

    theta = np.arctan2(u_comparison, v_comparison)
    mag = np.sqrt(np.square(u_comparison) + np.square(v_comparison))
    u_fitted_prime = (np.cos(theta) * u_fitted -
                      np.sin(theta) * v_fitted) / mag
    v_fitted_prime = (np.sin(theta) * u_fitted +
                      np.cos(theta) * v_fitted) / mag

    theta_ave = np.arctan2(u_ave_comparison, v_ave_comparison)
    mag_ave = np.sqrt(np.square(u_ave_comparison) +
                      np.square(v_ave_comparison))
    u_ave_fitted_prime = (np.cos(theta_ave) * u_ave_fitted -
                          np.sin(theta_ave) * v_ave_fitted) / mag_ave
    v_ave_fitted_prime = (np.sin(theta_ave) * u_ave_fitted +
                          np.cos(theta_ave) * v_ave_fitted) / mag_ave

    x_fitted = fitted_dict['x_box'] + 5
    y_fitted = fitted_dict['y_box']
    figure_whisker_rotated, axis_whisker_rotated = focal_plane_plot(
        x=x_fitted, y=y_fitted,
        u=u_fitted_prime, v=v_fitted_prime,
        u_ave=u_ave_fitted_prime, v_ave=v_ave_fitted_prime,
        u_var=[], v_var=[],
        focal_figure=figure_whisker_rotated,
        focal_axis=axis_whisker_rotated,
        color='k',
        scale=10 / 1,
        quiverkey_dict={'title': r'',
                        'value': 0},
        artpatch=2)
    figure_whisker_rotated.savefig(path_whisker_rotated_plot)
    close('all')
