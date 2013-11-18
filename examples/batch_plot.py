#!/usr/bin/env python
# batch_plot.py
from __future__ import print_function, division
import argparse
import numpy as np
from plot_wavefront import focal_plane_plot, collect_images
from decam_csv_routines import collect_dictionary_results, collect_fit_results
from focal_plane_routines import average_function, \
    ellipticity_variance_to_whisker_variance, ellipticity_to_whisker

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

# go through all the inputs
# collect into one big csv, and also collect averages
comparison_all = {'expid': [], 'n': [],
                  'x_box': [], 'y_box': [],
                  'e0': [], 'var_e0': [],
                  'e1': [], 'var_e1': [],
                  'e2': [], 'var_e2': []}
fitted_all = {'expid': [], 'n': [],
              'x_box': [], 'y_box': [],
              'e0': [], 'var_e0': [],
              'e1': [], 'var_e1': [],
              'e2': [], 'var_e2': []}

file_list = []

for iterator in xrange(len(args_dict['expid'])):
    directory = args_dict['input_directory'][iterator]
    expid = args_dict['expid'][iterator]

    # comparison
    path_comparison = directory + '{0:08d}_image_plane.npy'.format(expid)
    path_comparison_out = args_dict['output_directory'] + 'image_plane.csv'
    # get the items from the dictionary
    comparison_dict = np.load(path_comparison).item()
    comparison_dict.update({'expid': [expid] * len(comparison_dict['n'])})
    collect_dictionary_results(path_comparison_out, item_dict=comparison_dict)

    vector_whisker_image = average_function(
        comparison_dict, np.mean, 'vector_whisker')
    scalar_whisker_image = average_function(
        comparison_dict, np.mean, 'scalar_whisker')
    fwhm_image = average_function(
        comparison_dict, np.mean, 'fwhm')

    for key in comparison_all.keys():
        for key_i in xrange(len(comparison_dict[key])):
            comparison_all[key].append(comparison_dict[key][key_i])

    # fitted
    path_fitted = directory + '{0:08d}_fitted_plane.npy'.format(expid)
    path_fitted_out = args_dict['output_directory'] + 'fitted_plane.csv'
    # get the items from the dictionary
    fitted_dict = np.load(path_fitted).item()
    fitted_dict.update({'expid': [expid] * len(fitted_dict['n'])})
    collect_dictionary_results(path_fitted_out, item_dict=fitted_dict)

    vector_whisker_fitted = average_function(
        fitted_dict, np.mean, 'vector_whisker')
    scalar_whisker_fitted = average_function(
        fitted_dict, np.mean, 'scalar_whisker')
    fwhm_fitted = average_function(
        fitted_dict, np.mean, 'fwhm')

    for key in fitted_all.keys():
        for key_i in xrange(len(fitted_dict[key])):
            fitted_all[key].append(fitted_dict[key][key_i])

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
                 'vector_whisker_image': [vector_whisker_image],
                 'vector_whisker_fitted': [vector_whisker_fitted],
                 'scalar_whisker_image': [scalar_whisker_image],
                 'scalar_whisker_fitted': [scalar_whisker_fitted],
                 'fwhm_image': [fwhm_image],
                 'fwhm_fitted': [fwhm_fitted]}

    collect_fit_results(path_minuit, path_minuit_out,
                        user_dict)

    # plots
    file_list.append(args_dict['output_directory'] + '{0:08d}_'.format(expid))
    # e0
    path_e0_plot = args_dict['output_directory'] + \
        '{0:08d}_e0.pdf'.format(expid)
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

    # ellipticity
    path_ellipticity_plot = args_dict['output_directory'] + \
        '{0:08d}_ellipticity.pdf'.format(expid)
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

    # whisker
    path_whisker_plot = args_dict['output_directory'] + \
        '{0:08d}_whisker.pdf'.format(expid)
    # convert ellipticity to whisker
    u_var_comparison, w_var_comparison = \
        ellipticity_variance_to_whisker_variance(
            u_comparison, v_comparison, u_var_comparison, v_var_comparison)
    u_comparison, v_comparison = ellipticity_to_whisker(
        u_comparison, v_comparison)
    # vector whisker
    u_ave_comparison, v_ave_comparison = ellipticity_to_whisker(
        u_ave_comparison, v_ave_comparison)
    figure_whisker, axis_whisker = focal_plane_plot(
        x=x_comparison, y=y_comparison,
        u=u_comparison, v=v_comparison,
        u_ave=u_ave_comparison, v_ave=v_ave_comparison,
        u_var=u_var_comparison, v_var=v_var_comparison,
        color='r')
    # do fitted
    u_var_fitted, w_var_fitted = \
        ellipticity_variance_to_whisker_variance(
            u_fitted, v_fitted, u_var_fitted, v_var_fitted)
    u_fitted, v_fitted = ellipticity_to_whisker(
        u_fitted, v_fitted)
    # vector whisker
    u_ave_fitted, v_ave_fitted = ellipticity_to_whisker(
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

    # whisker rotated
    path_whisker_rotated_plot = args_dict['output_directory'] + \
        '{0:08d}_whisker_rotated.pdf'.format(expid)
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

# combine all the graphs
collect_images(file_list,
               args_dict['output_directory'],
               graphs_list=['e0', 'ellipticity', 'whisker', 'whisker_rotated'],
               rate=0)


# convert the alls into arrays
for key in comparison_all.keys():
    comparison_all[key] = np.array(comparison_all[key])
for key in fitted_all.keys():
    fitted_all[key] = np.array(fitted_all[key])
# average plot
