#!/usr/bin/env python
"""
File: best_analytic.py
Author: Chris Davis
Description: What are the best settings to minimize some parameter?

"""

from __future__ import print_function, division
import matplotlib
# the agg is so I can submit for batch jobs.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

from focal_plane_fit import FocalPlaneFit
from colors import blue_red
from minuit_fit import Minuit_Fit
from routines import minuit_dictionary, vary_one_parameter
from routines_moments import convert_moments
from routines_files import make_directory
from routines_plot import data_focal_plot, data_hist_plot

##############################################################################
# argparse
##############################################################################

parser = argparse. \
ArgumentParser(description=
               'Fit image and dump results.')
parser.add_argument("-b",
                    dest="boxdiv",
                    type=int,
                    default=1,
                    help="divisions of the box we care about")
parser.add_argument("-n",
                    dest="methodVal",
                    type=int,
                    default=20,
                    help="number of nearest neighbors used for reference mesh")
parser.add_argument("-v",
                    dest="verbose",
                    type=int,
                    default=1,
                    help="if > 0, print verbose statements")
parser.add_argument("-s",
                    dest="n_samples_box",
                    type=int,
                    default=10,
                    help="how many entries per box are we pulling")
parser.add_argument("-i",
                    dest="save_iter",
                    type=int,
                    default=10,
                    help="how often to save reference plot")
parser.add_argument("-o",
                    dest="output_directory",
                    default='/nfs/slac/g/ki/ki18/cpd/focus/outputs/',
                    help="where will the outputs go")
parser.add_argument("--chi_weights",
                    dest="chi_weights",
                    default="{" +
                            "'e0': 0.25, " +
                            "'e1': 1., " +
                            "'e2': 1., " +
                            "'delta1': 0.25, " +
                            "'delta2': 0.25, " +
                            "'zeta1': 0.05, " +
                            "'zeta2': 0.05, " +
                            "}",
                    help='what chi_weights will be used?')
parser.add_argument("--p_init",
                    dest="p_init",
                    default="{" +
                    ##     "'delta1' : 0, " +
                    ##     "'delta2' : 0, " +
                    ##     "'dx' :     0, " +
                    ##     "'dy' :     0, " +
                    ##     "'dz' :     0, " +
                    ##     "'e1' :     0, " +
                    ##     "'e2' :     0, " +
                           "'rzero' :  0.14, " +
                    ##     "'xt' :     0, " +
                    ##     "'yt' :     0, " +
                           "'z04d' :   0, " +
                           "'z04x' :   0, " +
                           "'z04y' :   0, " +
                           "'z05d' :   0, " +
                           "'z05x' :   0, " +
                           "'z05y' :   0, " +
                           "'z06d' :   0, " +
                           "'z06x' :   0, " +
                           "'z06y' :   0, " +
                           "'z07d' :   0, " +
                    ##     "'z07x' :   0, " +
                    ##     "'z07y' :   0, " +
                           "'z08d' :   0, " +
                    ##     "'z08x' :   0, " +
                    ##     "'z08y' :   0, " +
                           "'z09d' :   0, " +
                    ##     "'z09x' :   0, " +
                    ##     "'z09y' :   0, " +
                           "'z10d' :   0, " +
                    ##     "'z10x' :   0, " +
                    ##     "'z10y' :   0, " +
                           "'z11d' :   0, " +
                    ##     "'z11x' :   0, " +
                    ##     "'z11y' :   0, " +
                    ##     "'zeta1' :  0, " +
                    ##     "'zeta2' :  0, " +
                           "}",
                        help='initial guess for fit')
parser.add_argument("--analytic",
                    dest="analytic",
                    type=int,
                    default=1,
                    help='if > 0, use analytic model for fitting')
parser.add_argument("--par_names",
                    dest="par_names",
                    default="[" +
                    ##     "'delta1', " +
                    ##     "'delta2', " +
                    ##     "'dx', " +
                    ##     "'dy', " +
                    ##     "'dz', " +
                    ##     "'e1', " +
                    ##     "'e2', " +
                    ##     "'rzero', " +
                    ##     "'xt', " +
                    ##     "'yt', " +
                           "'z04d', " +
                           "'z04x', " +
                           "'z04y', " +
                           "'z05d', " +
                           "'z05x', " +
                           "'z05y', " +
                           "'z06d', " +
                           "'z06x', " +
                           "'z06y', " +
                           "'z07d', " +
                    ##     "'z07x', " +
                    ##     "'z07y', " +
                           "'z08d', " +
                    ##     "'z08x', " +
                    ##     "'z08y', " +
                           "'z09d', " +
                    ##     "'z09x', " +
                    ##     "'z09y', " +
                           "'z10d', " +
                    ##     "'z10x', " +
                    ##     "'z10y', " +
                           "'z11d', " +
                    ##     "'z11x', " +
                    ##     "'z11y', " +
                    ##     "'zeta1', " +
                    ##     "'zeta2', " +
                           "]",
                    help='par names we will use. If len(list(par_names)) == 0, just use p_init')

options = parser.parse_args()

args_dict = vars(options)

verbose = args_dict['verbose']

average = np.mean  # TODO: this may change!!!
boxdiv = args_dict['boxdiv']
n_samples_box = args_dict['n_samples_box']
subav = False
methodVal = (args_dict['methodVal'], 1.)

output_directory = args_dict['output_directory']
make_directory(output_directory)


##############################################################################
# set up fit
##############################################################################

chi_weights = eval(args_dict['chi_weights'])
p_init = eval(args_dict['p_init'])

# creat FocalPlaneFit object
FPF = FocalPlaneFit(methodVal=methodVal)

edges = FPF.decaminfo.getEdges(boxdiv=boxdiv)

coords_sample = FPF.random_coordinates(max_samples_box=n_samples_box,
                                       boxdiv=boxdiv)
FPF.coords = coords_sample

# give FPF some attributes
FPF.chi_weights = chi_weights

# define the fit function
chi2hist = []
FPF.history = []

if args_dict['analytic'] > 0:
    plane_func = FPF.analytic_plane_averaged
else:
    plane_func = FPF.plane_averaged

def FitFunc(in_dict):

    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict.keys():
        if np.isnan(in_dict[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FPF.remakedonut()
            return 1e20

    poles_i = plane_func(in_dict, coords=coords_sample,
                         average=average, boxdiv=boxdiv,
                         subav=subav)

    FPF.temp_plane = poles_i

    chi_dict = {'chi2': 0}
    for key in chi_weights.keys():
        val_a = poles_i[key]
        weight = chi_weights[key]

        chi2 = np.square(val_a)

        # update chi_dict
        chi_dict.update({key: chi2})
        chi_dict['chi2'] = chi_dict['chi2'] + np.sum(weight * chi2)

    # check whether chi_dict['chi2'] is an allowable number (finite
    # positive)
    if (chi_dict['chi2'] < 0) + (np.isnan(chi_dict['chi2'])):
        # if it isn't, make it really huge
        chi_dict['chi2'] = 1e20

    chi2hist.append(chi_dict)

    return chi_dict


par_names = eval(args_dict['par_names'])
if len(par_names) == 0:
    par_names = sorted(p_init.keys())
h_base = 1e-3
# set up initial guesses
minuit_dict, h_dict = minuit_dictionary(par_names, h_base=h_base)
for pkey in p_init:
    minuit_dict[pkey] = p_init[pkey]

# setup object
verbosity = 3
force_derivatives = 1
strategy = 1
tolerance = 1
max_iterations = len(par_names) * 100

minuit_fit = Minuit_Fit(FitFunc, minuit_dict, par_names=par_names,
                        h_dict=h_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)

##############################################################################
# do fit
##############################################################################

#minuit_fit.setupFit()
minuit_fit.doFit()

##############################################################################
# assess how well we did
##############################################################################

minuit_results = minuit_fit.outFit()
nCalls = (int(minuit_fit.nCalls / 1000) + 1) * 1000

# append to minuit results the argdict
minuit_results.update({'args_dict': args_dict})

np.save(output_directory + 'minuit_results', minuit_results)

# also save the coordinates
np.save(output_directory + 'coords_sample', coords_sample)

# save a plane!
in_dict = minuit_results['args']
poles_i = plane_func(in_dict, coords=coords_sample,
                     average=average, boxdiv=boxdiv,
                     subav=subav)

poles_i = convert_moments(poles_i)

np.save(output_directory + 'plane_fit', poles_i)

if verbose:
    for key in sorted(minuit_results['args']):
        print(key, p_init[key], minuit_results['args'][key],
              minuit_results['errors'][key])
    for key in sorted(minuit_results):
        if (key == 'correlation') + (key == 'covariance'):
            continue
        print(key, minuit_results[key])
    for key in sorted(minuit_results['args']):
        print("'" + key + "'", ":", minuit_results['args'][key], ",")

##############################################################################
# figures
##############################################################################

if verbose:
    fig = plt.figure()
    plt.imshow(minuit_results['correlation'], cmap=blue_red,
               interpolation='none', vmin=-1, vmax=1,
               origin='lower')
    plt.colorbar()
    fig.savefig(output_directory + '{0:04d}_correlation'.format(nCalls) +
                '.png')
    plt.close('all')

    chi2out = FitFunc(minuit_results['args'])
    # fit histograms
    figures, axes, scales = data_hist_plot(poles_i, edges)
    for fig in figures:
        axes[fig].set_title('{0}'.format(fig))
        figures[fig].savefig(output_directory +
           '{0:04d}_fit_histograms_{1}.png'.format(nCalls, fig))
    plt.close('all')

    # fit histograms
    figures, axes, scales = data_hist_plot(poles_i, edges, defaults=False)
    for fig in figures:
        axes[fig].set_title('{0}'.format(fig))
        figures[fig].savefig(output_directory +
           '{0:04d}_fit_histograms_unscaled_{1}.png'.format(nCalls, fig))
    plt.close('all')


    # do focal plane plots
    figures, axes, scales = data_focal_plot(poles_i, color='b',
                                            defaults=True)
    for fig in figures:
        axes[fig].set_title('{0}'.format(fig))
        figures[fig].savefig(output_directory +
           '{0:04d}_focal_plot_{1}.png'.format(nCalls, fig))
    plt.close('all')

    # do unscaled focal plane plots
    figures, axes, scales = data_focal_plot(poles_i, color='b',
                                            defaults=False)
    for fig in figures:
        axes[fig].set_title('{0}'.format(fig))
        figures[fig].savefig(output_directory +
           '{0:04d}_focal_plot_unscaled_{1}.png'.format(nCalls, fig))
    plt.close('all')

    # plots of zernikes
    zernikes = np.array(FPF.zernikes(in_dict, coords=coords_sample))
    zernike_dict = {'x': coords_sample[:, 0], 'y': coords_sample[:, 1]}
    zernike_keys = []
    for zi in xrange(11):
        zi_key = 'z{0:02d}'.format(zi + 1)
        zernike_dict.update({zi_key: zernikes[:, zi]})
        zernike_keys.append(zi_key)
    figures, axes, scales = data_hist_plot(zernike_dict, edges,
                                           keys=zernike_keys)
    for fig in figures:
        axes[fig].set_title('{0}'.format(fig))
        figures[fig].savefig(output_directory +
           '{0:04d}_fit_histograms_{1}.png'.format(nCalls, fig))
    plt.close('all')

    fig = plt.figure(figsize=(12, 12), dpi=300)
    for key in sorted(chi2hist[-1].keys()):
        x = range(len(chi2hist))
        y = []
        for i in x:
            if key == 'chi2':
                y.append(np.sum(chi2hist[i][key]))
            else:
                y.append(np.sum(chi2hist[i][key] * chi_weights[key]))
        if key == 'chi2':
            plt.plot(x, np.log10(y), 'k--', label=key)
        else:
            plt.plot(x, np.log10(y), label=key)
    plt.legend()
    fig.savefig(output_directory + '{0:04d}_chi2hist'.format(nCalls) + '.png')
    plt.close('all')

    # plot values
    for key in sorted(FPF.history[-1].keys()):
        fig = plt.figure(figsize=(12, 12), dpi=300)
        x = range(len(FPF.history))
        y = []
        for i in x:
            y.append(np.sum(FPF.history[i][key]))
        plt.plot(x, y, label=key)
        plt.title('{0}'.format(key))
        fig.savefig(output_directory + '{0:04d}_history_{1}'.format(nCalls,
                                                                    key)
                    + '.png')
        plt.close('all')

