#!/usr/bin/env python
"""
File: fit_analytic.py
Author: Chris Davis
Description: File that takes a DES catalog and fits via analytic function

TODO: Add plots from non-analytic
TODO: Consistency with image.py for plots and particularly error bars?
    soln: image used mean_trim, whereas this uses mean; these should be bigger
    error bars?

"""

from __future__ import print_function, division
import matplotlib
# the agg is so I can submit for batch jobs.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse

from focal_plane import FocalPlane
from focal_plane_fit import FocalPlaneFit
from colors import blue_red, blues_r, reds
from minuit_fit import Minuit_Fit
from routines import minuit_dictionary, mean_trim, vary_one_parameter, vary_two_parameters
from routines_moments import convert_moments
from routines_files import generate_hdu_lists, make_directory
from routines_plot import data_focal_plot, data_hist_plot, save_func_hists
from decamutil_cpd import decaminfo

##############################################################################
# argparse
##############################################################################

parser = argparse. \
ArgumentParser(description=
                'Fit image and dump results.')
parser.add_argument("-e", "--expid",
                    dest="expid",
                    type=int,
                    help="what image number will we fit now?")
parser.add_argument("-t", "--catalogs",
                    dest="catalogs",
                    default='/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',
                    help='directory containing the catalogs')
parser.add_argument("--name",
                    dest='name',
                    default='cat_cpd',
                    help='appended part of name for catalogs')
parser.add_argument("--extension",
                    dest='extension',
                    default=1,
                    type=int,
                    help='extension of hdu')
parser.add_argument("--conds",
                    dest="conds",
                    default='default',
                    help="filter conditions")
parser.add_argument("-b", "--boxdiv",
                    dest="boxdiv",
                    type=int,
                    default=1,
                    help="divisions of the box we care about")
parser.add_argument("-n", "--methodVal",
                    dest="methodVal",
                    type=int,
                    default=20,
                    help="number of nearest neighbors used for reference mesh")
parser.add_argument("-v", "--verbose",
                    dest="verbose",
                    type=int,
                    default=1,
                    help="if > 0, print verbose statements")
parser.add_argument("-s", "--n_samples_box",
                    dest="n_samples_box",
                    type=int,
                    default=10,
                    help="how many entries per box are we pulling")
parser.add_argument("-i", "--save_iter",
                    dest="save_iter",
                    type=int,
                    default=10,
                    help="how often to save reference plot")
parser.add_argument("-o", "--fits_directory",
                    dest="fits_directory",
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
                            "'delta1' : 0, " +
                            "'delta2' : 0, " +
                     ##     "'dx' :     0, " +
                     ##     "'dy' :     0, " +
                     ##     "'dz' :     0, " +
                            "'e1' :     0, " +
                            "'e2' :     0, " +
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
                            "'zeta1' :  0, " +
                            "'zeta2' :  0, " +
                            "}",
                        help='initial guess for fit')
parser.add_argument("--analytic",
                    dest="analytic",
                    type=int,
                    default=1,
                    help='if > 0, use analytic model for fitting')
parser.add_argument("--par_names",
                    dest="par_names",
                    default='[]',
                    help='par names we will use. If len(list(par_names)) == 0, just use p_init')


def do_fit(args):
    verbose = args['verbose']

    average = np.mean  # TODO: this may change!!!
    boxdiv = args['boxdiv']
    conds = args['conds']
    max_samples_box = 100000
    n_samples_box = args['n_samples_box']
    subav = False
    methodVal = (args['methodVal'], 1.)

    fits_directory = args['fits_directory']
    make_directory(fits_directory)

    expid = args['expid']


    ##############################################################################
    # create catalogs
    ##############################################################################

    path_base = args['catalogs']
    name = args['name']
    extension = args['extension']

    list_catalogs_base = \
        path_base + 'DECam_{0:08d}_'.format(expid)
    list_catalogs = [list_catalogs_base + '{0:02d}_{1}.fits'.format(i, name)
                     for i in xrange(1, 63)]
    list_catalogs.pop(60)
    # ccd 2 went bad too.
    if expid > 258804:
        list_catalogs.pop(1)

    list_chip = [[decaminfo().ccddict[i]] for i in xrange(1, 63)]
    list_chip.pop(60)
    # ccd 2 went bad too.
    if expid > 258804:
        list_chip.pop(1)

    # ccd 2 went bad too.
    if expid > 258804:
        list_fits_extension = [[extension]] * (63-3)
    else:
        list_fits_extension = [[extension]] * (63-2)

    ##############################################################################
    # set up fit
    ##############################################################################

    chi_weights = eval(args['chi_weights'])
    p_init = eval(args['p_init'])

    FP = FocalPlane(list_catalogs=list_catalogs,
                    list_fits_extension=list_fits_extension,
                    list_chip=list_chip,
                    boxdiv=boxdiv,
                    max_samples_box=max_samples_box,
                    conds=conds,
                    average=average,
                    subav=subav,
                    )
    edges = FP.decaminfo.getEdges(boxdiv=boxdiv)

    data_compare = FP.data
    data_unaveraged_compare = FP.data_unaveraged
    edges = FP.decaminfo.getEdges(boxdiv=boxdiv)

    # create subsample
    recdata_sample, extension_sample = FP.filter_number_in_box(
            recdata=FP.recdata,
            extension=FP.extension,
            max_samples_box=n_samples_box,
            boxdiv=boxdiv)
    data_sample, coords_sample, data_unaveraged_sample = FP.create_data(
        recdata=recdata_sample,
        extension=extension_sample,
        average=average,
        boxdiv=boxdiv,
        subav=False,
        )


    # creat FocalPlaneFit object
    FPF = FocalPlaneFit(methodVal=methodVal)

    # give FPF some attributes
    FPF.chi_weights = chi_weights
    FPF.coords = coords_sample

    # define the fit function
    chi2hist = []
    FPF.history = []

    if args['analytic'] > 0:
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

        chi2 = FPF.compare(poles_i, data_compare, var_dict=data_compare,
                           chi_weights=FPF.chi_weights)

        chi2hist.append(chi2)

        return chi2

    # define the save function
    def SaveFunc(steps, defaults=False):
        in_dict = {'steps': steps,
                   'state_history': FPF.history,
                   'chisquared_history': chi2hist,
                   'chi_weights': FPF.chi_weights,
                   'plane': FPF.temp_plane,
                   'reference_plane': data_compare,
                   'fits_directory': fits_directory,
                   'boxdiv': boxdiv,
                   'edges': edges,
                   'defaults': defaults,}
        save_func_hists(**in_dict)

        return

    par_names = eval(args['par_names'])
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
    save_iter = args['save_iter']
    max_iterations = len(par_names) * 100

    minuit_fit = Minuit_Fit(FitFunc, minuit_dict, par_names=par_names,
                            SaveFunc=SaveFunc,
                            save_iter=save_iter,
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
    minuit_results.update({'args': args})

    np.save(fits_directory + 'minuit_results', minuit_results)

    # also save the coordinates
    np.save(fits_directory + 'coords_sample', coords_sample)

    # save a plane!
    in_dict = minuit_results['args']
    poles_i = plane_func(in_dict, coords=coords_sample,
                         average=average, boxdiv=boxdiv,
                         subav=subav)

    poles_i = convert_moments(poles_i)

    np.save(fits_directory + 'plane_fit', poles_i)
    np.save(fits_directory + 'plane_compare', data_compare)

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
        fig.savefig(fits_directory + '{0:04d}_correlation'.format(nCalls) +
                    '.png')
        plt.close('all')

        chi2out = FitFunc(minuit_results['args'])
        SaveFunc(nCalls, defaults=True)
        # refrence histograms
        figures, axes, scales = data_hist_plot(data_compare, edges)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_reference_histograms_{1}.png'.format(nCalls, fig))
        plt.close('all')
        # refrence_sample histograms
        figures, axes, scales = data_hist_plot(data_unaveraged_compare, edges)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_sampled_histograms_{1}.png'.format(nCalls, fig))
        plt.close('all')
        # fit histograms
        figures, axes, scales = data_hist_plot(poles_i, edges)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_fit_histograms_{1}.png'.format(nCalls, fig))
        plt.close('all')

        # the above with defaults=False
        # refrence histograms
        figures, axes, scales = data_hist_plot(data_compare, edges, defaults=False)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_reference_histograms_unscaled_{1}.png'.format(nCalls, fig))
        plt.close('all')

        # refrence_sample histograms
        figures, axes, scales = data_hist_plot(data_unaveraged_compare, edges,
                                               defaults=False)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_sampled_histograms_unscaled_{1}.png'.format(nCalls, fig))
        plt.close('all')

        # fit histograms
        figures, axes, scales = data_hist_plot(poles_i, edges, defaults=False)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_fit_histograms_unscaled_{1}.png'.format(nCalls, fig))
        plt.close('all')

        # residual histograms
        data_residual = {'x_box': poles_i['x_box'], 'y_box': poles_i['y_box']}
        for key in poles_i.keys():
            if (key == 'x_box') + (key == 'y_box'):
                continue
            else:
                if key in data_compare.keys():
                    data_residual.update({key: data_compare[key] - poles_i[key]})
        figures, axes, scales = data_hist_plot(data_residual, edges, defaults=False)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_residual_histograms_{1}.png'.format(nCalls, fig))
        plt.close('all')


        # do focal plane plots
        figures, axes, scales = data_focal_plot(data_compare, color='r',
                                                defaults=True)
        figures, axes, scales = data_focal_plot(poles_i, color='b',
                                                figures=figures, axes=axes,
                                                scales=scales)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_focal_plot_{1}.png'.format(nCalls, fig))
        plt.close('all')

        # do unscaled focal plane plots
        figures, axes, scales = data_focal_plot(data_compare, color='r',
                                                defaults=False)
        figures, axes, scales = data_focal_plot(poles_i, color='b',
                                                figures=figures, axes=axes,
                                                scales=scales)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_focal_plot_unscaled_{1}.png'.format(nCalls, fig))
        plt.close('all')

        # do focal plane residual
        figures, axes, scales = data_focal_plot(data_residual, color='m',
                                                defaults=False)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
               '{0:04d}_focal_plot_residual_{1}.png'.format(nCalls, fig))
        plt.close('all')

        # plots of zernikes
        zernikes = np.array(FPF.zernikes(in_dict, coords=coords_sample))
        zernike_dict = {'x': coords_sample[:,0], 'y': coords_sample[:,1]}
        zernike_keys = []
        for zi in xrange(11):
            zi_key = 'z{0:02d}'.format(zi + 1)
            zernike_dict.update({zi_key: zernikes[:, zi]})
            zernike_keys.append(zi_key)
        figures, axes, scales = data_hist_plot(zernike_dict, edges,
                                               keys=zernike_keys)
        for fig in figures:
            axes[fig].set_title('{0:08d}: {1}'.format(expid, fig))
            figures[fig].savefig(fits_directory +
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
        plt.title('{0:08d}: chi2'.format(expid))
        fig.savefig(fits_directory + '{0:04d}_chi2hist'.format(nCalls) + '.png')
        plt.close('all')

        # plot values
        for key in sorted(FPF.history[-1].keys()):
            fig = plt.figure(figsize=(12, 12), dpi=300)
            x = range(len(FPF.history))
            y = []
            for i in x:
                y.append(np.sum(FPF.history[i][key]))
            plt.plot(x, y, label=key)
            plt.title('{0:08d}: {1}'.format(expid, key))
            fig.savefig(fits_directory + '{0:04d}_history_{1}'.format(nCalls,
                                                                        key)
                        + '.png')
            plt.close('all')

        # TODO: do varying parameters elsewhere?
        # do varying parameters
        for parameter in par_names:
            N = 11
            chi2, parameters = vary_one_parameter(parameter, FitFunc,
                                                  minuit_results['minuit'],
                                                  N = 11)
            fig = plt.figure(figsize=(12, 12))
            for key in sorted(chi2[-1].keys()):
                chi2par = [np.sum(chi2_i[key]) for chi2_i in chi2]
                if key == 'chi2':
                    plt.semilogy(parameters, chi2par, 'k--', label=key)
                    plt.plot([minuit_results['minuit'][parameter],
                              minuit_results['minuit'][parameter]],
                             [np.min(chi2par), np.max(chi2par)], 'k:', linewidth=2)
                else:
                    plt.semilogy(parameters, chi2par, '-', label=key)
            plt.legend()
            plt.title('{0:08d}: {1}'.format(expid, parameter))
            fig.savefig(fits_directory + '{0:04d}_vary_'.format(nCalls) +
                        parameter + '.png')
            plt.close('all')

        # two parameters
        for ij, parameter in enumerate(par_names[:-1]):
            for parameter2 in par_names[ij + 1:]:
                N = 7
                parameters_1, parameters_2, chi2, parameters, parameters2 = \
                        vary_two_parameters(parameter, parameter2,
                                            FitFunc,
                                            minuit_results['minuit'],
                                            N=N)
                for key in sorted(chi2[-1].keys()):
                    fig = plt.figure(figsize=(12, 12))
                    chi2par = [np.sum(chi2_i[key]) for chi2_i in chi2]
                    plt.xlabel(parameter)
                    plt.ylabel(parameter2)
                    plt.xlim(parameters.min(), parameters.max())
                    plt.ylim(parameters2.min(), parameters2.max())
                    extent = [parameters.min(), parameters.max(),
                              parameters2.min(), parameters2.max()]
                    levels = 9
                    X = np.array(parameters_1).reshape(N, N)
                    Y = np.array(parameters_2).reshape(N, N)
                    C = np.array(chi2par).reshape(N, N)
                    Image = plt.contourf(X, Y, C, levels, cmap=plt.get_cmap('Reds'),
                                         extent=extent)
                    _ = plt.contour(X, Y, C, levels, extent=extent,
                                    cmap=plt.get_cmap('Reds'))
                    CB = plt.colorbar(Image)
                    plt.title('{0:08d}, {3}: {1}, {2}'.format(expid, parameter,
                                                              parameter2, key))
                    fig.savefig(fits_directory + '{0:04d}_vary2d_{1}_'.format(
                        nCalls, key) +
                                parameter + '_and_' + parameter2 + '.png')
                    plt.close('all')


if __name__ == "__main__":
    options = parser.parse_args()

    args = vars(options)

    do_fit(args)
