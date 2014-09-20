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
import numpy as np
import argparse

from focal_plane import FocalPlane
from focal_plane_fit import FocalPlaneFit
from minuit_fit import Minuit_Fit
from routines import minuit_dictionary
from routines_moments import convert_moments
from routines_files import make_directory
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
parser.add_argument("--mesh_name",
                    dest="mesh_name",
                    default="Science-20130325s1-v1i2_All",
                    help="Name of mesh used for generating zernikes.")
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


###############################################################################
# git info
###############################################################################

from subprocess import call
def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

git_hash = get_git_revision_hash()

def do_fit(args):
    verbose = args['verbose']

    average = np.median  # TODO: this may change!!!
    boxdiv = args['boxdiv']
    conds = args['conds']
    n_samples_box = args['n_samples_box']
    if boxdiv >= 0:
        max_samples_box = 100000
    else:
        max_samples_box = n_samples_box
    subav = False
    methodVal = (args['methodVal'], 1.)
    mesh_name = args['mesh_name']

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

    chi_weights = args['chi_weights']
    p_init = args['p_init']

    FP = FocalPlane(list_catalogs=list_catalogs,
                    list_fits_extension=list_fits_extension,
                    list_chip=list_chip,
                    boxdiv=boxdiv,
                    max_samples_box=max_samples_box,
                    conds=conds,
                    average=average,
                    subav=subav,
                    )

    if boxdiv >= 0:
        data_compare = FP.data_averaged
        var_dict = data_compare
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
        # save memory?!
        del data_sample, recdata_sample, FP.recdata, FP.data_unaveraged
    else:
        data_compare = FP.data
        del FP.data_averaged, FP.recdata
        # create var_dict from snr_win parameter
        var_dict = {}
        weights = 1. / data_compare['snr_win']
        for key in chi_weights:
            var_dict.update({'var_' + key: weights})

    # create FocalPlaneFit object
    FPF = FocalPlaneFit(methodVal=methodVal,
                        mesh_name=mesh_name)

    # give FPF some attributes
    FPF.chi_weights = chi_weights
    if boxdiv >= 0:
        FPF.coords = coords_sample
    else:
        FPF.coords = FP.coords

    # for record sake:
    print('Number of objects used: ', len(FPF.coords))

    # define the fit function
    chi2hist = []
    FPF.history = []

    # get base zernikes for coords
    zernikes = FPF.interpolate_zernikes(FPF.coords)

    if args['analytic'] > 0:
        if boxdiv >= 0:
            def plane_func(in_dict):
                return FPF.analytic_plane_averaged(in_dict,
                        coords=FPF.coords,
                        average=average, boxdiv=boxdiv,
                        subav=subav,
                        zernikes=zernikes)
        else:
            def plane_func(in_dict):
                return FPF.analytic_plane(in_dict, FPF.coords, zernikes=zernikes)
    else:
        if boxdiv >= 0:
            def plane_func(in_dict):
                return FPF.plane_averaged(in_dict,
                        coords=FPF.coords,
                        average=average, boxdiv=boxdiv,
                        subav=subav,
                        zernikes=zernikes)
        else:
            def plane_func(in_dict):
                return FPF.plane(in_dict, FPF.coords, zernikes=zernikes)

    def FitFunc(in_dict):


        # go through the key_FP_funcs and make sure there are no nans
        for key_FP_func in in_dict.keys():
            if np.isnan(in_dict[key_FP_func]).any():
                # if there is a nan, don't even bother calling, just return a
                # big chi2
                FPF.remakedonut()
                return 1e20

        poles_i = plane_func(in_dict)

        FPF.temp_plane = poles_i

        chi2 = FPF.compare(poles_i, data_compare, var_dict=var_dict,
                           chi_weights=FPF.chi_weights)

        chi2hist.append(chi2)

        return chi2

    def SaveFunc(number):

        return

    ## import time
    ## t0 = time.time()
    ## for i in range(5):
    ##     FitFunc(p_init)
    ## t1 = time.time()
    ## print((t1 - t0) / 5.)
    ## print(data_compare['x'].size)

    par_names = args['par_names']
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
    minuit_results.update({'args_dict': args})
    minuit_results.update({'git_hash': git_hash})

    np.save(fits_directory + 'minuit_results', minuit_results)

    if boxdiv >= 0:
        # also save the coordinates
        np.save(fits_directory + 'coords_sample', coords_sample)

    # save a plane!
    in_dict = minuit_results['args']
    poles_i = plane_func(in_dict)

    # save the zernikes too
    np.save(fits_directory + 'zernikes', zernikes)

    np.save(fits_directory + 'plane_fit', poles_i)
    # pop vignet it's fucking huge
    _ = data_compare.pop('vignet')
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



if __name__ == "__main__":
    options = parser.parse_args()

    args = vars(options)

    args['chi_weights'] = eval(args['chi_weights'])
    args['p_init'] = eval(args['p_init'])
    args['p_names'] = eval(args['p_names'])
    do_fit(args)

