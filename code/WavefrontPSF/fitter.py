#!/usr/bin/env python
"""
File: fitter.py
Author: Chris Davis
Description: Fit data to either analytic or not analytic mesh
"""

from iminuit import Minuit
from WavefrontPSF.psf_interpolator import Mesh_Interpolator
from WavefrontPSF.wavefront import Wavefront
from WavefrontPSF.digestor import Digestor
from WavefrontPSF.analytic_interpolator import DECAM_Analytic_Wavefront, r0_guess, Zernike_Evaluator
from WavefrontPSF.donutengine import DECAM_Model_Wavefront
from WavefrontPSF.defaults import param_default_kils

import numpy as np


def do_fit(WF, misalignment, weights={}, do_limits=True, num_bins=-1,
           verbose=True):

    weights_default = {'e0': 0.5, 'e1': 1, 'e2': 1,
               'delta1': 0, 'delta2': 0,
               'zeta1': 0, 'zeta2': 0}
    weights_default.update(weights)
    weights = weights_default

    # figure out which misalignment parameters are fixed
    arguments = {'z04d': 0, 'z04x': 0, 'z04y': 0,
                 'z05d': 0, 'z05x': 0, 'z05y': 0,
                 'z06d': 0, 'z06x': 0, 'z06y': 0,
                 'z07d': 0, 'z07x': 0, 'z07y': 0,
                 'z08d': 0, 'z08x': 0, 'z08y': 0,
                 'z09d': 0, 'z09x': 0, 'z09y': 0,
                 'z10d': 0, 'z10x': 0, 'z10y': 0,
                 'z11d': 0, 'z11x': 0, 'z11y': 0,
                 'dz': 0,
                 'dx': 0, 'dy': 0,
                 'xt': 0, 'yt': 0,
                 'rzero': 0.15, 'e0': 0,
                 'e1': 0, 'e2': 0,
                 'delta1': 0, 'delta2': 0,
                 'zeta1': 0, 'zeta2': 0}
    arguments.update(misalignment)
    minuit_kwargs = {'errordef': 1}
    for arg in arguments.keys():
        limit_key = 'limit_{0}'.format(arg)
        error_key = 'error_{0}'.format(arg)
        minuit_kwargs[arg] = arguments[arg]
        # do limits
        if arg[0] == 'z' and arg[-1] == 'd':
            if do_limits:
                minuit_kwargs[limit_key] = (-2, 2)
            minuit_kwargs[error_key] = 1e-2
        elif arg[0] == 'z' and arg[-1] in ['x', 'y']:
            if do_limits:
                minuit_kwargs[limit_key] = (-2, 2)
            minuit_kwargs[error_key] = 1e-4
        elif arg == 'rzero':
            # always do rzero limits
            minuit_kwargs[limit_key] = (0.08, 0.25)
            minuit_kwargs[error_key] = 1e-2
        elif arg in ['e0', 'e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']:
            if do_limits:
                minuit_kwargs[limit_key] = (-0.5, 0.5)
            if 'delta' in arg:
                minuit_kwargs[error_key] = 1e-4
            elif 'zeta' in arg:
                minuit_kwargs[error_key] = 1e-4
            else:
                minuit_kwargs[error_key] = 1e-2
        elif arg == 'dz':
            if do_limits:
                minuit_kwargs[limit_key] = (-500, 500)
            minuit_kwargs[error_key] = 20
        elif arg in ['dx', 'dy']:
            if do_limits:
                minuit_kwargs[limit_key] = (-4500, 4500)
            minuit_kwargs[error_key] = 100
        elif arg in ['xt', 'yt']:
            if do_limits:
                minuit_kwargs[limit_key] = (-1000, 1000)
            minuit_kwargs[error_key] = 50

        # do fixing
        if arg not in misalignment.keys():
            minuit_kwargs['fix_{0}'.format(arg)] = True
        else:
            minuit_kwargs['fix_{0}'.format(arg)] = False

    if num_bins > -1:
        # create the WF field
        WF.reduce(num_bins=num_bins)

    def plane(rzero,
             z04d, z04x, z04y,
             z05d, z05x, z05y,
             z06d, z06x, z06y,
             z07d, z07x, z07y,
             z08d, z08x, z08y,
             z09d, z09x, z09y,
             z10d, z10x, z10y,
             z11d, z11x, z11y,
             dz, dx, dy, xt, yt,
             e0, e1, e2,
             delta1, delta2,
             zeta1, zeta2):
        wf_misalignment = {'z04d': z04d, 'z04x': z04x, 'z04y': z04y,
                        'z05d': z05d, 'z05x': z05x, 'z05y': z05y,
                        'z06d': z06d, 'z06x': z06x, 'z06y': z06y,
                        'z07d': z07d, 'z07x': z07x, 'z07y': z07y,
                        'z08d': z08d, 'z08x': z08x, 'z08y': z08y,
                        'z09d': z09d, 'z09x': z09x, 'z09y': z09y,
                        'z10d': z10d, 'z10x': z10x, 'z10y': z10y,
                        'z11d': z11d, 'z11x': z11x, 'z11y': z11y,
                        'dz': dz, 'dx': dx, 'dy': dy, 'xt': xt, 'yt': yt}
        WF.data['rzero'] = rzero
        # get evaluated PSFs
        test = WF(WF.data, misalignment=wf_misalignment)

        # add dc factors
        test['e0'] += e0
        test['e1'] += e1
        test['e2'] += e2
        test['delta1'] += delta1
        test['delta2'] += delta2
        test['zeta1'] += zeta1
        test['zeta2'] += zeta2

        return test, wf_misalignment

    # attach to class to let this float through namespaces...
    WF.num_execute = 0
    def chi2(rzero,
             z04d, z04x, z04y,
             z05d, z05x, z05y,
             z06d, z06x, z06y,
             z07d, z07x, z07y,
             z08d, z08x, z08y,
             z09d, z09x, z09y,
             z10d, z10x, z10y,
             z11d, z11x, z11y,
             dz, dx, dy, xt, yt,
             e0, e1, e2,
             delta1, delta2,
             zeta1, zeta2):

        test, wf_misalignment = plane(rzero,
             z04d, z04x, z04y,
             z05d, z05x, z05y,
             z06d, z06x, z06y,
             z07d, z07x, z07y,
             z08d, z08x, z08y,
             z09d, z09x, z09y,
             z10d, z10x, z10y,
             z11d, z11x, z11y,
             dz, dx, dy, xt, yt,
             e0, e1, e2,
             delta1, delta2,
             zeta1, zeta2)

        chi2 = 0
        weight_sum = 0
        if num_bins > -1:
            Y = WF
            X, _, _ = WF.reduce_data_to_field(test, num_bins=num_bins)
        else:
            X = test
            Y = WF.data

        if verbose:
            if WF.num_execute%50 == 0:
                print_misalignment = {}
                print_misalignment.update(wf_misalignment)
                print_misalignment['rzero'] = rzero
                print_misalignment['e0'] = e0
                print_misalignment['e1'] = e1
                print_misalignment['e2'] = e2
                print_misalignment['delta1'] = delta1
                print_misalignment['delta2'] = delta2
                print_misalignment['zeta1'] = zeta1
                print_misalignment['zeta2'] = zeta2
                # filter out misalignment entries
                bad_keys = [key for key in print_misalignment if key not in misalignment]
                for key in bad_keys:
                    _ = print_misalignment.pop(key)
                print(WF.num_execute, print_misalignment)
        for key in weights:
            chi2_w = np.mean(np.square(X[key] - Y[key]))
            if verbose:
                if WF.num_execute%50 == 0:
                    print(key, chi2_w, weights[key])
            chi2 += chi2_w * weights[key]
            weight_sum += weights[key]
        chi2 /= weight_sum
        if verbose:
            if WF.num_execute%50 == 0:
                print(chi2)
                print('================================================\n\n')

        WF.num_execute += 1
        return chi2

    if verbose:
        print_level = 2
    else:
        print_level = 0

    minuit = Minuit(chi2, print_level=print_level, **minuit_kwargs)
    minuit.migrad()

    return minuit, chi2, plane

def plot_results(WF, plane, minuit, num_bins=2):
    keys = ['e0', 'e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']

    fig, axs = plt.subplots(nrows=len(keys), ncols=3,
                            figsize=(4 * 3, 3 * len(keys)))

    comp, misalignment = plane(**minuit.values)
    for key_i, key in enumerate(keys):
        comp['true_{0}'.format(key)] = WF.data[key]
        comp['diff_{0}'.format(key)] = comp[key] - WF.data[key]

        ax = axs[key_i, 0]
        zkey = 'true_{0}'.format(key)
        fig, ax = WF.plot_colormap(comp, zkey=zkey,
            fig=fig, ax=ax, num_bins=num_bins)
        ax.set_title(zkey)
        ax = axs[key_i, 1]
        zkey = '{0}'.format(key)
        fig, ax = WF.plot_colormap(comp, zkey=zkey,
            fig=fig, ax=ax, num_bins=num_bins)
        ax.set_title(zkey)
        ax = axs[key_i, 2]
        zkey = 'diff_{0}'.format(key)
        fig, ax = WF.plot_colormap(comp, zkey=zkey,
            fig=fig, ax=ax, num_bins=num_bins)
        ax.set_title(zkey)
    return comp, fig, ax


def drive_fit(expid, params={}, skip_fit=False):

    params_default = {'analytic': True, 'number_sample': 0, 'verbose': True}
    params_default.update(param_default_kils(expid))
    params_default.update(params)
    params = params_default

    model = Digestor().digest_directory(
            params['data_directory'],
            file_type=params['data_name'])
    if params['number_sample'] > 0:
        rows = np.random.choice(model.index.values,
                params['number_sample'], replace=False)
        model = model.ix[rows]

    # guess rzero
    rzero = r0_guess(model['e0'].min())
    rzeros_float = np.array([0.08 + 0.01 * i for i in xrange(15)])
    rzeros = ['{0:.2f}'.format(0.08 + 0.01 * i) for i in xrange(15)]
    # always want the rzero one above so that we underestimate it
    # note that this also implies that we want a dc component for e0 in
    # addition to e1 and e2
    rzero_i = np.searchsorted(rzeros_float, rzero)
    rzero_key = rzeros[rzero_i]
    model['rzero'] = rzero

    # get the PSF_Interpolator
    PSF_Interpolator = Mesh_Interpolator(mesh_name=params['mesh_name'],
            directory=params['mesh_directory'])

    if params['analytic']:
        PSF_Evaluator = Zernike_Evaluator(*np.load(params['analytic_coeffs']).item()[rzero_key])
        WF = DECAM_Analytic_Wavefront(rzero=rzero,
                PSF_Interpolator=PSF_Interpolator,
                PSF_Evaluator=PSF_Evaluator,
                num_bins=1, model=model)
    else:
        WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator,
                                   num_bins=1,
                                   model=model)

    misalignment = translate_misalignment_to_arguments({})
    # fix some keys
    pop_keys = ['rzero', 'delta1', 'delta2', 'zeta1', 'zeta2']
    for pop_key in pop_keys:
        _ = misalignment.pop(pop_key)

    if skip_fit:
        return WF

    minuit, chi2, plane = do_fit(WF, misalignment=misalignment, verbose=params['verbose'])

    return minuit, chi2, plane, WF

def translate_misalignment_to_arguments(misalignment={}):
    # can also call this to get all params
    arguments = {'z04d': 0, 'z04x': 0, 'z04y': 0,
                 'z05d': 0, 'z05x': 0, 'z05y': 0,
                 'z06d': 0, 'z06x': 0, 'z06y': 0,
                 'z07d': 0, 'z07x': 0, 'z07y': 0,
                 'z08d': 0, 'z08x': 0, 'z08y': 0,
                 'z09d': 0, 'z09x': 0, 'z09y': 0,
                 'z10d': 0, 'z10x': 0, 'z10y': 0,
                 'z11d': 0, 'z11x': 0, 'z11y': 0,
                 'dz': 0, 'dx': 0, 'dy': 0, 'xt': 0, 'yt': 0,
                 'rzero': 0.15, 'e0': 0,
                 'e1': 0, 'e2': 0,
                 'delta1': 0, 'delta2': 0,
                 'zeta1': 0, 'zeta2': 0}
    arguments.update(misalignment)

    return arguments


# data wavefront will be either analytic or decam_model_wavefront with a model
# paraemter passed in that is the data of the assessed fits files

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from WavefrontPSF.psf_interpolator import Mesh_Interpolator
    from WavefrontPSF.wavefront import Wavefront
    from WavefrontPSF.digestor import Digestor
    from WavefrontPSF.analytic_interpolator import DECAM_Analytic_Wavefront, r0_guess
    from WavefrontPSF.donutengine import DECAM_Model_Wavefront

    model = Digestor().digest_directory(
            '/Users/cpd/Projects/WavefrontPSF/meshes/00253794/',
            file_type='_selpsfcat.fits')
    # guess rzero
    model['rzero'] = r0_guess(model['e0'].min())

    mesh_directory = '/Users/cpd/Projects/WavefrontPSF/meshes/Science-20140212s2-v1i2'
    mesh_name = 'Science-20140212s2-v1i2_All'
    PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=mesh_directory)


    #from astropy.stats import mad_std

    WF = DECAM_Model_Wavefront(PSF_Interpolator=PSF_Interpolator,
                               num_bins=1,
                               model=model)#, reducer=mad_std)
    # WF.plot_field('e0')
    # naieve = WF(WF.data)
    # WF.plot_colormap(naieve, 'x', 'y', 'e0', num_bins=1)

    # create a dummy dataset from 100 random points
    WF.data = model
    rows = np.random.choice(WF.data.index.values, 500, replace=False)
    test = WF.data.ix[rows].copy()
    # reset index
    test.index = np.arange(len(test))
    misalignment_true = {'z09d':0.5, 'z04d':-0.1, 'z06x':0.002,
                         'rzero': 0.13, 'z07y': -0.003}
    test = WF(test, misalignment=misalignment_true)
    WF.data = test

    misalignment = {mi: 0 for mi in misalignment_true}
    minuit, chi2, plane = do_fit(WF, misalignment, weights={}, do_limits=True, num_bins=-1, verbose=True)


