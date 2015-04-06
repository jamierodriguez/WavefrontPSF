#!/usr/bin/env python
"""
File: validation.py
Author: Chris Davis
Description: For a given expid, create the validation data.

TODO: generate images for wavefrontpsf
"""

from __future__ import print_function, division
import numpy as np
import argparse
from focal_plane_psfex import FocalPlanePSFEx
from focal_plane import FocalPlane
from focal_plane_fit import FocalPlaneFit
from routines_files import generate_hdu_lists

##############################################################################
# argparse
##############################################################################

parser = argparse. \
ArgumentParser(description=
                validation.__doc__)
parser.add_argument("--expid",
                    dest="expid",
                    type=int,
                    required=True,
                    help="what image number will we fit now?")
parser.add_argument("--catalog",
                    dest="catalogs",
                    default='/nfs/slac/g/ki/ki18/cpd/psfex_catalogs/',
                    help='directory containing the catalogs')
parser.add_argument("--results",
                    dest="results",
                    default='/nfs/slac/g/ki/ki18/cpd/psfex_catalogs/' +
                            'SVA1_FINALCUT/fits/combined_fits/results.npy',
                    help='path to results file')
parser.add_argument("--tag",
                    dest="tag",
                    default='SVA1_FINALCUT',
                    help='tag for run')
parser.add_argument("--output",
                    dest="output_directory",
                    default='/nfs/slac/g/ki/ki18/cpd/psfex_catalogs/' +
                            'SVA1_FINALCUT/fits/validation/',
                    help="where will the outputs go")
parser.add_argument("--stamp",
                    dest="stamp",
                    action="store_true",
                    default=False,
                    help="save stamp / vignet parameter?")
parser.add_argument("--training",
                    dest="training",
                    default=False,
                    action="store_true",
                    help="Do training ones?")
parser.add_argument("--psfex",
                    dest="psfex",
                    default=False,
                    action="store_true",
                    help="Do PSFex ones?")
parser.add_argument("--data",
                    dest="data",
                    default=False,
                    action="store_true",
                    help="Do data ones?")
parser.add_argument("--fit",
                    dest="fit",
                    default=False,
                    action="store_true",
                    help="Do fit ones?")
parser.add_argument("--fit_analytic",
                    dest="fit_analytic",
                    default=False,
                    action="store_true",
                    help="Do fit_analytic ones?")

def validation(args):
    """
    Compare my results with PSFEx.
    TODO: Docs!
    """

    lowest_path = args['catalogs']
    tag = args['tag']
    output_directory = args['output_directory']
    results_path = args['results']
    expid = args['expid']
    stamp_save = args['stamp']
    if stamp_save == 0:
        verbosity_list = ['history']
    else:
        verbosity_list = ['history', 'stamp']


    ##############################################################################
    # load up data
    ##############################################################################
    path_base = lowest_path + tag + '/psfcat/{0:08d}/'.format(expid - expid%1000)
    results = np.load(results_path)

    # val cat
    psfex_list_catalogs, psfex_list_fits_extension, psfex_list_chip = \
            generate_hdu_lists(
                expid, path_base=path_base,
                name='valpsfcat',
                extension=2)
    if args['data']:
        FP = FocalPlane(psfex_list_catalogs, psfex_list_fits_extension,
                        psfex_list_chip,
                        conds='minimal',
                        n_samples_box=10000)

    psfex_list_catalogs_sel = [psfex_list_catalogs_i.replace(
                                'val', 'sel')
                                for psfex_list_catalogs_i in psfex_list_catalogs]

    if args['training']:
        if args['data']:
            FP_sel = FocalPlane(psfex_list_catalogs_sel, psfex_list_fits_extension,
                                psfex_list_chip,
                                conds='minimal',
                                n_samples_box=10000)


    # psfex fit
    psfex_list_catalogs_2 = [psfex_list_catalogs_i.replace(
                                'fits', 'psf').replace(
                                    'valpsfcat', 'psfcat_validation_subtracted')
                           for psfex_list_catalogs_i in psfex_list_catalogs]

    if args['psefex']:
        FPP = FocalPlanePSFEx(psfex_list_catalogs_2,
                              np.array(psfex_list_chip).flatten(),
                              verbosity=verbosity_list)

    # psfex fit
    psfex_list_catalogs_2_sel = [psfex_list_catalogs_i.replace(
                                'fits', 'psf').replace(
                                    'valpsfcat', 'psfcat')
                           for psfex_list_catalogs_i in psfex_list_catalogs]

    if args['training']:
        if args['psfex']:
            FPP_sel = FocalPlanePSFEx(psfex_list_catalogs_2_sel,
                                  np.array(psfex_list_chip).flatten(),
                                  verbosity=verbosity_list)



    dat = results[results['expid'] == expid]

    FPF = FocalPlaneFit(verbosity=verbosity_list)

    # get FPF dict
    FPFdict = {}
    for key in dat.dtype.names:
        if ('args' in key) * ('covariance' not in key):
            FPFdict.update({key[5:]: dat[key][0]})

    ##############################################################################
    # generate catalogs
    ##############################################################################

    if args['data']:
        FP_unaveraged = FP.data_unaveraged
    if args['psfex']:
        PSFEx_unaveraged = FPP.plane(FP.coords)
    if args['fit_analytic']
        WavefrontPSF_analytic_unaveraged = FPF.analytic_plane(FPFdict, FP.coords)
    if args['fit']:
        WavefrontPSF_unaveraged = FPF.plane(FPFdict, FP.coords)

    if args['training']:
        if args['data']:
            FP_unaveraged_sel = FP_sel.data_unaveraged
        if args['psfex']:
            PSFEx_unaveraged_sel = FPP.plane(FP_sel.coords)
        if args['fit_analytic']:
            WavefrontPSF_analytic_unaveraged_sel = FPF.analytic_plane(FPFdict, FP_sel.coords)
        if args['fit']:
            WavefrontPSF_unaveraged_sel = FPF.plane(FPFdict, FP_sel.coords)

    # filter out stamp parameters
    # TODO: this is cludge
    stamp_names = ['vignet', 'stamp', 'VIGNET', 'STAMP']
    if stamp_save == 0:
        # these are all dicts so remake them
        if args['data']:
            FP_unaveraged_fil = {}
            for key in FP_unaveraged.keys():
                if key in stamp_names:
                    continue
                else:
                    FP_unaveraged_fil.update({key: FP_unaveraged[key]})
            FP_unaveraged = FP_unaveraged_fil
        if args['psfex']:
            PSFEx_unaveraged_fil = {}
            for key in PSFEx_unaveraged.keys():
                if key in stamp_names:
                    continue
                else:
                    PSFEx_unaveraged_fil.update({key: PSFEx_unaveraged[key]})
            PSFEx_unaveraged = PSFEx_unaveraged_fil
        if args['fit_analytic']:
            WavefrontPSF_analytic_unaveraged_fil = {}
            for key in WavefrontPSF_analytic_unaveraged.keys():
                if key in stamp_names:
                    continue
                else:
                    WavefrontPSF_analytic_unaveraged_fil.update({key: WavefrontPSF_analytic_unaveraged[key]})
            WavefrontPSF_analytic_unaveraged = WavefrontPSF_analytic_unaveraged_fil
        if args['fit']:
            WavefrontPSF_unaveraged_fil = {}
            for key in WavefrontPSF_unaveraged.keys():
                if key in stamp_names:
                    continue
                else:
                    WavefrontPSF_unaveraged_fil.update({key: WavefrontPSF_unaveraged[key]})
            WavefrontPSF_unaveraged = WavefrontPSF_unaveraged_fil

        if args['training']:
            if args['data']:
                FP_unaveraged_sel_fil = {}
                for key in FP_unaveraged_sel.keys():
                    if key in stamp_names:
                        continue
                    else:
                        FP_unaveraged_sel_fil.update({key: FP_unaveraged_sel[key]})
                FP_unaveraged_sel = FP_unaveraged_sel_fil
            if args['psfex']:
                PSFEx_unaveraged_sel_fil = {}
                for key in PSFEx_unaveraged_sel.keys():
                    if key in stamp_names:
                        continue
                    else:
                        PSFEx_unaveraged_sel_fil.update({key: PSFEx_unaveraged_sel[key]})
                PSFEx_unaveraged_sel = PSFEx_unaveraged_sel_fil
            if args['fit_analytic']:
                WavefrontPSF_analytic_unaveraged_sel_fil = {}
                for key in WavefrontPSF_analytic_unaveraged_sel.keys():
                    if key in stamp_names:
                        continue
                    else:
                        WavefrontPSF_analytic_unaveraged_sel_fil.update({key: WavefrontPSF_analytic_unaveraged_sel[key]})
                WavefrontPSF_analytic_unaveraged_sel = WavefrontPSF_analytic_unaveraged_sel_fil
            if args['fit']:
                WavefrontPSF_unaveraged_sel_fil = {}
                for key in WavefrontPSF_unaveraged_sel.keys():
                    if key in stamp_names:
                        continue
                    else:
                        WavefrontPSF_unaveraged_sel_fil.update({key: WavefrontPSF_unaveraged_sel[key]})
                WavefrontPSF_unaveraged_sel = WavefrontPSF_unaveraged_sel_fil





    ##############################################################################
    # save catalogs
    ##############################################################################

    if args['data']:
        np.save(output_directory + 'validation_data',
                FP_unaveraged)
    if args['psfex']:
        np.save(output_directory + 'validation_psfex',
                PSFEx_unaveraged)
    if args['fit']:
        np.save(output_directory + 'validation_fit',
                WavefrontPSF_unaveraged)
    if args['fit_analytic']:
        np.save(output_directory + 'validation_fit_analytic',
                WavefrontPSF_analytic_unaveraged)

    if args['training']:
        if args['data']:
            np.save(output_directory + 'trained_data',
                    FP_unaveraged_sel)
        if args['psfex']:
            np.save(output_directory + 'trained_psfex',
                    PSFEx_unaveraged_sel)
        if args['fit']:
            np.save(output_directory + 'trained_fit',
                    WavefrontPSF_unaveraged_sel)
        if args['fit_analytic']:
            np.save(output_directory + 'trained_fit_analytic',
                    WavefrontPSF_analytic_unaveraged_sel)


if __name__ == '__main__':
    # parse args
    options = parser.parse_args()

    args = vars(options)

    validation(args)
