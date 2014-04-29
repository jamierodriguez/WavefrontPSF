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
                'Compare my results with PSFEx.')
parser.add_argument("-e",
                    dest="expid",
                    type=int,
                    help="what image number will we fit now?")
parser.add_argument("-c",
                    dest="catalogs",
                    default='/nfs/slac/g/ki/ki18/cpd/psfex_catalogs/',
                    help='directory containing the catalogs')
parser.add_argument("-r",
                    dest="results",
                    default='/nfs/slac/g/ki/ki18/cpd/psfex_catalogs/' +
                            'SVA1_FINALCUT/fits/combined_fits/results.npy',
                    help='path to results file')
parser.add_argument("-t",
                    dest="tag",
                    default='SVA1_FINALCUT',
                    help='tag for run')
parser.add_argument("-o",
                    dest="output_directory",
                    default='/nfs/slac/g/ki/ki18/cpd/psfex_catalogs/' +
                            'SVA1_FINALCUT/fits/validation/',
                    help="where will the outputs go")
parser.add_argument("-s",
                    dest="stamp",
                    default=0,
                    type=int,
                    help="save stamp / vignet parameter?")

options = parser.parse_args()

args_dict = vars(options)

lowest_path = args_dict['catalogs']
tag = args_dict['tag']
output_directory = args_dict['output_directory']
results_path = args_dict['results']
expid = args_dict['expid']
stamp_save = args_dict['stamp']

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
FP = FocalPlane(psfex_list_catalogs, psfex_list_fits_extension,
                psfex_list_chip,
                conds='minimal',
                n_samples_box=10000)

psfex_list_catalogs_sel = [psfex_list_catalogs_i.replace(
                            'val', 'sel')
                            for psfex_list_catalogs_i in psfex_list_catalogs]

FP_sel = FocalPlane(psfex_list_catalogs_sel, psfex_list_fits_extension,
                    psfex_list_chip,
                    conds='minimal',
                    n_samples_box=10000)


# psfex fit
psfex_list_catalogs_2 = [psfex_list_catalogs_i.replace(
                            'fits', 'psf').replace(
                                'valpsfcat', 'psfcat_validation_subtracted')
                       for psfex_list_catalogs_i in psfex_list_catalogs]

FPP = FocalPlanePSFEx(psfex_list_catalogs_2,
                      np.array(psfex_list_chip).flatten(),
                      verbosity=['history', 'stamp'])

# psfex fit
psfex_list_catalogs_2_sel = [psfex_list_catalogs_i.replace(
                            'fits', 'psf').replace(
                                'valpsfcat', 'psfcat')
                       for psfex_list_catalogs_i in psfex_list_catalogs]

FPP_sel = FocalPlanePSFEx(psfex_list_catalogs_2_sel,
                      np.array(psfex_list_chip).flatten(),
                      verbosity=['history', 'stamp'])



dat = results[results['expid'] == expid]

FPF = FocalPlaneFit(verbosity=['history', 'stamp'])

# get FPF dict
FPFdict = {}
for key in dat.dtype.names:
    if ('args' in key) * ('covariance' not in key):
        FPFdict.update({key[5:]: dat[key][0]})

##############################################################################
# generate catalogs
##############################################################################

FP_unaveraged = FP.data_unaveraged
PSFEx_unaveraged = FPP.plane(FP.coords)
WavefrontPSF_analytic_unaveraged = FPF.analytic_plane(FPFdict, FP.coords)
WavefrontPSF_unaveraged = FPF.plane(FPFdict, FP.coords)

FP_unaveraged_sel = FP_sel.data_unaveraged
PSFEx_unaveraged_sel = FPP.plane(FP_sel.coords)
WavefrontPSF_analytic_unaveraged_sel = FPF.analytic_plane(FPFdict, FP_sel.coords)
WavefrontPSF_unaveraged_sel = FPF.plane(FPFdict, FP_sel.coords)

# filter out stamp parameters
# TODO: this is cludge
stamp_names = ['vignet', 'stamp', 'VIGNET', 'STAMP']
if stamp_save == 0:
    # these are all dicts so remake them
    FP_unaveraged_fil = {}
    for key in FP_unaveraged.keys():
        if key in stamp_names:
            continue
        else:
            FP_unaveraged_fil.update({key: FP_unaveraged[key]})
    FP_unaveraged = FP_unaveraged_fil

    PSFEx_unaveraged_fil = {}
    for key in PSFEx_unaveraged.keys():
        if key in stamp_names:
            continue
        else:
            PSFEx_unaveraged_fil.update({key: PSFEx_unaveraged[key]})
    PSFEx_unaveraged = PSFEx_unaveraged_fil

    WavefrontPSF_analytic_unaveraged_fil = {}
    for key in WavefrontPSF_analytic_unaveraged.keys():
        if key in stamp_names:
            continue
        else:
            WavefrontPSF_analytic_unaveraged_fil.update({key: WavefrontPSF_analytic_unaveraged[key]})
    WavefrontPSF_analytic_unaveraged = WavefrontPSF_analytic_unaveraged_fil

    WavefrontPSF_unaveraged_fil = {}
    for key in WavefrontPSF_unaveraged.keys():
        if key in stamp_names:
            continue
        else:
            WavefrontPSF_unaveraged_fil.update({key: WavefrontPSF_unaveraged[key]})
    WavefrontPSF_unaveraged = WavefrontPSF_unaveraged_fil


    FP_unaveraged_sel_fil = {}
    for key in FP_unaveraged_sel.keys():
        if key in stamp_names:
            continue
        else:
            FP_unaveraged_sel_fil.update({key: FP_unaveraged_sel[key]})
    FP_unaveraged_sel = FP_unaveraged_sel_fil

    PSFEx_unaveraged_sel_fil = {}
    for key in PSFEx_unaveraged_sel.keys():
        if key in stamp_names:
            continue
        else:
            PSFEx_unaveraged_sel_fil.update({key: PSFEx_unaveraged_sel[key]})
    PSFEx_unaveraged_sel = PSFEx_unaveraged_sel_fil

    WavefrontPSF_analytic_unaveraged_sel_fil = {}
    for key in WavefrontPSF_analytic_unaveraged_sel.keys():
        if key in stamp_names:
            continue
        else:
            WavefrontPSF_analytic_unaveraged_sel_fil.update({key: WavefrontPSF_analytic_unaveraged_sel[key]})
    WavefrontPSF_analytic_unaveraged_sel = WavefrontPSF_analytic_unaveraged_sel_fil

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

np.save(output_directory + 'validation_data',
        FP_unaveraged)
np.save(output_directory + 'validation_psfex',
        PSFEx_unaveraged)
np.save(output_directory + 'validation_fit',
        WavefrontPSF_unaveraged)
np.save(output_directory + 'validation_fit_analytic',
        WavefrontPSF_analytic_unaveraged)

np.save(output_directory + 'trained_data',
        FP_unaveraged_sel)
np.save(output_directory + 'trained_psfex',
        PSFEx_unaveraged_sel)
np.save(output_directory + 'trained_fit',
        WavefrontPSF_unaveraged_sel)
np.save(output_directory + 'trained_fit_analytic',
        WavefrontPSF_analytic_unaveraged_sel)


