#!/usr/bin/env python
"""
File: batch_images.py
Author: Chris Davis
Description: File that takes a set of DES images and object catalogs and makes
a catalog I would want (i.e. with the parameters I am interested in and an
associated image stamp.
"""

from __future__ import print_function, division
import matplotlib
# the agg is so I can submit for batch jobs.
matplotlib.use('Agg')
import numpy as np
import argparse
from matplotlib.pyplot import close
from os import path, makedirs

from focal_plane import FocalPlane
from routines_files import generate_hdu_lists
from routines_plot import data_focal_plot, data_hist_plot, data_contour_plot
from decamutil_cpd import decaminfo

##############################################################################
# argparse
##############################################################################

parser = argparse. \
    ArgumentParser(description=
                   'Fit image and dump results.')
parser.add_argument("-e",
                    dest="expid",
                    type=int,
                    help="what image number will we fit now?")
parser.add_argument("-t",
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
parser.add_argument("-o",
                    dest="output_directory",
                    default='/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',
                    help="where will the outputs go")
parser.add_argument("-s",
                    dest="size",
                    type=int,
                    default=32,
                    help="Stamp size")
parser.add_argument("-f",
                    dest="conds",
                    default='default',
                    help="String for filter conditions")
options = parser.parse_args()

args_dict = vars(options)

##############################################################################
# create catalogs
##############################################################################

path_base = args_dict['catalogs']
name = args_dict['name']
extension = args_dict['extension']

list_catalogs_base = \
    path_base + 'DECam_{0:08d}_'.format(args_dict['expid'])
list_catalogs = [list_catalogs_base + '{0:02d}_{1}.fits'.format(i, name)
                 for i in xrange(1, 63)]
list_catalogs.pop(60)
# ccd 2 went bad too.
if args_dict['expid'] > 258804:
    list_catalogs.pop(1)

list_chip = [[decaminfo().ccddict[i]] for i in xrange(1, 63)]
list_chip.pop(60)
# ccd 2 went bad too.
if args_dict['expid'] > 258804:
    list_chip.pop(1)

# ccd 2 went bad too.
if args_dict['expid'] > 258804:
    list_fits_extension = [[extension]] * (63-3)
else:
    list_fits_extension = [[extension]] * (63-2)

##############################################################################
# create images for the catalog
##############################################################################

FP = FocalPlane(list_catalogs=list_catalogs,
                list_fits_extension=list_fits_extension,
                list_chip=list_chip,
                boxdiv=2,
                max_samples_box=20000,
                conds=args_dict['conds'],
                nPixels=args_dict['size'],
                )

# make directory if it doesn't exist
if not path.exists(args_dict['output_directory']):
    makedirs(args_dict['output_directory'])

# save data_unaveraged
np.save(args_dict['output_directory'] +
        'DECam_{0:08d}_'.format(args_dict['expid']) +
        'cat_cpd_combined',
        FP.data_unaveraged)

figures, axes, scales = data_focal_plot(FP.data,
                                        average=FP.average)
for figure_key in figures:
    axes[figure_key].set_title('{0:08d}: {1}'.format(args_dict['expid'],
                                                     figure_key))
    figures[figure_key].savefig(args_dict['output_directory'] +
                                'DECam_{0:08d}_'.format(args_dict['expid']) +
                                'focal_{0}.png'.format(figure_key))
close('all')

edges = FP.decaminfo.getEdges(FP.boxdiv)
figures, axes, scales = data_hist_plot(FP.data_unaveraged, edges=edges)
for figure_key in figures:
    axes[figure_key].set_title('{0:08d}: {1}'.format(args_dict['expid'],
                                                     figure_key))
    figures[figure_key].savefig(args_dict['output_directory'] +
                                'DECam_{0:08d}_'.format(args_dict['expid']) +
                                'hist_{0}.png'.format(figure_key))
close('all')

figures, axes, scales = data_contour_plot(FP.data_unaveraged, edges=edges,
                                          scales=scales)
for figure_key in figures:
    axes[figure_key].set_title('{0:08d}: {1}'.format(args_dict['expid'],
                                                     figure_key))
    figures[figure_key].savefig(args_dict['output_directory'] +
                                'DECam_{0:08d}_'.format(args_dict['expid']) +
                                'contour_{0}.png'.format(figure_key))
close('all')

# TODO: other images (distribution of e0 vs median value of a chip, etc)

