#!/usr/bin/env python
"""
File: catalog_maker.py
Author: Chris Davis
Description: File that takes a set of DES images and object catalogs and makes
a catalog I would want (i.e. with the parameters I am interested in and an
associated image stamp.
"""

from __future__ import print_function, division
import numpy as np
import pyfits
import argparse
from subprocess import call
from os import path, makedirs, chdir, system, remove

from focal_plane import FocalPlane
from decamutil_cpd import decaminfo
"""
TODO:
    [ ] Consider looking at DESDB for the 'smart' way of downloading the files.

input_list = [
    [20130906105326, 20130905, 231046, 231053],
    [20130906105326, 20130905, 231089, 231096],
    [20130911103044, 20130910, 232608, 232849],
    [20130913151017, 20130912, 233377, 233571],
    [20130913151017, 20130912, 233584, 233642],
    ]

rid, date, minImage, maxImage = input_list[0]

"""
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
parser.add_argument("-m",
                    dest="path_mesh",
                    default='/u/ec/roodman/Astrophysics/Donuts/Meshes/',
                    help="where is the meshes are located")
parser.add_argument("-n",
                    dest="mesh_name",
                    default="Science20120915s1v3_134239",
                    help="Name of mesh used.")
parser.add_argument("-o",
                    dest="output_directory",
                    default='/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',
                    help="where will the outputs go")
parser.add_argument("-t",
                    dest="catalogs",
                    default='/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',
                    help='directory containing the catalogs')
parser.add_argument("-rid",
                    dest="rid",
                    help="Run ID")
parser.add_argument("-d",
                    dest="date",
                    help="Date")
parser.add_argument("-s",
                    dest="size",
                    type=int,
                    default=16,
                    help="Stamp size")
parser.add_argument("-f",
                    dest="conds",
                    default='default',
                    help="String for filter conditions")
options = parser.parse_args()

args_dict = vars(options)

nPixels = args_dict['size']

##############################################################################
# download the image and catalog
##############################################################################

# from wgetscript:
dataDirectory = "$CPD/catalogs/wgetscript/{0:08d}".format(args_dict['expid'])
dataDirectoryExp = path.expandvars(dataDirectory)

# make directory if it doesn't exist
if not path.exists(dataDirectoryExp):
    makedirs(dataDirectoryExp)

# move there!
chdir(dataDirectoryExp)

##############################################################################
# create the new catalogs
##############################################################################
for i in xrange(1, 63):

    if i == 61:
        # screw N30
        continue

    # get cat
    command = "wget --no-check-certificate --http-user=cpd --http-password=cpd70chips -nc -nd -nH -r -k -p -np  --cut-dirs=3 https://desar2.cosmology.illinois.edu/DESFiles/desardata/OPS/red/{0}_{3}/red/DECam_{1:08d}/DECam_{1:08d}_{2:02d}_cat.fits".format(args_dict['rid'], args_dict['expid'], i, args_dict['date'])
    system(command)
    # get image
    command = "wget --no-check-certificate --http-user=cpd --http-password=cpd70chips -nc -nd -nH -r -k -p -np  --cut-dirs=3 https://desar2.cosmology.illinois.edu/DESFiles/desardata/OPS/red/{0}_{3}/red/DECam_{1:08d}/DECam_{1:08d}_{2:02d}.fits.fz".format(args_dict['rid'], args_dict['expid'], i, args_dict['date'])
    system(command)
    # decompress image
    command = "funpack DECam_{0:08d}_{1:02d}.fits.fz".format(args_dict['expid'], i)
    system(command)
    # remove old compressed image
    remove("DECam_{0:08d}_{1:02d}.fits.fz".format(args_dict['expid'], i))

    # create an FP object
    path_base = args_dict['catalogs']
    list_catalogs_base = \
        path_base + '{0:08d}/DECam_{0:08d}_'.format(args_dict['expid'])
    list_catalogs = [list_catalogs_base + '{0:02d}_cat.fits'.format(i)]
    list_chip = [[decaminfo().ccddict[i]]]
    list_fits_extension = [[2]]

    FP = FocalPlane(list_catalogs=list_catalogs,
                    list_fits_extension=list_fits_extension,
                    list_chip=list_chip,
                    path_mesh=args_dict['path_mesh'],
                    mesh_name=args_dict['mesh_name'],
                    boxdiv=0,
                    max_samples_box=20000,
                    conds=args_dict['conds'],
                    )

    # now use the catalog here to make a new catalog
    image = pyfits.getdata(list_catalogs_base + '{0:02d}.fits'.format(i), ext=0)

    # generate dictionary of columns
    pyfits_dict = {}
    pyfits_dict.update({
        'XWIN_IMAGE': dict(name='XWIN_IMAGE',
                           array=FP.recdata['XWIN_IMAGE'],
                           format='1D',
                           unit='pixel',),
        'YWIN_IMAGE': dict(name='YWIN_IMAGE',
                           array=FP.recdata['YWIN_IMAGE'],
                           format='1D',
                           unit='pixel',),
        'FLAGS':      dict(name='FLAGS',
                           array=FP.recdata['FLAGS'],
                           format='1I',),
        'BACKGROUND':      dict(name='BACKGROUND',
                           array=FP.recdata['BACKGROUND'],
                           format='1E',
                           unit='count',),
        'THRESHOLD':      dict(name='THRESHOLD',
                           array=FP.recdata['THRESHOLD'],
                           format='1E',
                           unit='count',),
        'MAG_AUTO':   dict(name='MAG_AUTO',
                           array=FP.recdata['MAG_AUTO'],
                           format='1E',
                           unit='mag',),
        'MAG_PSF':   dict(name='MAG_PSF',
                           array=FP.recdata['MAG_PSF'],
                           format='1E',
                           unit='mag',),
        'FLUX_RADIUS':   dict(name='FLUX_RADIUS',
                           array=FP.recdata['FLUX_RADIUS'],
                           format='1E',
                           unit='pixel',),
        'CLASS_STAR': dict(name='CLASS_STAR',
                           array=FP.recdata['CLASS_STAR'],
                           format='1E',),
        'SPREAD_MODEL':dict(name='SPREAD_MODEL',
                            array=FP.recdata['SPREAD_MODEL'],
                            format='1E',),
        'SPREADERR_MODEL':dict(name='SPREADERR_MODEL',
                               array=FP.recdata['SPREADERR_MODEL'],
                               format='1E',),
        'FWHMPSF_IMAGE': dict(name='FWHMPSF_IMAGE',
                              array=FP.recdata['FWHMPSF_IMAGE'],
                              format='1D',
                              unit='pixel',),

        # include the sextractor ones too
        'X2WIN_IMAGE_SEX': dict(name='X2WIN_IMAGE_SEX',
                            array=FP.recdata['X2WIN_IMAGE'],
                           format='1D',
                           unit='pixel**2',),
        'XYWIN_IMAGE_SEX': dict(name='XYWIN_IMAGE_SEX',
                            array=FP.recdata['XYWIN_IMAGE'],
                           format='1D',
                           unit='pixel**2',),
        'Y2WIN_IMAGE_SEX': dict(name='Y2WIN_IMAGE_SEX',
                            array=FP.recdata['Y2WIN_IMAGE'],
                           format='1D',
                           unit='pixel**2',),
        'FWHM_WORLD_SEX': dict(name='FWHM_WORLD_SEX',
                            array=FP.recdata['FWHM_WORLD'],
                           format='1D',
                           unit='deg',),

        'X2WIN_IMAGE': dict(name='X2WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**2',),
        'XYWIN_IMAGE': dict(name='XYWIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**2',),
        'Y2WIN_IMAGE': dict(name='Y2WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**2',),
        'X3WIN_IMAGE': dict(name='X3WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'X2YWIN_IMAGE': dict(name='X2YWIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'XY2WIN_IMAGE': dict(name='XY2WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'Y3WIN_IMAGE': dict(name='Y3WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'FWHM_WORLD': dict(name='FWHM_WORLD',
                           array=[],
                           format='1D',
                           unit='deg',),
        'STAMP':      dict(name='STAMP',
                           array=[],
                           format='{0}D'.format(nPixels ** 2),
                           unit='count')
        })

    Y, X = np.indices((16, 16)) + 0.5

    # go through each entry and make stamps and other parameters
    for recdata in FP.recdata:

        x_center = recdata[FP.x_coord_name]
        y_center = recdata[FP.y_coord_name]

        # coords are y,x ordered
        y_start = int(y_center - nPixels / 2)
        y_end = y_start + nPixels
        x_start = int(x_center - nPixels / 2)
        x_end = x_start + nPixels

        stamp = image[y_start:y_end, x_start:x_end].astype(np.float64)

        order_dict = {'x2': {'p': 2, 'q': 0},
                      'y2': {'p': 0, 'q': 2},
                      'xy': {'p': 1, 'q': 1},
                      'x3': {'p': 3, 'q': 0},
                      'y3': {'p': 0, 'q': 3},
                      'x2y': {'p': 2, 'q': 1},
                      'xy2': {'p': 1, 'q': 2}}

        background = recdata['BACKGROUND']
        threshold = recdata['THRESHOLD']
        moment_dict = FP.moments(stamp, indices=[Y, X],
                                 background=background,
                                 threshold=threshold,
                                 order_dict=order_dict)

        # append to pyfits_dict
        pyfits_dict['STAMP']['array'].append(
            stamp.flatten())
        pyfits_dict['X2WIN_IMAGE']['array'].append(
            moment_dict['x2'])
        pyfits_dict['Y2WIN_IMAGE']['array'].append(
            moment_dict['y2'])
        pyfits_dict['XYWIN_IMAGE']['array'].append(
            moment_dict['xy'])
        pyfits_dict['X3WIN_IMAGE']['array'].append(
            moment_dict['x3'])
        pyfits_dict['Y3WIN_IMAGE']['array'].append(
            moment_dict['y3'])
        pyfits_dict['X2YWIN_IMAGE']['array'].append(
            moment_dict['x2y'])
        pyfits_dict['XY2WIN_IMAGE']['array'].append(
            moment_dict['xy2'])
        pyfits_dict['FWHM_WORLD']['array'].append(
            moment_dict['fwhm'] * 0.27 / 3600)

    # create the columns
    columns = []
    for key in pyfits_dict:
        columns.append(pyfits.Column(**pyfits_dict[key]))
    tbhdu = pyfits.new_table(columns)

    tbhdu.writeto(args_dict['output_directory'] + \
                  'DECam_{0:08d}_'.format(args_dict['expid']) + \
                  '{0:02d}_cat_cpd.fits'.format(i),
                  clobber=True)

    if i != 1:
        # remove image
        remove("DECam_{0:08d}_{1:02d}.fits".format(args_dict['expid'], i))
