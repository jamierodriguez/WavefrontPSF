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
parser.add_argument("-c",
                    dest="csv",
                    default='/nfs/slac/g/ki/ki18/cpd/focus/september_27/image_data.csv',
                    help="where is the csv of the image data located")
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
                    default="/nfs/slac/g/ki/ki18/cpd/focus/november_8/",
                    help="where will the outputs go (modulo image number)")
parser.add_argument("-t",
                    dest="catalogs",
                    default='/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',
                    help='directory containing the catalogs')
parser.add_argument("-rid",
                    dest="rid",
                    help="Run ID")
parser.add_argument("-d",
                    dest="d",
                    help="Date")
parser.add_argument("-s",
                    dest="size",
                    default='16',
                    help="Stamp size")
parser.add_argument("-f",
                    dest="conds",
                    default='eli',  # eli's filterings
                    help="String for filter conditions")
options = parser.parse_args()

args_dict = vars(options)
args_dict['size'] = eval(args_dict['size'])
args_dict['expid'] = eval(args_dict['expid'])

# download the image and catalog
wget_command = ['python',
                '/u/ki/cpd/secret-adventure/examples/wgetscript.py',
                '-min', '{0}'.format(args_dict['expid']),
                '-max', '{0}'.format(args_dict['expid']),
                '-i', '1',
                '-rid', '{0}'.format(args_dict['rid']),
                '-d', '{0}'.format(args_dict['d'])]

call(wget_command)

csv = np.recfromcsv(args_dict['csv'], usemask=True)
image_data = csv[csv['expid'] == args_dict['expid']]

nPixels = args_dict['size']

# create the new catalogs
for i in xrange(1, 63):
    if i == 61:
        # screw N30
        continue

    # create an FP object
    path_base = args_dict['catalogs']
    list_catalogs_base = \
        path_base + '{0:08d}/DECam_{0:08d}_'.format(args_dict['expid'])
    list_catalogs = [list_catalogs_base + '{0:02d}_cat.fits'.format(i)]
    list_chip = [[decaminfo().ccddict[i]]]
    list_fits_extension = [[2]]

    FP = FocalPlane(image_data=image_data,
                    list_catalogs=list_catalogs,
                    list_fits_extension=list_fits_extension,
                    list_chip=list_chip,
                    path_mesh=args_dict['path_mesh'],
                    mesh_name=args_dict['mesh_name'],
                    boxdiv=0,
                    max_samples_box=200,
                    conds=args_dict['conds'],
                    )

    # now use the catalog here to make a new catalog
    image = pyfits.getdata(list_catalogs_base + '{0:02d}.fits'.format(i), ext=0)

    # generate dictionary of columns
    pyfits_dict = {}
    pyfits_dict.update({
        'XWIN_IMAGE': dict(name='XWIN_IMAGE',
                           array=FP.recdata['XWIN_IMAGE'],#[],
                           format='1D',
                           unit='pixel',),
        'YWIN_IMAGE': dict(name='YWIN_IMAGE',
                           array=FP.recdata['YWIN_IMAGE'],#[],
                           format='1D',
                           unit='pixel',),
        'MAG_AUTO':   dict(name='MAG_AUTO',
                           array=FP.recdata['MAG_AUTO'],#[],
                           format='1E',
                           unit='mag',),
        'CLASS_STAR': dict(name='CLASS_STAR',
                           array=FP.recdata['CLASS_STAR'],#[],
                           format='1E',),
        'FLAGS':      dict(name='FLAGS',
                           array=FP.recdata['FLAGS'],#[],
                           format='1I',),
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
                           format='256D')
        })

    copy_keys = ['XWIN_IMAGE', 'YWIN_IMAGE', 'MAG_AUTO', 'CLASS_STAR',
                 'FLAGS']

    Y, X = np.indices((16, 16)) + 0.5

    # go through each entry and make stamps and other parameters
    for recdata in FP.recdata:

        x_center = recdata[FP.x_coord_name]
        y_center = recdata[FP.y_coord_name]

        # coords are y,x ordered
        y_start = y_center - nPixels / 2
        y_end = y_center + nPixels / 2 + 1
        x_start = x_center - nPixels / 2
        x_end = x_center + nPixels / 2 + 1

        stamp = image[y_start:y_end, x_start:x_end].astype(np.float64)

        order_dict = {'x2': {'p': 2, 'q': 0},
                      'y2': {'p': 0, 'q': 2},
                      'xy': {'p': 1, 'q': 1},
                      'x3': {'p': 3, 'q': 0},
                      'y3': {'p': 0, 'q': 3},
                      'x2y': {'p': 2, 'q': 1},
                      'xy2': {'p': 1, 'q': 2}}

        moment_dict = FP.moments(stamp, indices=[Y, X],
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

    tbhdu.writeto(list_catalogs_base + '{0:02d}_cat_cpd.fits'.format(i),
                  clobber=True)

