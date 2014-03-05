#!/usr/bin/env python
"""
File: batch_catalog.py
Author: Chris Davis
Description: File that takes a set of DES images and object catalogs and makes
a catalog I would want (i.e. with the parameters I am interested in and an
associated image stamp.
"""

from __future__ import print_function, division

import numpy as np
import pyfits
import argparse
from os import path, makedirs, chdir, remove

from focal_plane import FocalPlane
from decamutil_cpd import decaminfo

from routines_files import download_desdm
from routines import MAD

"""
TODO:
    [ ] Get rid of the WIN_IMAGE tags?
    [ ] Create a basic set of scales for both parameters

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

def check_centered_peak(stamp_in,
                        background, threshold,
                        xwin_image, ywin_image,
                        x2win_image, xywin_image, y2win_image,
                        return_stamps=False):

    # filter via background and threshold
    stamp = stamp_in.copy()
    stamp -= background
    stamp = np.where(stamp > threshold, stamp, 0)

    stamp = stamp.reshape(args_dict['size'], args_dict['size'])

    # do window
    max_nsig2 = 25
    y, x = np.indices(stamp.shape)
    Mx = args_dict['size'] / 2 + xwin_image % int(xwin_image) - 1
    My = args_dict['size'] / 2 + ywin_image % int(ywin_image) - 1

    r2 = (x - Mx) ** 2 + (y - My) ** 2

    Mxx = 2 * x2win_image
    Myy = 2 * y2win_image
    Mxy = 2 * xywin_image
    detM = Mxx * Myy - Mxy * Mxy
    Minv_xx = Myy / detM
    TwoMinv_xy = -Mxy / detM * 2.0
    Minv_yy = Mxx / detM
    Inv2Minv_xx = 0.5 / Minv_xx
    rho2 = Minv_xx * (x - Mx) ** 2 + TwoMinv_xy * (x - Mx) * (y - My) + Minv_yy * (y - My) ** 2
    stamp = np.where(rho2 < max_nsig2, stamp, 0)

    # take central 14 x 14 pixels and 5 x 5
    center = args_dict['size'] / 2
    if center > 10:
        cutout = stamp[center - 10: center + 10, center - 10: center + 10]
    else:
        cutout = stamp
    central_cutout = stamp[center - 3: center + 3, center - 3: center + 3]

    if return_stamps:
        return cutout, central_cutout
    else:
        return (np.max(cutout) == np.max(central_cutout))

def check_recdata(recdata, return_stamps=True):
    val = check_centered_peak(recdata['STAMP'],
                              recdata['BACKGROUND'], recdata['THRESHOLD'],
                              recdata['XWIN_IMAGE'], recdata['YWIN_IMAGE'],
                              recdata['X2WIN_IMAGE'], recdata['XYWIN_IMAGE'],
                              recdata['Y2WIN_IMAGE'],
                              return_stamps)
    return val

##############################################################################
# create the new catalogs
##############################################################################
list_catalogs_base = \
    path.expandvars(args_dict['output_directory'])
for i in xrange(1, 63):

    if i == 61:
        # screw N30
        continue

    # get cat and image
    download_desdm(args_dict['expid'], list_catalogs_base, ccd=i)

    ## # get cat
    ## download_cat(args_dict['rid'], args_dict['expid'], args_dict['date'], i)
    ## # get image
    ## download_image(args_dict['rid'], args_dict['expid'], args_dict['date'], i)

    # create an FP object
    list_catalogs = [list_catalogs_base +
                     'DECam_{0:08d}_{1:02d}_cat.fits'.format(
                         args_dict['expid'], i)]
    list_chip = [[decaminfo().ccddict[i]]]
    list_fits_extension = [[2]]

    FP = FocalPlane(list_catalogs=list_catalogs,
                    list_fits_extension=list_fits_extension,
                    list_chip=list_chip,
                    boxdiv=0,
                    max_samples_box=20000,
                    conds=args_dict['conds'],
                    nPixels=args_dict['size'],
                    )

    # now use the catalog here to make a new catalog
    image = pyfits.getdata(list_catalogs_base +
                           'DECam_{0:08d}_{1:02d}.fits.fz'.format(
                           args_dict['expid'], i),
                           ext=1)
    header = pyfits.getheader(list_catalogs_base +
                              'DECam_{0:08d}_{1:02d}.fits.fz'.format(
                              args_dict['expid'], i),
                              ext=1)
    gain = np.mean([header['GAINA'], header['GAINB']])
    gain_std = np.std([header['GAINA'], header['GAINB']])
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
        'FLUX_AUTO':   dict(name='FLUX_AUTO',
                           array=FP.recdata['FLUX_AUTO'],
                           format='1E',
                           unit='counts',),
        'FLUXERR_AUTO':   dict(name='FLUXERR_AUTO',
                           array=FP.recdata['FLUXERR_AUTO'],
                           format='1E',
                           unit='counts',),
        'MAG_AUTO':   dict(name='MAG_AUTO',
                           array=FP.recdata['MAG_AUTO'],
                           format='1E',
                           unit='mag',),
        'MAGERR_AUTO':   dict(name='MAGERR_AUTO',
                           array=FP.recdata['MAGERR_AUTO'],
                           format='1E',
                           unit='mag',),
        'MAG_PSF':   dict(name='MAG_PSF',
                           array=FP.recdata['MAG_PSF'],
                           format='1E',
                           unit='mag',),
        'MAGERR_PSF':   dict(name='MAGERR_PSF',
                           array=FP.recdata['MAGERR_PSF'],
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

        'SN_FLUX': dict(name='SN_FLUX',
                           array=2.5 / np.log(10) / FP.recdata['MAGERR_AUTO'],
                           format='1E',),

        # to chip number
        'CHIP': dict(name='CHIP',
                     array=np.ones(FP.recdata['MAGERR_AUTO'].size) * i,
                     format='1I'),

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

        'THRESHOLD':      dict(name='THRESHOLD',
                           array=[],
                           format='1E',
                           unit='count',),
        'WHISKER': dict(name='WHISKER',
                           array=[],
                           format='1D',
                           unit='pixel',),
        'FLUX_ADAPTIVE': dict(name='FLUX_ADAPTIVE',
                           array=[],
                           format='1D',
                           unit='counts',),
        'A4_ADAPTIVE': dict(name='A4_ADAPTIVE',
                           array=[],
                           format='1D',
                           unit='pixel**4',),

        'STAMP':      dict(name='STAMP',
                           array=[],
                           format='{0}D'.format(args_dict['size'] ** 2),
                           unit='count',),
        })

    # go through each entry and make stamps and other parameters
    for recdata in FP.recdata:

        x_center = recdata['X' + FP.coord_name]
        y_center = recdata['Y' + FP.coord_name]

        # coords are y,x ordered
        y_start = int(y_center - args_dict['size'] / 2)
        y_end = y_start + args_dict['size']
        x_start = int(x_center - args_dict['size'] / 2)
        x_end = x_start + args_dict['size']

        stamp = image[y_start:y_end, x_start:x_end].astype(np.float64)

        background = recdata['BACKGROUND']
        # only cut off at one std; it seems like 2 std actually biases the
        # data...
        # 1 std and esp 2 std are plenty for 16 x 16...
        threshold = np.sqrt(background / gain)
        moment_dict = FP.moments(stamp,
                                 background=background,
                                 threshold=threshold,
                                 )

        # append to pyfits_dict
        pyfits_dict['STAMP']['array'].append(
            stamp.flatten())
        pyfits_dict['THRESHOLD']['array'].append(
            threshold)
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

        pyfits_dict['WHISKER']['array'].append(
            moment_dict['whisker'])
        pyfits_dict['FLUX_ADAPTIVE']['array'].append(
            moment_dict['flux'])
        pyfits_dict['A4_ADAPTIVE']['array'].append(
            moment_dict['a4'])

    # almost universally, any a4 > 0.3 is either a completely miscentered
    # image or a cosmic (though there are some of both that have a4 < 0.3!)
    conds = ((np.array(pyfits_dict['A4_ADAPTIVE']['array']) < 0.15) *
             (np.array(pyfits_dict['A4_ADAPTIVE']['array']) > 0) *
             (np.array(pyfits_dict['FLUX_ADAPTIVE']['array']) > 10))
    # cull crazy things
    mad_keys =  ['X2WIN_IMAGE', 'XYWIN_IMAGE', 'Y2WIN_IMAGE',
                 'X3WIN_IMAGE', 'X2YWIN_IMAGE', 'XY2WIN_IMAGE', 'Y3WIN_IMAGE',
                 'A4_ADAPTIVE', 'WHISKER', 'FWHM_WORLD', 'FLUX_RADIUS']
    for key in mad_keys:
        conds *= MAD(pyfits_dict[key]['array'], sigma=5)[0]
    # cut in the stamps
    for ij in xrange(conds.size):
        conds[ij] *= check_centered_peak(
            pyfits_dict['STAMP']['array'][ij],
            pyfits_dict['BACKGROUND']['array'][ij],
            pyfits_dict['THRESHOLD']['array'][ij],
            pyfits_dict['XWIN_IMAGE']['array'][ij],
            pyfits_dict['YWIN_IMAGE']['array'][ij],
            pyfits_dict['X2WIN_IMAGE']['array'][ij],
            pyfits_dict['XYWIN_IMAGE']['array'][ij],
            pyfits_dict['Y2WIN_IMAGE']['array'][ij],)
    # create the columns
    columns = []
    for key in pyfits_dict:
        pyfits_dict[key]['array'] = np.array(pyfits_dict[key]['array'])[conds]
        columns.append(pyfits.Column(**pyfits_dict[key]))

    # TODO: I ought to include the old 1st extension (with image header info)
    tbhdu = pyfits.new_table(columns)

    tbhdu.writeto(list_catalogs_base + \
                  'DECam_{0:08d}_'.format(args_dict['expid']) + \
                  '{0:02d}_cat_cpd.fits'.format(i),
                  clobber=True)

    if i != 1:
        # remove image
        remove(list_catalogs_base +
               "DECam_{0:08d}_{1:02d}.fits.fz".format(args_dict['expid'], i))
