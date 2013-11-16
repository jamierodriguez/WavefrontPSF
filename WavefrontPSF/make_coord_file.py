#!/usr/bin/env python
# make_coord_file.py
from __future__ import print_function, division
import numpy as np
from os import path, makedirs
import astropy.fits.io as pyfits
from donutlib.decamutil import decaminfo
#TODO: PEP8 compliant; comments
#TODO: use focal_plane method for comparison to make data array here.

# load up the convenience decam dictionaries
decaminf = decaminfo()
ccddict = {}
for keyi in decaminf.infoDict.keys():
    ccddict.update({decaminf.infoDict[keyi]['CCDNUM']: keyi})

keys = decaminf.infoDict.keys()[:-8]
ext_names_list = [keys[i:i+5] for i in range(0, len(keys), 5)]


def make_coord_file(path_catalog, image_number, external_conds='True', max_use=0,
                    outpath=None):
    """from external_conds, draw locations from the firstcut catalogs

    Parameters
    ----------

    path_catalog : string
        Location of directory containing the catalogs.
        Catalogs themselves are assumed to be in format
        DECam_{image_number}_{extension_number}_cat.fits

    image_number : int
        Image number of the DECam file

    external_conds : string, optional
        List of conditions for filtering 'data'
        If not specified, then just the most basic cuts on borders and flags is
        used. The borders are 30 pixels. These basic cuts are always used.

    max_use : int, optional
        Maximum number of stars per chip that we will consider.
        If not specified, then use all the stars

    outpath : string, optional
        Path to output the selected coordinates.
        If not specified, then no file is created.

    Returns
    -------

    coords_final : array
        3 dimensional array of the stars used
        The first two coordinates correspond to the X and Y locations in the
        Focal Plane coordinate system (mm), while the third tells you the
        extension number


    """

    # load up the coordinates from the FITS catalogs
    extNum = 1
    extName = ccddict[extNum]
    hdu = pyfits.open(path_catalog + 'DECam_{0:08}_{1:02}_cat.fits'.
                      format(image_number, extNum))
    data = hdu[2].data
    pixel_border = 30  # keep a safe distance from edge
    conds = np.where(
                    (data['FLAGS'] == 0) *
                    (data['XWIN_IMAGE'] > pixel_border) *
                    (data['XWIN_IMAGE'] < 2048 - pixel_border) *
                    (data['YWIN_IMAGE'] > pixel_border) *
                    (data['YWIN_IMAGE'] < 4096 - pixel_border) *
                    (eval(external_conds)))
    data_use = data[conds]
    if (data_use.size > max_use) * (max_use > 0):
        chooselist = np.arange(data_use.size)
        np.random.shuffle(chooselist)
        data_use = data_use[chooselist[:max_use]]

    ext_all = [extName for i in xrange(data_use.size)]
    x_all = data_use['XWIN_IMAGE']
    y_all = data_use['YWIN_IMAGE']

    hdu.close()

    for extNum in xrange(2, 63):
        extName = ccddict[extNum]
        try:
            hdu = pyfits.open(path_catalog + 'DECam_{0:08}_{1:02}_cat.fits'.
                              format(image_number, extNum))
        except IOError:
            print('Cannot find extNum ', extNum)
            continue
        data = hdu[2].data
        conds = np.where(
                        (data['FLAGS'] == 0) *
                        (data['XWIN_IMAGE'] > pixel_border) *
                        (data['XWIN_IMAGE'] < 2048 - pixel_border) *
                        (data['YWIN_IMAGE'] > pixel_border) *
                        (data['YWIN_IMAGE'] < 4096 - pixel_border) *
                        (eval(external_conds)))

        data_use = data[conds]
        if (data_use.size > max_use) * (max_use > 0):
            chooselist = np.arange(data_use.size)
            np.random.shuffle(chooselist)
            data_use = data_use[chooselist[:max_use]]

        for i in xrange(data_use.size):
            ext_all.append(extName)

        x_all = np.append(x_all, data_use['XWIN_IMAGE'])
        y_all = np.append(y_all, data_use['YWIN_IMAGE'])
        # axis is so we don't flatten the thing.

        hdu.close()

    # get x and y coords in mm on focal plane
    coords = np.array([decaminf.getPosition(ext_all[i], x_all[i], y_all[i])
                       for i in xrange(len(ext_all))])
    extNumbers = [decaminf.infoDict[i]['CCDNUM'] for i in ext_all]

    coords_final = np.append(coords.T, [extNumbers], axis=0)

    if outpath:
        if not path.exists(path.dirname(outpath)):
            makedirs(path.dirname(outpath))
        np.savetxt(outpath, coords_final.T,
                   fmt='%f %f %i')

    return coords_final

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=
                                     'Make coordinates for artificial ' +
                                     'star catalog.')

    parser.add_argument("-cp", "--coordinates_path",
                        dest="coords_path",
                        default="/nfs/slac/g/ki/ki18/cpd/focus/roodman.dat",
                        help="Location of example set of coords"
                        )
    parser.add_argument("-in", "--image_number",
                        dest="image_number",
                        default=158999,
                        type=int,
                        help="Image number for reference header."
                        )
    parser.add_argument("-mx", "--max_stars",
                        dest="max_stars",
                        default=200,
                        type=int,
                        help="Maximum number of stars to use per chip."
                        )
    parser.add_argument("-e", "--external",
                        dest="external_conds",
                        default='True',
                        help="Filtering conditions, e.g. " +
                             "'(data['CLASS_STAR'] > 0.9)'."
                        )

    options = parser.parse_args()
    aDict = vars(options)

    # make coordinate file
    in_catalog_path = \
        "/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/{0:08}/". \
        format(aDict['image_number'])

    positions = make_coord_file(in_catalog_path, aDict['coords_path'],
                                aDict['image_number'],
                                aDict['max_stars'], aDict['external_conds'])
