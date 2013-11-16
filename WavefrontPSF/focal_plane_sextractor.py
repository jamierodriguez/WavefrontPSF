#!/usr/bin/env python
# focalplane.py
from __future__ import print_function, division
import numpy as np
from focal_plane_shell import FocalPlaneShell
import astropy.fits.io as pyfits
from os import path, makedirs
from subprocess import call
# TODO: DOCS!!
# TODO: update attributes and methods

class FocalPlaneSextractor(FocalPlaneShell):
    """FocalPlaneShell tied to a specific image. Comparisons and such possible.

    Attributes
    ----------



    Methods
    -------


    """

    def image(self, zernikes, rzero, coords,
            output_directory=None):
        '''
        coords are in pixel coordinates!
        '''

        decam_x_size = 2048
        decam_y_size = 4096
        # you can also specify where in relation to the grid the "xdecam" etc
        # references
        decam_x_center = decam_x_size/2
        decam_y_center = decam_y_size/2
        decam_dict = {'x_size': decam_x_size, 'y_size': decam_y_size,
                      'x_center': decam_x_center, 'y_center': decam_y_center}

        # the array goes f[y,x]!!
        decam_grid = np.zeros((decam_dict['y_size'], decam_dict['x_size']))

        self.background = 0
        for i in range(len(coords)):
            coord = coords[i]
            zernike = zernikes[i]
            stamp_data = self.stamp(zernike, rzero, coord)
            # add to grid; account for edge
            xpix_lower = np.int(coord[0] - stamp_data.shape[1] / 2)
            xpix_upper = xpix_lower + stamp_data.shape[1]
            ypix_lower = np.int(coord[1] - stamp_data.shape[0] / 2)
            ypix_upper = ypix_lower + stamp_data.shape[0]
            xstamp_upper = min(stamp_data.shape[1],
                               stamp_data.shape[1] -
                               (xpix_upper - decam_x_size))
            xstamp_lower = max(0, - xpix_lower)
            ystamp_upper = min(stamp_data.shape[0],
                               stamp_data.shape[0] -
                               (ypix_upper - decam_y_size))
            ystamp_lower = max(0, - ypix_lower)

            ypix_lower = max(0, ypix_lower)
            xpix_lower = max(0, xpix_lower)
            try:
                # data goes [y,x]!!!
                #print(ypix_lower, ypix_upper, xpix_lower, xpix_upper,
                #        ystamp_lower, ystamp_upper, xstamp_lower, xstamp_upper)
                decam_grid[ypix_lower:ypix_upper, xpix_lower:xpix_upper] += \
                    stamp_data[
                        ystamp_lower:ystamp_upper,
                        xstamp_lower:xstamp_upper]
            except ValueError as err:
                # if we have an empty array, let's not worry
                print('I believe we have an empty array (how?!) for image at ' +
                      'location ', coord)
                print(err)
            del stamp_data

        # add a non-zero background level
        decam_grid = decam_grid + self.background

        # smear by photo-statistics
        nranval = np.random.normal(0.0, 1.0, decam_grid.shape)
        decam_grid = decam_grid + nranval * np.sqrt(decam_grid)

        # save file?
        if output_directory:
            hdu_corr = pyfits.ImageHDU(data=decam_grid)
            if not path.exists(path.dirname(output_directory)):
                makedirs(path.dirname(output_directory))
            hdu_corr.writeto(output_directory, clobber=True)
            del hdu_corr

        return decam_grid

    def analyze_sextractor(self, in_dict,
            coords_in, path_out,
            sex_config_path='/afs/slac.stanford.edu/u/ki/cpd/WavefrontPSF/sex.config_cpd'):
        # do each chip
        temp_cat = path_out + \
                        '/images/stamp_sex_temp.fits'
        final_cat = path_out + \
                        '/images/stamp_sex.fits'
        temp_fits = path_out + \
                        '/images/stamp.fits'

        zernike_corrections = self.zernike_corrections_from_dictionary(in_dict)
        if 'rzero' in in_dict.keys():
            rzero = in_dict['rzero']
        else:
            rzero = 0.14

        x2 = []
        y2 = []
        xy = []
        fwhm = []
        for ext_num in range(1, 63):
            if ext_num == 61:
                #N30 sucks
                continue

            coordmm = coords_in[np.where(coords_in[:, 2] == ext_num)]
            # make zernike list
            zernikes = self.zernikes(coordmm,
                zernike_corrections)

            # make the grid
            # get the coords in pixels
            coords = [
                self.decaminfo.getPixel(self.decaminfo.ccddict[coordmmj[2]],
                                        coordmmj[0], coordmmj[1])
                for coordmmj in coordmm]  # pixel coordinates
            self.image(zernikes=zernikes, rzero=rzero, coords=coords,
                    output_directory=temp_fits)

            # make sex catalog
            call(['sex',
                  temp_fits,
                  '-c', sex_config_path,
                  '-CATALOG_NAME', temp_cat
                  ])
            cat = pyfits.open(temp_cat)
            data = cat[2].data
            header = cat[2].header

            # filter out bad data
            #data = data[data.field("FLUX_AUTO") > 1e4]
            header['EXTNAME'] = self.decaminfo.ccddict[ext_num]

            pyfits.append(final_cat, data, header=header)

            [x2.append(i) for i in data['X2WIN_IMAGE'].astype(np.float64)]
            [y2.append(i) for i in data['Y2WIN_IMAGE'].astype(np.float64)]
            [xy.append(i) for i in data['XYWIN_IMAGE'].astype(np.float64)]
            [fwhm.append(i * 0.27)
                for i in data['FWHM_IMAGE'].astype(np.float64)]
            cat.close()

        # load up the final pyfits catalog
        cat = pyfits.open(final_cat)

        comparison_dict = dict(
                x=coords_in[:,0], y=coords_in[:,1],
                x2=np.array(x2),
                y2=np.array(y2),
                xy=np.array(xy),
                fwhm=np.array(fwhm))
        comparison_dict.update(dict(
                e1=(comparison_dict['x2'] - comparison_dict['y2']) * 0.27 ** 2,
                e2=2 * comparison_dict['xy'] * 0.27 ** 2,
                e0=(comparison_dict['x2'] +
                    comparison_dict['y2']) * 0.27 ** 2))
        return cat, comparison_dict

