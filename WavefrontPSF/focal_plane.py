#!/usr/bin/env python
"""
File: focal_plane.py
Author: Chris Davis
Description: Class for focal plane shell tied to specific image.

TODO: update attributes and methods
TODO: update __init__

"""

from __future__ import print_function, division
import numpy as np
from focal_plane_shell import FocalPlaneShell
from focal_plane_routines import image_zernike_corrections
import pyfits


class FocalPlane(FocalPlaneShell):
    """FocalPlaneShell tied to a specific image. Comparisons and such possible.

    Attributes
    ----------
    image_data

    image_correction

    average

    x_coord_name, y_coord_name

    recdata

    extension

    data
        Dictionary containing focal plane position and relevant moment
        information

    coords
        3 dimensional array of the stars used
        The first two coordinates correspond to the X and Y locations in
        the Focal Plane coordinate system (mm), while the third tells you
        the extension number

    Methods
    -------
    filter
        A method for generating coordinates from actual images

    filter_number_in_box
        Filter such that each box has no more than max_samples_box.

    create_data
        Create data attribute and coords

    combine_decam_catalogs
        assemble an array from all the focal plane chips

    """

    def __init__(self, image_data,
                 list_catalogs, list_fits_extension, list_chip,
                 max_samples_box=300, boxdiv=0,
                 path_mesh='/u/ec/roodman/Astrophysics/Donuts/Meshes/',
                 mesh_name="Science20120915s1v3_134239",
                 verbosity=['history'],
                 conds='default'):

        # do the old init for Wavefront
        super(FocalPlane, self).__init__(path_mesh, mesh_name, verbosity)
        # could put in nEle etc here

        self.image_data = image_data
        self.image_correction = image_zernike_corrections(image_data)

        self.average = np.mean
        self.boxdiv = boxdiv
        self.max_samples_box = max_samples_box

        self.x_coord_name = 'XWIN_IMAGE'
        self.y_coord_name = 'YWIN_IMAGE'

        # generate recdata
        recdata_unfiltered, \
            extension_unfiltered = \
            self.combine_decam_catalogs(
                list_catalogs, list_fits_extension, list_chip)

        conds = self.filter(
            recdata_unfiltered, conds)
        self.recdata = recdata_unfiltered[conds]
        self.extension = extension_unfiltered[conds]

        self.recdata, self.extension = \
            self.filter_number_in_box(recdata=self.recdata,
                                      extension=self.extension,
                                      max_samples_box=self.max_samples_box,
                                      boxdiv=self.boxdiv)

        self.data, self.coords = self.create_data(
                recdata=self.recdata,
                extension=self.extension)

        ## self.comparison_dict, self.comparison_data, \
        ##     self.comparison_extension, self.coords_comparison \
        ##         = self.comparison_coordinates(
        ##             self.comparison_data_unfiltered,
        ##             self.comparison_extension_unfiltered,
        ##             max_samples, conds)

        ## # generate comparisons with max_samples per chip
        ## self.comparison_data_unfiltered, \
        ##     self.comparison_extension_unfiltered = \
        ##     self.comparison(list_catalogs, list_fits_extension, list_chip)
        ## self.comparison_dict_all, self.comparison_data_all, \
        ##     self.comparison_extension_all, self.coords_all \
        ##         = self.comparison_coordinates(
        ##             self.comparison_data_unfiltered,
        ##             self.comparison_extension_unfiltered,
        ##             10000000, conds)
        ## self.comparison_dict, self.comparison_data, \
        ##     self.comparison_extension, self.coords_comparison \
        ##         = self.comparison_coordinates(
        ##             self.comparison_data_unfiltered,
        ##             self.comparison_extension_unfiltered,
        ##             max_samples, conds)
        ## # double check that coords_comparison hits all the boxes with a minimum
        ## # number of entries
        ## success = self.check_full_bounds(self.comparison_dict, boxdiv)
        ## while not success:
        ##     self.comparison_dict, self.comparison_data, \
        ##         self.comparison_extension, self.coords_comparison \
        ##             = self.comparison_coordinates(
        ##                 self.comparison_data_unfiltered,
        ##                 self.comparison_extension_unfiltered,
        ##                 max_samples, conds)
        ##     success = self.check_full_bounds(self.comparison_dict, boxdiv,
        ##                                      minimum_number=5)

        ## self.coords_random = self.random_coordinates(max_samples_box,
        ##                                              boxdiv=boxdiv)jk

    def combine_decam_catalogs(self, list_catalogs, list_fits_extension,
                               list_chip):
        """assemble an array from all the focal plane chips

        Parameters
        ----------
        list_catalogs : list
            a list pointing to all the catalogs we wish to combine.

        list_fits_extension : list of integers
            a list pointing which extension on a given fits file we open
            format: [[2], [3,4]] says for the first in list_catalog, combine
            the 2nd extension with the 2nd list_catalog's 3rd and 4th
            extensions.

        list_chip : list of strings
            a list containing the extension name of the chip. ie [['N1'],
            ['S29', 'S5']]

        Returns
        -------
        recdata_all : recarray
            The entire contents of all the fits extensions combined

        ext_all : array
            Array of all the extension names

        """

        for catalog_i in xrange(len(list_catalogs)):
            hdu_path = list_catalogs[catalog_i]

            try:
                hdu = pyfits.open(hdu_path)
            except IOError:
                print('Cannot open ', hdu_path)
                continue

            fits_extension_i = list_fits_extension[catalog_i]
            chip_i = list_chip[catalog_i]

            for fits_extension_ij in xrange(len(fits_extension_i)):
                ext_name = chip_i[fits_extension_ij]
                recdata = hdu[fits_extension_i[fits_extension_ij]].data

                try:
                    recdata_all = np.append(recdata_all, recdata)
                    ext_all = np.append(ext_all,
                                       [ext_name] * recdata.size)

                except NameError:
                    # haven't made recdata_combined yet!
                    recdata_all = recdata.copy()
                    ext_all = np.array([ext_name] * recdata.size)

            hdu.close()

        return recdata_all, ext_all

    def filter(self, recdata, conds=None):
        ##self, recdata, extension, max_samples=10000000, conds=None):
        """A method for generating coordinates from actual images

        Parameters
        ----------
        recdata : recarray
            recarray with all the recdata

        conds : string, optional
            List of conditions for filtering 'recdata'.  The borders are 30
            pixels. These basic cuts are always used.

            If no cut is specified, one is estimated from the fwhm of the
            image.

        Depreciated:
        extension : list
            Array of all the extension names

        max_samples : int, optional
            Maximum number of stars per chip that we will consider.
            If not specified, then use all the stars

        Returns
        -------
        conds : boolean array
            Returns which points satisfy the conditions

        """
        pixel_border = 30

        # now do the star selection
        if conds == 'default':
            # set the fwhm from the image recdata; account for mask
            if not np.ma.getmaskarray(self.image_data['fwhm']):
                fwhm = self.image_data['fwhm'][0]
            elif not np.ma.getmaskarray(self.image_data['qrfwhm']):
                fwhm = self.image_data['qrfwhm'][0]
            else:
                fwhm = 1.0

            conds = (
                (recdata['CLASS_STAR'] > 0.9) *
                (recdata['MAG_AUTO'] < 16) *
                (recdata['MAG_AUTO'] > 13) *
                (recdata['FWHM_WORLD'] > 0) *
                (recdata['FWHM_WORLD'] * 60 ** 2 < 2 * fwhm) *
                (recdata['FLAGS'] <= 3) *
                (recdata[self.x_coord_name] > pixel_border) *
                (recdata[self.x_coord_name] < 2048 - pixel_border) *
                (recdata[self.y_coord_name] > pixel_border) *
                (recdata[self.y_coord_name] < 4096 - pixel_border))
        elif conds == 'minimal':
            # eliminate base on flags and coordnames only
            conds = (
                (recdata['FLAGS'] <= 3) *
                (recdata[self.x_coord_name] > pixel_border) *
                (recdata[self.x_coord_name] < 2048 - pixel_border) *
                (recdata[self.y_coord_name] > pixel_border) *
                (recdata[self.y_coord_name] < 4096 - pixel_border))
        elif conds == 'all':
            # take everything
            conds = np.array([True] * recdata.size)
        elif conds == 'eli':
            """
            A set of cuts taken from:
            https://cdcvs.fnal.gov/redmine/projects/des-sci-verification/wiki/A_Modest_Proposal_for_Preliminary_StarGalaxy_Separation
            (FLAGS_I <=3) AND (((CLASS_STAR_I > 0.3) AND (MAG_AUTO_I < 18.0) AND (MAG_PSF_I < 30.0)) OR (((SPREAD_MODEL_I + 3*SPREADERR_MODEL_I) < 0.003) AND ((SPREAD_MODEL_I +3*SPREADERR_MODEL_I) > -0.003)))
            """
            conds = (
                ((recdata['FLAGS'] <= 3)) *
                (((recdata['CLASS_STAR'] > 0.3) *
                  (recdata['MAG_AUTO'] < 18.0) *
                  (recdata['MAG_PSF'] < 30.0)
                 ) +
                 ((recdata['SPREAD_MODEL'] +
                   3 * recdata['SPREADERR_MODEL'] < 0.003) *
                  (recdata['SPREAD_MODEL'] +
                   3 * recdata['SPREADERR_MODEL'] > -0.003)
                 )
                ) *
                ((recdata[self.x_coord_name] > pixel_border) *
                 (recdata[self.x_coord_name] < 2048 - pixel_border) *
                 (recdata[self.y_coord_name] > pixel_border) *
                 (recdata[self.y_coord_name] < 4096 - pixel_border)
                )
                )
        else:
            # evaluate the string
            conds = eval(conds)

        ## recdata_use = recdata[conds]
        ## extension_use = extension[conds]
        ## if (recdata_use.size > max_samples) * (max_samples > 0):
        ##     chooselist = np.arange(recdata_use.size)
        ##     np.random.shuffle(chooselist)
        ##     recdata_use = recdata_use[chooselist[:max_samples]]
        ##     extension_use = extension_use[chooselist[:max_samples]]

        ## coords = np.array([self.decaminfo.getPosition(
        ##     extension_use[i],
        ##     recdata_use[self.x_coord_name][i],
        ##     recdata_use[self.y_coord_name][i])
        ##                     for i in xrange(len(extension_use))])
        ## extNumbers = [self.decaminfo.infoDict[i]['CCDNUM']
        ##               for i in extension_use]

        ## coords_final = np.append(coords.T, [extNumbers], axis=0).T

        ## comparison_dict = dict(
        ##         x=coords_final[:,0], y=coords_final[:,1],
        ##         x2=recdata_use['X2WIN_IMAGE'].astype(np.float64),
        ##         y2=recdata_use['Y2WIN_IMAGE'].astype(np.float64),
        ##         xy=recdata_use['XYWIN_IMAGE'].astype(np.float64),
        ##         fwhm=recdata_use['FWHM_WORLD'].astype(np.float64) * 3600,)


        ## return comparison_dict, recdata_use, extension_use, coords_final
        return conds

    def filter_number_in_box(
            self, recdata, extension, max_samples_box, boxdiv=0):
        """Filter such that each box has no more than max_samples_box.

        Parameters
        ----------
        recdata : dictionary
            contains the example sampling.

        extension : list
            Array of all the extension names

        boxdiv : int
            How many divisions we will put into the chip. Default is zero
            divisions on each chip.

        max_samples_box : int
            The max number that should be in each box.

        Returns
        -------
        recdata_return : dictionary
            contains the example sampling. Filtered

        extension_return : list
            Array of all the extension names. Filtered


        """

        recdata_return = np.copy(recdata)
        extension_return = np.copy(extension)
        # create the bounds [[x0, x1, xn], [y0, y1, y2, yn]]
        bounds = []
        for i in range(1, 63):
            if i == 61:
                #n30 sucks
                continue
            extname = self.decaminfo.ccddict[i]
            box = self.decaminfo.getBounds(extname, boxdiv)
            bounds.append(box)

            for x in range(len(box[0]) - 1):
                for y in range(len(box[1]) - 1):
                    xmin = box[0][x]
                    xmax = box[0][x + 1]
                    ymin = box[1][y]
                    ymax = box[1][y + 1]
                    # convert coordinates to pixel coordinates
                    xmin, ymin = self.decaminfo.getPixel(extname, xmin, ymin)
                    xmax, ymax = self.decamaxfo.getPixel(extname, xmax, ymax)

                    conds = (
                        (recdata_return[self.x_coord_name] > xmin) *
                        (recdata_return[self.x_coord_name] < xmax) *
                        (recdata_return[self.y_coord_name] > ymin) *
                        (recdata_return[self.y_coord_name] < ymax))

                    # This is pretty kludgey.

                    # find the number of Trues we need to exclude
                    N = np.sum(conds) - max_samples_box
                    # we want the False's AND only max_samples (or all, if less
                    # than max_samples) of True's !
                    if N > 0:
                        true_list = [True] * N + [False] * (np.sum(conds) - N)
                        np.random.shuffle(true_list)
                        indices = np.nonzero(conds)[0]
                        for i in xrange(len(true_list)):
                            conds[indices[i]] = true_list[i]

                        # select the False's
                        recdata_return = recdata_return[~conds]
                        extension_return = extension_return[~conds]

        return recdata_return, extension_return


    def create_data(self, recdata, extension):
        """Create the data attribute

        Parameters
        ----------
        recdata : recarray
            recarray with all the recdata

        extension : list
            Array of all the extension names

        Returns
        -------
        data : dictionary
            dictionary of important attributes

        coords : array
            3 dimensional array of the stars used
            The first two coordinates correspond to the X and Y locations in
            the Focal Plane coordinate system (mm), while the third tells you
            the extension number

        """

        coords = np.array([self.decaminfo.getPosition(
            extension[i],
            recdata[self.x_coord_name][i],
            recdata[self.y_coord_name][i])
                            for i in xrange(len(extension))])
        extNumbers = [self.decaminfo.infoDict[i]['CCDNUM']
                      for i in extension]
        coords = np.append(coords.T, [extNumbers], axis=0).T

        data = dict(
                x=coords[:,0], y=coords[:,1],
                x2=recdata['X2WIN_IMAGE'].astype(np.float64),
                y2=recdata['Y2WIN_IMAGE'].astype(np.float64),
                xy=recdata['XYWIN_IMAGE'].astype(np.float64),
                fwhm=recdata['FWHM_WORLD'].astype(np.float64) * 3600,)

        return data, coords
