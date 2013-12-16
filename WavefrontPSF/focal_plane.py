#!/usr/bin/env python
"""
File: focal_plane.py
Author: Chris Davis
Description: Class for focal plane shell tied to specific image.

TODO: DOCS!!
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



    Methods
    -------


    """

    def __init__(self, image_data,
                 list_catalogs, list_fits_extension, list_chip,
                 max_samples=750,
                 max_samples_box=10, boxdiv=0,
                 path_mesh='/u/ec/roodman/Astrophysics/Donuts/Meshes/',
                 mesh_name="Science20120915s1v3_134239",
                 verbosity=['history'],
                 conds=None):

        # do the old init for Wavefront
        super(FocalPlane, self).__init__(path_mesh, mesh_name, verbosity)
        # could put in nEle etc here

        self.image_data = image_data
        self.image_correction = image_zernike_corrections(image_data)

        self.average = np.mean

        # generate comparisons with max_samples per chip
        self.comparison_data_unfiltered, \
            self.comparison_extension_unfiltered = \
            self.comparison(list_catalogs, list_fits_extension, list_chip)
        self.comparison_dict_all, self.comparison_data_all, \
            self.comparison_extension_all, self.coords_all \
                = self.comparison_coordinates(
                    self.comparison_data_unfiltered,
                    self.comparison_extension_unfiltered,
                    10000000, conds)
        self.comparison_dict, self.comparison_data, \
            self.comparison_extension, self.coords_comparison \
                = self.comparison_coordinates(
                    self.comparison_data_unfiltered,
                    self.comparison_extension_unfiltered,
                    max_samples, conds)
        # double check that coords_comparison hits all the boxes with a minimum
        # number of entries
        success = self.check_full_bounds(self.comparison_dict, boxdiv)
        while not success:
            self.comparison_dict, self.comparison_data, \
                self.comparison_extension, self.coords_comparison \
                    = self.comparison_coordinates(
                        self.comparison_data_unfiltered,
                        self.comparison_extension_unfiltered,
                        max_samples, conds)
            success = self.check_full_bounds(self.comparison_dict, boxdiv,
                                             minimum_number=5)



        self.coords_random = self.random_coordinates(max_samples_box,
                                                     boxdiv=boxdiv)

    def check_full_bounds(self, data, boxdiv, minimum_number):
        """Convenience method for checking whether my random sampling hits all
        possible divisions over the chip.

        Parameters
        ----------
        data : dictionary
            contains the example sampling.

        boxdiv : int
            How many divisions we will put into the chip. Default is zero
            divisions on each chip.

        minimum_number : int
            The number that should be in each box.

        Returns
        -------

        success : bool
            True / False for whether the lengths match.

        """

        bounds = []
        for i in range(1, 63):
            if i == 61:
                #n30 sucks
                continue
            extname = self.decaminfo.ccddict[i]
            boundi = self.decaminfo.getBounds(extname, boxdiv)
            bounds.append(boundi)
        # get the midpoints of each box
        x_box = []
        for box in bounds:
            for x in range(len(box[0]) - 1):
                for y in range(len(box[1]) - 1):
                    x_box.append((box[0][x] + box[0][x + 1]) / 2.)
        # average x coord
        x = data['x']
        y = data['y']
        x_av, x_av2, N, _ = self.decaminfo.average_boxdiv(x, y, x, self.average
                                                          boxdiv=boxdiv,
                                                          Ntrue=True)
        # check that all N >= minimum_number

        success = (len(x_av) == len(x_box)) * np.all(N >= minimum_number)

        return success


    def random_coordinates(self, max_samples_box=5, boxdiv=0):
        """A method for generating coordinates by sampling over boxes

        Parameters
        ----------
        max_samples_box : int, optional
            Integer for the maximum number of stars per box that we sample
            from. Default is 5 stars per box.

        boxdiv : int, optional
            How many divisions we will put into the chip. Default is zero
            divisions on each chip.

        Returns
        -------
        coords_final : array
            3 dimensional array of the stars used
            The first two coordinates correspond to the X and Y locations in
            the Focal Plane coordinate system (mm), while the third tells you
            the extension number

        """

        # sample over [a,b) is
        # (b - a ) * np.random.random_sample(max_samples_box) + a
        coords_final = []
        for ext_num in range(1, 63):
            ext_name = self.decaminfo.ccddict[ext_num]
            if ext_name == 'N31':
                # N31 is bad
                continue
            boundaries = self.decaminfo.getBounds(ext_name, boxdiv=boxdiv)
            for x in xrange(len(boundaries[0]) - 1):
                for y in xrange(len(boundaries[1]) - 1):
                    # get the bounds
                    x_lower = boundaries[0][x]
                    x_upper = boundaries[0][x + 1]
                    y_lower = boundaries[1][y]
                    y_upper = boundaries[1][y + 1]

                    # make the uniform sample
                    x_samples = (x_upper - x_lower) * np.random.random_sample(
                        max_samples_box) + x_lower
                    y_samples = (y_upper - y_lower) * np.random.random_sample(
                        max_samples_box) + y_lower
                    for i in xrange(max_samples_box):
                        coord = [x_samples[i], y_samples[i], ext_num]
                        coords_final.append(coord)
        coords_final = np.array(coords_final)

        return coords_final

    def comparison_coordinates(
            self, data, extension, max_samples=10000000, conds=None):
        """A method for generating coordinates from actual images

        Parameters
        ----------
        data : recarray
            recarray with all the data

        conds : string, optional
            List of conditions for filtering 'data'.  The borders are 30
            pixels. These basic cuts are always used.

            If no cut is specified, one is estimated from the fwhm of the
            image.

        max_samples : int, optional
            Maximum number of stars per chip that we will consider.
            If not specified, then use all the stars

        Returns
        -------
        coords_final : array
            3 dimensional array of the stars used
            The first two coordinates correspond to the X and Y locations in
            the Focal Plane coordinate system (mm), while the third tells you
            the extension number

        """

        # now do the star selection
        if not conds:
            # set the fwhm from the image data; account for mask
            if not np.ma.getmaskarray(self.image_data['fwhm']):
                fwhm = self.image_data['fwhm'][0]
            elif not np.ma.getmaskarray(self.image_data['qrfwhm']):
                fwhm = self.image_data['qrfwhm'][0]
            else:
                fwhm = 1.3

            pixel_border = 30

            # TODO: consider cuts on flux radius instead of fwhm world?
            conds = (
                (data['CLASS_STAR'] > 0.9) *
                (data['MAG_AUTO'] < 16) *
                (data['MAG_AUTO'] > 13) *
                (data['FWHM_WORLD'] > 0) *
                (data['FWHM_WORLD'] * 60 ** 2 < 2 * fwhm) *
                (data['FLAGS'] <= 0) *
                (data['XWIN_IMAGE'] > pixel_border) *
                (data['XWIN_IMAGE'] < 2048 - pixel_border) *
                (data['YWIN_IMAGE'] > pixel_border) *
                (data['YWIN_IMAGE'] < 4096 - pixel_border))
        else:
            # evaluate the string
            conds = eval(conds)

        data_use = data[conds]
        extension_use = extension[conds]
        if (data_use.size > max_samples) * (max_samples > 0):
            chooselist = np.arange(data_use.size)
            np.random.shuffle(chooselist)
            data_use = data_use[chooselist[:max_samples]]
            extension_use = extension_use[chooselist[:max_samples]]

        coords = np.array([self.decaminfo.getPosition(
            extension_use[i],
            data_use['XWIN_IMAGE'][i],
            data_use['YWIN_IMAGE'][i])
                            for i in xrange(len(extension_use))])
        extNumbers = [self.decaminfo.infoDict[i]['CCDNUM']
                      for i in extension_use]

        coords_final = np.append(coords.T, [extNumbers], axis=0).T

        comparison_dict = dict(
                x=coords_final[:,0], y=coords_final[:,1],
                x2=data_use['X2WIN_IMAGE'].astype(np.float64),
                y2=data_use['Y2WIN_IMAGE'].astype(np.float64),
                xy=data_use['XYWIN_IMAGE'].astype(np.float64),
                fwhm=data_use['FWHM_WORLD'].astype(np.float64) * 3600,)


        return comparison_dict, data_use, extension_use, coords_final


    def comparison(self, list_catalogs, list_fits_extension,
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
        data_all : recarray
            The entire contents of all the fits extensions combined

        ext_all : list
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
                data = hdu[fits_extension_i[fits_extension_ij]].data

                try:
                    data_all = np.append(data_all, data)
                    ext_all = np.append(ext_all,
                                       [ext_name] * data.size)

                except NameError:
                    # haven't made data_combined yet!
                    data_all = data.copy()
                    ext_all = np.array([ext_name] * data.size)

            hdu.close()

        return data_all, ext_all

    def comparison_filter(self, coords, data_all, ext_all):
        # do combining here (not sure why I was doing it earlier before)
        # probably because you are now comparing with about 63x data more
        # than you need to.
        coords_image = np.array([self.decaminfo.getPixel(
            self.decaminfo.ccddict[int(coords[i][2])],
            coords[i][0],
            coords[i][1])
            for i in xrange(coords[:, 0].size)])
        chosen = []
        chosen_coords = []
        # cut down on our matrices by using the extension
        for ext_num in xrange(1, 63):
            ext_name = self.decaminfo.ccddict[ext_num]
            # skip N30, cause it usually sucks
            if ext_name == 'N30':
                continue

            coords_used_indices = np.nonzero(coords[:, 2] == ext_num)[0]
            data_all_used_indices = np.nonzero(ext_all == ext_name)[0]
            x_compare, x_star = np.meshgrid(
                coords_image[:, 0][coords_used_indices],
                data_all['XWIN_IMAGE'][data_all_used_indices])
            y_compare, y_star = np.meshgrid(
                coords_image[:, 1][coords_used_indices],
                data_all['YWIN_IMAGE'][data_all_used_indices])
            test_statistic = np.square(x_star - x_compare) + \
                             np.square(y_star - y_compare)
            chosen_i = np.argmin(test_statistic, axis=0)

            for chosen_ii in chosen_i:
                chosen.append(data_all_used_indices[chosen_ii])
            for chosen_coords_ii in coords_used_indices:
                chosen_coords.append(chosen_coords_ii)

            ## data_chosen_by_extension = np.where(ext_all == ext_name)
            ## x_compare, x = np.meshgrid(
            ##     coords_image[:, 0][coords[:, 2] == ext_num],
            ##     data_all['XWIN_IMAGE'][ext_all == ext_name])
            ## y_compare, y = np.meshgrid(coords_image[:, 1][
            ##     coords[:, 2] == ext_num],
            ##     data_all['YWIN_IMAGE'][ext_all == ext_name])
            ## TS = np.square(x - x_compare) + np.square(y - y_compare)

            ## chosen_i = np.argmin(TS, axis=0)

            ## chosen_i = np.where(TS == TS.min(axis=0))
            ## '''
            ## chosen_i[0] gives the arrays in data that are the minimum
            ## chosen_i[1] gives their corresponding coords in coords_image;
            ## so unless you make your final list be in the order of chosen,
            ## they will be "out of order" vis-a-vis the coords file
            ## ... which probably isn't that big a deal since we aren't
            ## tracking those too heavily
            ## '''
            ## # assume that our slices are blocked by their extension
            ## for i in chosen_i[0].tolist():
            ##     chosen.append(i +
            ##                   data_chosen_by_extension[0][0])
        coords_reordered = coords[chosen_coords]
        data_combined = data_all[chosen]
        ext_combined = np.array([self.decaminfo.ccddict[int(ext_num)]
                                for ext_num in coords_reordered[:, 2]])

        # convert window positions and extension numbers to mm positions
        x_all, y_all = [[], []]
        for i in range(len(ext_combined)):
            x_all_i, y_all_i = self.decaminfo.getPosition(
                ext_combined[i],
                data_combined['XWIN_IMAGE'][i],
                data_combined['YWIN_IMAGE'][i])
            x_all.append(x_all_i)
            y_all.append(y_all_i)
        x_all = np.array(x_all, dtype=np.float64)
        y_all = np.array(y_all, dtype=np.float64)

        # now make the dictionary
        try:
            comparison_dict = dict(
                x=x_all, y=y_all,
                x2=data_combined['X2WIN_IMAGE'].astype(np.float64),
                y2=data_combined['Y2WIN_IMAGE'].astype(np.float64),
                xy=data_combined['XYWIN_IMAGE'].astype(np.float64),
                fwhm=data_combined['FWHM_WORLD'].astype(np.float64) * 3600,
                e1=(data_combined['X2WIN_IMAGE'].astype(np.float64) -
                    data_combined['Y2WIN_IMAGE'].astype(np.float64)) *
                    0.27**2,
                e2=2 * data_combined['XYWIN_IMAGE'].astype(np.float64) *
                    0.27**2,
                e0=(data_combined['X2WIN_IMAGE'].astype(np.float64) +
                    data_combined['Y2WIN_IMAGE'].astype(np.float64)) *
                    0.27**2,)
        except ValueError:
            # probably doesn't have the win values, so try unwindowed
            comparison_dict = dict(
                x=x_all, y=y_all,
                x2=data_combined['X2_IMAGE'].astype(np.float64),
                y2=data_combined['Y2_IMAGE'].astype(np.float64),
                xy=data_combined['XY_IMAGE'].astype(np.float64),
                fwhm=data_combined['FWHM_WORLD'].astype(np.float64) * 3600,
                e1=(data_combined['X2_IMAGE'].astype(np.float64) -
                    data_combined['Y2_IMAGE'].astype(np.float64)) *
                    0.27**2,
                e2=2 * data_combined['XY_IMAGE'].astype(np.float64) *
                    0.27**2,
                e0=(data_combined['X2_IMAGE'].astype(np.float64) +
                    data_combined['Y2_IMAGE'].astype(np.float64)) *
                    0.27**2,)

        return comparison_dict, data_combined, ext_combined, coords_reordered
