#!/usr/bin/env python
"""
File: focal_plane.py
Author: Chris Davis
Description: Class for focal plane shell tied to specific image.

"""

from __future__ import print_function, division
import numpy as np
from wavefront import Wavefront
import pyfits
from os import path
from routines_files import combine_decam_catalogs, generate_hdu_lists
from routines import average_dictionary, mean_trim
from routines_moments import convert_moments

class FocalPlane(Wavefront):
    """Wavefront tied to a specific image. Comparisons and such
    possible.

    Attributes
    ----------
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

    """

    def __init__(self,
                 list_catalogs, list_fits_extension=None, list_chip=None,
                 max_samples_box=10000, boxdiv=0, subav=0,
                 conds='default', average=mean_trim,
                 coord_name='WIN_IMAGE',
                 **args):

        # do the old init for Wavefront
        super(FocalPlane, self).__init__(**args)
        # could put in nEle etc here


        if not list_chip:
            # assume list_catalogs is expid
            if path.exists('/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/'):
                list_catalogs, list_fits_extension, list_chip = \
                    generate_hdu_lists(list_catalogs)
            elif path.exists('/Users/cpd/Desktop/Images/'):
                list_catalogs, list_fits_extension, list_chip = \
                    generate_hdu_lists(list_catalogs,
                        path_base='/Users/cpd/Desktop/Images/')
            elif path.exists('/Volumes/Seagate/Images/'):
                list_catalogs, list_fits_extension, list_chip = \
                    generate_hdu_lists(list_catalogs,
                        path_base='/Volumes/Seagate/Images/')
        elif type(list_catalogs) == list:
            if len(list_catalogs) > 0:
                if type(list_catalogs[0]) == int:
                    # we have a list of expids!
                    expids = list_catalogs
                    list_catalogs = []
                    list_fits_extension = []
                    list_chip = []
                    for expid in expid:
                        if path.exists(
                            '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/'):
                            temp = \
                                generate_hdu_lists(expid,
                                    path_base='/nfs/slac/g/ki/ki18/' +
                                              'cpd/catalogs/wgetscript/')
                        elif path.exists('/Users/cpd/Desktop/Images/'):
                            temp = \
                                generate_hdu_lists(expid,
                                    path_base='/Users/cpd/Desktop/Images/')
                        elif path.exists('/Volumes/Seagate/Images/'):
                            temp = \
                                generate_hdu_lists(expid,
                                    path_base='/Volumes/Seagate/Images/')
                        list_catalogs += temp[0]
                        list_fits_extension += temp[1]
                        list_chip += temp[2]


        self.list_catalogs = list_catalogs
        self.list_fits_extension = list_fits_extension
        self.list_chip = list_chip

        self.average = average
        self.boxdiv = boxdiv
        self.conds = conds
        self.coord_name = coord_name
        self.max_samples_box = max_samples_box
        self.subav = subav

        # generate recdata
        recdata_unfiltered, \
            extension_unfiltered, self.recheader = \
            combine_decam_catalogs(
                list_catalogs, list_fits_extension, list_chip)

        conds_array = self.filter(
            recdata_unfiltered, self.conds)
        self.recdata = recdata_unfiltered[conds_array]
        self.extension = extension_unfiltered[conds_array]

        self.recdata, self.extension = \
            self.filter_number_in_box(recdata=self.recdata,
                                      extension=self.extension,
                                      max_samples_box=self.max_samples_box,
                                      boxdiv=self.boxdiv)

        self.data, self.coords, self.data_unaveraged = self.create_data(
                recdata=self.recdata,
                extension=self.extension,
                average=self.average,
                boxdiv=self.boxdiv,
                subav=self.subav,
                )

    def filter(self, recdata, conds='default'):
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
        if conds == 'old':
            fwhm = 1.0

            conds = (
                (recdata['CLASS_STAR'] > 0.9) *
                (recdata['MAG_AUTO'] < 16) *
                (recdata['MAG_AUTO'] > 13) *
                (recdata['FWHM_WORLD'] > 0) *
                (recdata['FWHM_WORLD'] * 60 ** 2 < 2 * fwhm) *
                (recdata['FLAGS'] <= 3) *
                (recdata['X' + self.coord_name] > pixel_border) *
                (recdata['X' + self.coord_name] < 2048 - pixel_border) *
                (recdata['Y' + self.coord_name] > pixel_border) *
                (recdata['Y' + self.coord_name] < 4096 - pixel_border))
        elif conds == 'minimal':
            # eliminate base on flags and coordnames only
            conds = (
                (recdata['FLAGS'] <= 3) *
                (recdata['X' + self.coord_name] > pixel_border) *
                (recdata['X' + self.coord_name] < 2048 - pixel_border) *
                (recdata['Y' + self.coord_name] > pixel_border) *
                (recdata['Y' + self.coord_name] < 4096 - pixel_border))
        elif conds == 'all':
            # take everything
            conds = np.array([True] * recdata.size)
        elif conds == 'default':
            """
            A set of cuts taken from:
            https://cdcvs.fnal.gov/redmine/projects/des-sci-verification/wiki/A_Modest_Proposal_for_Preliminary_StarGalaxy_Separation
            (FLAGS_I <=3) AND (((CLASS_STAR_I > 0.3) AND (MAG_AUTO_I < 18.0) AND (MAG_PSF_I < 30.0)) OR (((SPREAD_MODEL_I + 3*SPREADERR_MODEL_I) < 0.003) AND ((SPREAD_MODEL_I +3*SPREADERR_MODEL_I) > -0.003)))

            throw in an MAD cuts for FWHM, Y2, X2, and XY moments
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
                ((recdata['X' + self.coord_name] > pixel_border) *
                 (recdata['X' + self.coord_name] < 2048 - pixel_border) *
                 (recdata['Y' + self.coord_name] > pixel_border) *
                 (recdata['Y' + self.coord_name] < 4096 - pixel_border)
                )
                )

            # add an SN cut
            SN = 2.5 / np.log(10) / recdata['MAGERR_AUTO']
            conds_SN = (SN > 20) #* (SN < 100)
            conds *= conds_SN
        else:
            # evaluate the string
            conds = eval(conds)

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

        recdata_return = recdata #np.copy(recdata)
        extension_return = extension #np.copy(extension)
        # create the bounds [[x0, x1, xn], [y0, y1, y2, yn]] .  to do this,
        # realize that you only need to make one box (since we will filter by
        # pixel coordinates)

        conds_return = np.array([False] * extension_return.size)
        # find out which ones are also nans etc
        conds_finite = (
            np.isfinite(recdata_return['Y2' + self.coord_name]) *
            np.isfinite(recdata_return['X2' + self.coord_name]) *
            np.isfinite(recdata_return['XY' + self.coord_name]) *
            (recdata_return['Y2' + self.coord_name] > 0) *
            (recdata_return['X2' + self.coord_name] > 0)
            )

        # think about reordering below:
        box = self.decaminfo.getBounds_pixel(boxdiv=boxdiv)
        for x in xrange(len(box[0]) - 1):
            conds_coords_x = ((recdata_return['X' + self.coord_name] >
                               box[0][x]) *
                              (recdata_return['X' + self.coord_name] <
                               box[0][x+1]))

            for y in xrange(len(box[1]) - 1):
                conds_coords_y = ((recdata_return['Y' + self.coord_name] >
                                   box[1][y]) *
                                  (recdata_return['Y' + self.coord_name] <
                                   box[1][y+1]))

                conds_coords = conds_coords_x * conds_coords_y
                for i in range(1, 63):
                    if i == 61:
                        #n30 sucks
                        continue
                    extname = self.decaminfo.ccddict[i]

                    conds = conds_coords * (extension_return == extname)

                    # TODO: I really should never have to actually do the
                    # following, because I shouldn't be getting nan moments!

                    # conds_kill = inside chip AND not finite
                    conds_kill = conds * ~conds_finite

                    # conds_okay = inside chip AND finite
                    conds_okay = conds * conds_finite

                    # This is pretty kludgey.  We only want UP TO
                    # max_samples_box of conds_okay, so we find the N of
                    # conds_okay that need to be excluded

                    N = np.sum(conds_okay) - max_samples_box
                    if N > 0:
                        true_list = [False] * N + \
                                    [True] * (max_samples_box)
                        np.random.shuffle(true_list)
                        indices = np.nonzero(conds_okay)[0]
                        for ii in xrange(len(true_list)):
                            conds_okay[indices[ii]] = true_list[ii]
                            # in effect, conds_okay now becomes the opposite of
                            # those that will be excluded via max_samples

                    conds_final = conds_okay

                    conds_return += conds_final


        recdata_return = recdata_return[conds_return]
        extension_return = extension_return[conds_return]

        return recdata_return, extension_return

    def create_data_unaveraged(self, recdata, extension):
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
            recdata['X' + self.coord_name][i],
            recdata['Y' + self.coord_name][i])
                            for i in xrange(len(extension))])
        extNumbers = [self.decaminfo.infoDict[i]['CCDNUM']
                      for i in extension]
        coords = np.append(coords.T, [extNumbers], axis=0).T

        moments_unaveraged = dict(
                x=coords[:, 0], y=coords[:, 1],
                x2=recdata['X2' + self.coord_name].astype(np.float64),
                y2=recdata['Y2' + self.coord_name].astype(np.float64),
                xy=recdata['XY' + self.coord_name].astype(np.float64),
                flux_radius=recdata['FLUX_RADIUS'].astype(np.float64),)
        if 'FWHM_WORLD' in recdata.dtype.names:
            moments_unaveraged.update(dict(
                fwhm=recdata['FWHM_WORLD'].astype(np.float64) * 3600,
                ))
        if 'FWHM_IMAGE' in recdata.dtype.names:
            moments_unaveraged.update(dict(
                fwhm=recdata['FWHM_IMAGE'].astype(np.float64),
                ))

        if ('X3' + self.coord_name in recdata.dtype.names):
            moments_unaveraged.update(dict(
                    x3=recdata['X3' + self.coord_name].astype(np.float64),
                    x2y=recdata['X2Y' + self.coord_name].astype(np.float64),
                    xy2=recdata['XY2' + self.coord_name].astype(np.float64),
                    y3=recdata['Y3' + self.coord_name].astype(np.float64),
                    a4=recdata['A4_ADAPTIVE'].astype(np.float64),
                    flux=recdata['FLUX_ADAPTIVE'].astype(np.float64),
                    sn_flux=recdata['SN_FLUX'].astype(np.float64),
                    chip=recdata['CHIP'].astype(np.int),
                    ))

        moments_unaveraged = convert_moments(moments_unaveraged)

        return moments_unaveraged, coords

    def create_data(self, recdata, extension, average, boxdiv, subav):
        """Create the data attribute

        Parameters
        ----------
        recdata : recarray
            recarray with all the recdata

        extension : list
            Array of all the extension names

        average : function
            Function used to average data

        boxdiv : int
            Controls chip divisions

        subav : int
            Subtract the mean value of the entire focal plane?

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

        moments_unaveraged, coords = self.create_data_unaveraged(recdata,
                                                                 extension)

        # now average
        moments = average_dictionary(moments_unaveraged, average,
                                     boxdiv=boxdiv, subav=subav)

        return moments, coords, moments_unaveraged

    def __add__(self, other):
        other_keys = other.data.keys()
        # pop the keys that are not in self:
        self_keys = self.data.keys()
        keys = [key for key in other_keys if key in self_keys]

        return_dict = {}
        for key in keys:
            return_dict.update({key: np.append(self.data[key],
                                               other.data[key])})
        return return_dict

