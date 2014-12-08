#!/usr/bin/env python
"""
File: focal_plane_fit.py
Author: Chris Davis
Description: Class for creating wavefronts on a generic focal plane.
"""

from __future__ import print_function, division
import numpy as np
from wavefront import Wavefront
from routines import average_dictionary
from routines_moments import convert_moments

from psfex import PSFEx

class FocalPlanePSFEx(Wavefront):
    """Wavefront object that now has coordinates and the ability to generate an
    entire wavefront.

    Attributes
    ----------
    da
        aaron's donutana object

    verbosity
        a list with strings indicating what should be saved.
        currently only implements 'stamp' for saving the stamps

    decaminfo
        a decamutil class object with useful routines relating the DECam focal
        plane pixels to positions


    Methods
    -------
    plane
        from coordinates, generate a focal plane of moments. if the 'stamp' is
        in verbosity, the stamps of the stars are also saved.


    """

    def __init__(self,
                 list_catalogs, list_chip,
                 verbosity=['history'],
                 **args):

        # do the old init for Wavefront
        super(FocalPlanePSFEx, self).__init__(**args)

        self.list_catalogs = list_catalogs
        self.list_chip = list_chip
        self.psfex = self.generate_psfex(list_catalogs, list_chip)

        self.background = 0
        self.thresholds = 0
        self.verbosity = verbosity

    def generate_psfex(self, list_catalogs, list_chip):
        """Generate the psfex dictionary delineated by chip

        Parameters
        ----------
        list_catalogs : list
            Locations of all the PSFEx objects we are interested in.

        list_chip : list
            How we group the PSFEx objects

        Returns
        -------
        psfex : dictionary
            A dictionary of the PSFEx objects delineated by list_chip
            (typically the chip name or number). Each entry is a list of all
            the PSFEx objects with that value.

        """


        psfex = {}
        uniques = np.unique(list_chip)
        for unique in uniques:
            psfex.update({unique: []})

        for i in xrange(len(list_catalogs)):
            temp = PSFEx(list_catalogs[i])
            psfex[list_chip[i]].append(temp)

        return psfex

    def interpolate(self, ix, iy, ext):
        """Create psfex image

        Parameters
        ----------
        ix, iy : floats
            Pixel coordinates we would like to sample at the center of.

        ext : int
            Name of the psfex extension (usually chip name or integer) we
            should like to interpolate from.

        Returns
        -------
        image : array
            2d array of the psfex image.

        """
        # select the extnum of psfex items
        ext_name = self.decaminfo.ccddict[ext]
        psfex_list = self.psfex[ext_name]
        n_psfex = len(psfex_list)

        image = 0
        for psfex in psfex_list:
            # TODO: is this true: row is ix and col is iy?
            # cpd 19.4.14: flipped ix to col and iy to row
            image += psfex.get_rec(iy, ix)

        # if you have multiple images, average each one
        image /= n_psfex

        return image

    def interpolate_factory(self, coords):
        """Interpolate over psfex and get the images

        Parameters
        ----------
        coords : array
            An array of [[coordx, coordy, ext_num]] of the locations sampled

        Returns
        -------
        stamps : array of 2d arrays
            The psfex stamps!

        """

        # convert coords to coords_pixel
        coords_pixel = self.position_to_pixel(coords)

        # make stamps
        stamps = []
        for coord in coords_pixel:
            stamp_i = self.interpolate(coord[0], coord[1], coord[2])
            stamps.append(stamp_i)
        stamps = np.array(stamps)

        return stamps

    def plane(self, coords,
              windowed=True, order_dict={}):
        """create a wavefront across the focal plane

        Parameters
        ----------
        coords : array
            An array of [[coordx, coordy, ext_num]] of the locations sampled

        windowed : bool, optional ; depreciated
            Do we calculate the windowed moments, or unwindowed? Default true.

        order_dict : dictionary, optional
            A dictionary of dictionaries indicating the name and the powers of
            the moments calculated.
            Default calculates the second and third moments.

        Returns
        -------
        moments : dictionary
            A dictionary containing all the moments calculated and
            corresponding convenient linear combinations (ellipticities, etc).

        """

        N = len(coords)
        backgrounds = [self.background] * N
        thresholds = [0] * N

        stamps = self.interpolate_factory(coords)

        # make moments
        moments = self.moment_dictionary(stamps,
                                         coords,
                                         backgrounds=backgrounds,
                                         thresholds=thresholds,
                                         verbosity=self.verbosity,
                                         windowed=windowed,
                                         order_dict=order_dict)

        moments = convert_moments(moments)

        return moments

    def plane_averaged(
            self, coords, average=np.mean, boxdiv=0, subav=False,
            windowed=True, order_dict={}):
        """create a wavefront across the focal plane and average into boxes

        Parameters
        ----------
        coords : array
            An array of [[coordx, coordy, ext_num]] of the locations sampled

        windowed : bool, optional ; depreciated
            Do we calculate the windowed moments, or unwindowed? Default true.

        order_dict : dictionary, optional
            A dictionary of dictionaries indicating the name and the powers of
            the moments calculated.
            Default calculates the second and third moments.

        average : function
            Function used for averaging

        boxdiv : int
            Sets the divisions of the chip that we average over.

        subav : bool
            True subtracts the mean when averaging


        Returns
        -------
        moments : dictionary
            Dictionary with the averaged moments and the variance

        """

        # get the moments
        moments_unaveraged = self.plane(coords, windowed=windowed,
                                        order_dict=order_dict)

        # now average
        moments = average_dictionary(moments_unaveraged, average,
                                     boxdiv=boxdiv, subav=subav)

        return moments

