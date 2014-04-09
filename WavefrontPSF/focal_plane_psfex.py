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
        from a dictionary of zernike corrections and coordinates, generate a
        focal plane of moments. if the 'stamp' is in verbosity,
        the stamps of the stars are also saved.


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

        psfex = {}
        uniques = np.unique(list_chip)
        for unique in uniques:
            psfex.update({unique: []})

        for i in xrange(len(list_catalogs)):
            temp = PSFEx(list_catalogs[i])
            psfex[list_chip[i]].append(temp)

        return psfex

    def interpolate(self, ix, iy, ext):

        # select the extnum of psfex items
        psfex_list = self.psfex[ext]
        n_psfex = len(psfex_list)

        image = 0
        for psfex in psfex_list:
            # TODO: is this true: row is ix and col is iy?
            image += psfex.get_rec(ix, iy)

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
