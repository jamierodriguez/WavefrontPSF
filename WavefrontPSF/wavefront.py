#!/usr/bin/env python
"""
File: wavefront.py
Author: Chris Davis
Description: Module for generating PSF objects and their moments.

"""

from __future__ import print_function, division
import numpy as np
from donutlib.makedonut import makedonut
from adaptive_moments import adaptive_moments, centered_moment
from os import path, makedirs
import pickle
from routines_moments import convert_moments

class Wavefront(object):
    """Class with the ability to generate stars as well as their images.  Given
    a list of positions and zernikes, generates a set of moments (to your
    specification).

    Attributes
    ----------
    number_electrons
        the number of Electrons used in making the stamp (ie the normalization)

    background
        the background value for each stamp

    input_dict
        a dictionary containing the parameters for aaron's makedonut routine

    make_donut
        aaron's makedonut object

    Methods
    -------
    save
        save this whole object, sans make_donut. Should inherit OK

    stamp
        takes the zernike polynomials and creates a 2d image array

    moments
        given a stamp, calculates the moments

    moment_dictionary
        given a list of zernikes, get all their moments plus useful linear
        combinations (e.g. ellipticity)

    """

    def __init__(self, number_electrons=1e6, background=4000, randomFlag=0,
                 nbin=256, nPixels=32, pixelOverSample=8, scaleFactor=1.,
                 **args):

        self.number_electrons = number_electrons
        self.background = background
        if nPixels == 32:
            self.input_dict = {
                "nbin": 256,  # 128,
                "nPixels": 32,  # 32,
                "pixelOverSample": 8,  # 4,
                "scaleFactor": 1.,  # 2.,
                "randomFlag": randomFlag,
                }
        elif nPixels == 16:
            self.input_dict = {
                "nbin": 64,  # 256,  # 128,
                "nPixels": 16,  # 32,  # 32,
                "pixelOverSample": 4,  # 8,  # 4,
                "scaleFactor": 2.,  # 1.,  # 2.,
                "randomFlag": randomFlag,
                }
        self.make_donut = makedonut(**self.input_dict)

    def save(self, out_path):
        """Take the data and save it!

        Parameters
        ----------
        out_path : string
            The location where we will dump the pickle.

        Notes
        -----
        in order to save as a pickleable object, I need to set make_donut
        (which is a pyswig object) to none. So when you reload this object, it
        can have everything else /except/ the make_donut property.

        """

        if not path.exists(path.dirname(out_path)):
            makedirs(path.dirname(out_path))
        self.make_donut = None  # this is the offender!
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)
        # give FP back its make_donut
        self.make_donut = makedonut(**self.input_dict)

    def remakedonut(self):
        """Remake make_donut

        Notes
        -----
        No input or return.

        """
        self.make_donut = makedonut(**self.input_dict)
        return

    def stamp(self, zernike, rzero, coord):
        """Create a stamp from list of zernike parameters

        Parameters
        ----------
        zernike : list
            The coefficients to the zernike polynomials used.

        rzero : float
            Kolmogorov spectrum parameter. Basically, larger means smaller PSF.

        coord : list
            [x_decam, y_decam] in mm and Aaron's coordinate convention.

        Returns
        -------
        stamp : array
            The image of the zernike polynomial convolved with Kolmogorov
            spectrum.

        """

        x_decam, y_decam = coord

        stamp = self.make_donut.make(inputZernikeArray=zernike,
                                     rzero=rzero,
                                     nEle=self.number_electrons,
                                     background=self.background,
                                     xDECam=x_decam,
                                     yDECam=y_decam).astype(np.float64)

        return stamp

    def stamp_factory(self, zernikes, rzeros, coords):
        """Make lots of stamps

        Parameters
        ----------
        zernikes : list of lists
            Each entry in the list corresponds to a coordinate point and
            contains some number of zernike polynomial coefficients to be used
            in generating the stamp.

        coords : list of lists
            Each entry has the coordinates in [X mm, Y mm, Sensor], with the x
            and y in aaron's coordinate convention.

        rzeros : list of floats
            Kolmogorov spectrum parameter. Basically, larger means smaller PSF.

        Returns
        -------
        stamps : list of array
            The resultant stamps

        """
        stamps = []
        for i in range(len(coords)):
            coord = coords[i]
            zernike = zernikes[i]
            rzero = rzeros[i]
            # make stamp
            stamp_i = self.stamp(zernike=zernike,
                                 rzero=rzero,
                                 coord=coord[:2])
            stamps.append(stamp_i)

        stamps = np.array(stamps)

        return stamps

    def moments(self, data,
                background=0, threshold=-1e9,
                order_dict={}):
        """Given a stamp, create moments

        Parameters
        ----------
        data : array
            2d image array

        threshhold : float, optional
            Threshold value in the data array for pixels to consider in this
            fit. If not specified, then takes all data.

        background : float, optional
            Background level to be subtracted out. If not specified, set to
            zero.

        windowed : bool, optional ; Depreciated
            Decide whether to use a gaussian window.
            Default to True

        order_dict : dictionary, optional
            Gives a list of the x and y orders of the moments we want to
            calculate.
            Defaults to x2 y2 and xy moments.
            Ex:            {'x2': {'p': 2, 'q': 0},
                            'y2': {'p': 0, 'q': 2},
                            'xy': {'p': 1, 'q': 1}}

        Returns
        -------
        return_dict : dictionary
            Returns a dictionary of the calculated moments, centroid, fwhm.

        """
        stamp = (data - background)
        conds = (stamp > threshold)
        stamp = np.where(conds, stamp, 0)  # mask noise

        # get moment matrix
        Mx, My, Mxx, Mxy, Myy, A, rho4, \
            x2, xy, y2, x3, x2y, xy2, y3 \
            = adaptive_moments(stamp)

        fwhm = np.sqrt(np.sqrt(Mxx * Myy - Mxy * Mxy))
        whisker = np.sqrt(np.sqrt(Mxy * Mxy + 0.25 * np.square(Mxx - Myy)))
        # 2 (1 + a4) = rho4
        a4 = 0.5 * rho4 - 1
        # update return_dict
        return_dict = {
                'Mx': Mx, 'My': My,
                'Mxx': Mxx, 'Mxy': Mxy, 'Myy': Myy,
                'fwhm': fwhm, 'flux': A, 'a4': a4, 'whisker': whisker,
                'x2': x2, 'xy': xy, 'y2': y2,
                'x3': x3, 'x2y': x2y, 'xy2': xy2, 'y3': y3,
                }

        # now go through moment_dict and create any other moments
        for order in order_dict:
            pq = order_dict[order]
            p = pq['p']
            q = pq['q']
            return_dict.update({order: centered_moment(stamp,
                                                       Mx=Mx, My=My,
                                                       p=p, q=q,
                                                       Mxx=Mxx, Mxy=Mxy,
                                                       Myy=Myy
                                                       )})
        return return_dict


    def moment_dictionary(
            self, stamps, coords,
            backgrounds=[], thresholds=[],
            verbosity=[], windowed=True,
            order_dict={}):

        """create a bunch of

        Parameters
        ----------
        stamps : list of arrays
            All the stamps we shall analyze

        coords : list of lists
            Each entry has the coordinates in [X mm, Y mm, Sensor], with the x
            and y in aaron's coordinate convention.

        threshhold : list of floats, optional
            Threshold value in the data array for pixels to consider in this
            fit. If not specified, then takes all data.

        background : list of floats, optional
            Background level to be subtracted out. If not specified, set to
            zero.

        verbosity : list, optional
            If 'stamp' is in verbosity, then the stamps are also saved.
            Default is that stamps are not saved.

        windowed : bool, optional
            Decide whether to use a gaussian window.
            Default to True

        order_dict : dictionary, optional
            Gives a list of the x and y orders of the moments we want to
            calculate.
            Defaults to x2 y2 and xy moments.

        Returns
        -------
        return_dict : dictionary
            A dictionary with the moments, xDECam, yDECam, fwhm, and zernikes.

        Notes
        -----
        By default the moment dictionary will have fwhm and the xDECam and
        yDECam positions in them, and stamp for verbosity dictionary with
        'stamp', as well as every entry in order_dict

        """

        # create return_dict
        return_dict = dict(x=[], y=[])
                           #zernikes=zernikes)
        if 'stamp' in verbosity:
            return_dict.update(dict(stamp=[]))
        # add terms from order_dict
        for order in order_dict:
            return_dict.update({order: []})

        for i in range(len(coords)):
            coord = coords[i]
            background = backgrounds[i]
            threshold = thresholds[i]
            stamp = stamps[i]
            # get moments
            moment_dict = self.moments(
                stamp,
                background=background, threshold=threshold,
                order_dict=order_dict)

            if i == 0:
                for key in moment_dict.keys():
                    return_dict.update({key: []})
            for key in moment_dict.keys():
                return_dict[key].append(moment_dict[key])

            # append to big list
            return_dict['x'].append(coord[0])
            return_dict['y'].append(coord[1])

            # append the stamp if verbosity is high enough
            if 'stamp' in verbosity:
                return_dict['stamp'].append(stamp)

        # turn all these lists into arrays
        for key in return_dict:
            entry = return_dict[key]
            if type(entry) == list:
                return_dict.update({key: np.array(entry)})

        return_dict = convert_moments(return_dict)

        return return_dict
