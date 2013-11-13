#!/usr/bin/env python
# wavefront.py
from __future__ import print_function, division
import numpy as np
from donutlib.makedonut import makedonut
from moment_calc import windowed_centroid, FWHM, centered_moment, \
    gaussian_window
from os import path, makedirs
import pickle

"""
Module for generating stars as well as their moments.
"""


class Wavefront(object):
    """Class with the ability to generate stars as well as their images.  Given
    a list of positions and zernikes, generates a set of moments (to your
    specification).

    Attributes
    ----------
    nEle
        the number of Electrons used in making the stamp (ie the normalization)

    background
        the background value for each stamp

    input_dict
        a dictionary containing the parameters for aaron's makedonut routine

    md
        aaron's makedonut object

    Methods
    -------
    save
        save this whole object, sans md. Should inherit OK -- I ought to double
        check that #TODO.

    stamp
        takes the zernike polynomials and creates a 2d image array

    moments
        given a stamp, calculates the moments

    moment_dictionary
        given a list of zernikes, get all their moments plus useful linear
        combinations (e.g. ellipticity)

    """

    def __init__(self, nEle=1e6, background=1000, randomFlag=0,
                 nbin=64, nPixels=16, pixelOverSample=4, scaleFactor=2.):

        self.nEle = nEle
        self.background = background
        self.input_dict = {
            "nbin": nbin,  # 64,  # 256,  # 128,
            "nPixels": nPixels,  # 16,  # 32,  # 32,
            "pixelOverSample": pixelOverSample,  # 4,  # 8,  # 4,
            "scaleFactor": scaleFactor,  # 2.,  # 1.,  # 2.,
            "randomFlag": randomFlag,
            }
        self.md = makedonut(**self.input_dict)

    def save(self, out_path):
        """Take the data and save it!

        Parameters
        ----------
        out_path : string
            The location where we will dump the pickle.

        Notes
        -----
        in order to save as a pickleable object, I need to set md (which is a
        pyswig object) to none. So when you reload this object, it can have
        everything else /except/ the md property.
        """

        if not path.exists(path.dirname(out_path)):
            makedirs(path.dirname(out_path))
        self.md = None  # this is the offender!
        with open(out_path, 'wb') as f:
            pickle.dump(self, f)
        # give FP back its md
        self.md = makedonut(**self.input_dict)

    def stamp(self, zernike, rzero, coord):
        """Create a stamp from list of zernike parameters

        Parameters
        ----------
        zernike : list
            The coefficients to the zernike polynomials used.

        rzero : float
            Kolmogorov spectrum parameter. Basically, larger means smaller PSF.

        coord : list
            [xDECam, yDECam] in mm and Aaron's coordinate convention.

        Returns
        -------
        stamp : array
            The image of the zernike polynomial convolved with Kolmogorov
            spectrum.

        """

        xDECam, yDECam = coord

        stamp = self.md.make(inputZernikeArray=zernike,
                             rzero=rzero,
                             nEle=self.nEle,
                             background=self.background,
                             xDECam=xDECam,
                             yDECam=yDECam).astype(np.float64)

        return stamp

    def moments(self, stamp, indices=None, windowed=True,
                background=-1, thresh=-1,
                order_dict={'x2': {'p': 2, 'q': 0},
                            'y2': {'p': 0, 'q': 2},
                            'xy': {'p': 1, 'q': 1}}):
        """Given a stamp, create moments

        Parameters
        ----------
        stamp : array
            2d image array

        indices : length 2 list, optional
            length 2 list of 2d arrays Y and X such that Y and X indicate the
            index values (ie indices[0][i,j] = i, indices[1][i,j] = j) Default
            is None; constructs the indices with each call.

        background : float, optional
            Background data value. This is subtracted out.
            If not given, estimate by an annulus around the edge of the image.

        thresh : float, optional
            Threshold value in the data array for pixels to consider in this
            fit.  If no value is specified, then it is set to be max(data) / 5
            .

        windowed : bool, optional
            Decide whether to use a gaussian window.

        order_dict : dictionary, optional
            Gives a list of the x and y orders of the moments we want to
            calculate.

        Returns
        -------
        return_dict : dictionary
            Returns a dictionary of the calculated moments, centroid, fwhm.

        """

        if background == -1:
            background = self.background
        # get windowed centroid
        y, x = windowed_centroid(stamp, indices=indices,
                                 background=background, thresh=thresh)
        # get fwhm
        fwhm = FWHM(stamp, centroid=[y, x], indices=indices,
                    background=background, thresh=thresh)
        # get weight for other windowed moments
        if not windowed:
            w = 1
        else:
            w = gaussian_window(stamp,
                                centroid=[y, x], indices=indices,
                                background=background, thresh=thresh)

        return_dict = dict(x=x, y=y, fwhm=fwhm)

        # now go through moment_dict and create the other moments
        for order in order_dict:
            pq = order_dict[order]
            return_dict.update({order: centered_moment(stamp, w=w,
                                                       centroid=[y, x],
                                                       indices=indices,
                                                       **pq)})
        return return_dict

    def moment_dictionary(
            self, zernikes, coords, rzero,
            verbosity=0, windowed=True,
            order_dict={'x2': {'p': 2, 'q': 0},
                        'y2': {'p': 0, 'q': 2},
                        'xy': {'p': 1, 'q': 1}}):

        """create a bunch of

        Parameters
        ----------
        TODO: fill me in

        Returns
        -------
        return_dict : dictionary
            A dictionary with the moments, xDECam, yDECam, fwhm, and zernikes.

        Notes
        -----
        By default the moment dictionary will have fwhm and the xDECam and
        yDECam positions in them, and stamp for high verbosity, as
        well as every entry in order_dict

        """

        # temporary fix for a list creation problem
        Yi, Xi = np.indices((self.input_dict["nPixels"],
                             self.input_dict["nPixels"]))
        # create return_dict
        return_dict = dict(x=[], y=[], fwhm=[],
                           zernikes=zernikes)
        if verbosity >= 4:
            return_dict.update(dict(stamp=[]))
        # add terms from order_dict
        for order in order_dict:
            return_dict.update({order: []})

        for i in range(len(coords)):
            coord = coords[i]
            zernike = zernikes[i]
            # make stamp
            stamp_i = self.stamp(zernike=zernike,
                                 rzero=rzero,
                                 coord=coord[:2])
            # get moments
            moment_dict = self.moments(
                stamp_i, indices=[Yi, Xi],
                windowed=windowed, order_dict=order_dict)
            fwhm_i = moment_dict['fwhm']

            # append to big list
            return_dict['x'].append(coord[0])
            return_dict['y'].append(coord[1])

            # append the arcsecond things
            return_dict['fwhm'].append(fwhm_i * 0.27)

            # append the stamp if verbosity is high enough
            if verbosity >= 4:
                return_dict['stamp'].append(stamp_i)

            # now append things from order_dict
            for order in order_dict.keys():
                return_dict[order].append(moment_dict[order])

        # turn all these lists into arrays
        for key in return_dict:
            entry = return_dict[key]
            if type(entry) == list:
                return_dict.update({key: np.array(entry)})

        return return_dict
