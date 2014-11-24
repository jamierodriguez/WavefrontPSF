#!/usr/bin/env python
"""
File: wavefront.py
Author: Chris Davis
Description: Module for generating PSF objects and their moments.

TODO: Add ability to apply shear to generated images.
"""

from __future__ import print_function, division
import numpy as np
from donutlib.makedonut import makedonut
from adaptive_moments import adaptive_moments, centered_moment
from os import path, makedirs
from scipy.ndimage.interpolation import affine_transform

import pickle
from routines_moments import convert_moments
from decamutil_cpd import decaminfo

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
        else:
            self.input_dict = {
                "nbin": nbin,  # 256,  # 128,
                "nPixels": nPixels,  # 32,  # 32,
                "pixelOverSample": pixelOverSample,  # 8,  # 4,
                "scaleFactor": scaleFactor,  # 1.,  # 2.,
                "randomFlag": randomFlag,
                }

        self.make_donut = makedonut(**self.input_dict)
        # decaminfo stuff (since it's useful)
        self.decaminfo = decaminfo()

    def edges(self, boxdiv):
        """Convenience wrapper for decaminfo.getEdges
        """

        edges = self.decaminfo.getEdges(boxdiv)
        return edges

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

    def stamp(self, zernike, rzero, coord, jitter={}):
        """Create a stamp from list of zernike parameters

        Parameters
        ----------
        zernike : list
            The coefficients to the zernike polynomials used.

        rzero : float
            Kolmogorov spectrum parameter. Basically, larger means smaller PSF.

        coord : list
            [x_decam, y_decam] in mm and Aaron's coordinate convention.

        jitter : dictionary
            Jitter terms to apply to image.

        Returns
        -------
        stamp : array
            The image of the zernike polynomial convolved with Kolmogorov
            spectrum.

        TODO
        ----

        If shearing, consider over sampling?

        """

        x_decam = coord[0]
        y_decam = coord[1]

        stamp = self.make_donut.make(inputZernikeArray=zernike,
                                     rzero=rzero,
                                     nEle=self.number_electrons,
                                     background=self.background,
                                     xDECam=x_decam,
                                     yDECam=y_decam).astype(np.float64)

        if len(jitter.keys()) > 0:
            stamp = self.jitter(stamp, jitter)

        return stamp

    def jitter(self, data, jitter={},
               Mx = 15.937499,
               My = 15.937499):
        """Apply jitter

        Parameters
        ----------

        data : array
            2d array of image

        jitter : dictionary
            dictionary of various ellipticity or shear terms (e.g. e1, g1,
            delta2)

        Mx, My : floats
            The centers of the object

        Returns
        -------

        data_jittered : array
            2d array of image post-jitter


        Notes
        -----

        Currently only does NORMALIZED ellip e1 and e2!!
        Largely taken from CppShear.cpp in GalSim

        """

        if ('e1' in jitter.keys()) * ('e2' in jitter.keys()):

            try:
                e1 = jitter['e1']
            except KeyError:
                # no e1
                e1 = 0
            try:
                e2 = jitter['e2']
            except KeyError:
                e2 = 0

            esq = e1 * e1 + e2 * e2

            if esq < 1e-8:
                A = 1 + esq / 8 + e1 * (0.5 + esq * 3 / 16)
                B = 1 + esq / 8 - e1 * (0.5 + esq * 3 / 16)
                C = e2 * (0.5 + esq * 3 / 16)
            else:
                temp = np.sqrt(1 - esq)
                cc = np.sqrt(0.5 * (1 + 1 / temp))
                temp = cc * (1 - temp) / esq
                C = temp * e2
                temp *= e1
                A = cc + temp
                B = cc - temp
            matrix = np.array([[A, C], [C, B]])
            matrix /= A * B - C * C
            offset = (-Mx * (A + C - 1),
                      -My * (C + B - 1))

            # apply transformation
            data = affine_transform(data, matrix, offset)

        return data


    def stamp_factory(self, zernikes, rzeros, coords, jitters=[]):
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

        jitter : list of dictionaries
            Jitter terms to apply to image.


        Returns
        -------
        stamps : list of array
            The resultant stamps

        """
        stamps = []
        for i in xrange(len(coords)):
            coord = coords[i]
            zernike = zernikes[i]
            rzero = rzeros[i]
            if len(jitters) > 0:
                jitter = jitters[i]
            else:
                jitter = {}
            # make stamp
            stamp_i = self.stamp(zernike=zernike,
                                 rzero=rzero,
                                 coord=coord,
                                 jitter=jitter)
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
            attribute background.

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

        if 'stamp' in verbosity:
            return_dict.update(dict(stamp=[]))
        if 'zernikes' in verbosity:
            return_dict.update(dict(zernikes=zernikes))
        # add terms from order_dict
        for order in order_dict:
            return_dict.update({order: []})

        for i in range(len(coords)):
            coord = coords[i]
            if len(backgrounds) > 0:
                background = backgrounds[i]
            else:
                background = self.background
            if len(thresholds) > 0:
                threshold = thresholds[i]
            else:
                threshold = 1e-9
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

    def position_to_pixel(self, coords):
        """Go from focal coordinates to pixel coordinates

        Paramters
        ---------
        coords : array
            Each entry has the coordinates in [X mm, Y mm, Sensor], with the x
            and y in aaron's coordinate convention.

        Returns
        -------
        coords_pixel: array
            Each entry is now [ix, iy, Sensor]

        """

        ix, iy = self.decaminfo.getPixel_extnum(coords[:,2].astype(np.int),
                                                coords[:,0],
                                                coords[:,1])
        coords_pixel = np.vstack((ix, iy, coords[:,2])).T

        return coords_pixel

    def pixel_to_position(self, coords):

        """Go from focal coordinates to pixel coordinates

        Paramters
        ---------
        coords : array
            Each entry has the coordinates in [ix, iy, Sensor].

        Returns
        -------
        coords_focal: array
            Each entry is now [X mm, Y mm, Sensor],

        """

        xPos, yPos = self.decaminfo.getPosition_extnum(coords[:,2].astype(
            np.int),
                                                       coords[:,0],
                                                       coords[:,1])
        coords_focal = np.vstack((xPos, yPos, coords[:,2])).T

        return coords_focal

    def plot(self, x='x', y='y', z='e0', bins=25, ax=None,
            vmax=None, vmin=None):
        """Create plot

        Parameters
        ----------
        x, y, z : strings or arrays
            If strings, goes to self.data[x]
            else, use them directly.

        bins : int
            If less than 10, uses decaminfo.getEdges
            Otherwise is the number of bins to use.

        Returns
        -------
        fig, ax : figure and axis
            The plot!

        Notes
        -----
        stuff

        See Also
        --------
        other function: brief description

        References
        ----------
        stuff

        Examples
        --------
        >>> example code where
        ...     it goes over multiple lines

        for more doc help, see https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
        """

        from routines_plot import focal_graph, focal_graph_axis
        from colors import blue_red, shiftedColorMap

        if type(x) == str:
            x = self.data[x]
        if type(y) == str:
            y = self.data[y]
        if type(z) == str:
            z = self.data[param]

        if bins < 10:
            bins = self.decaminfo.getEdges(boxdiv=bins)

        counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
        weighted_counts, xedges, yedges = np.histogram2d(x, y,
                bins=[xedges, yedges], weights=z)
        C = np.ma.masked_invalid(weighted_counts.T / counts.T)
        if not vmax:
            vmax = C.max()
        if not vmin:
            vmin = C.min()

        if np.ma.all(C <= 0):
            cmap = plt.get_cmap('Blues_r')
        elif np.ma.all(C >= 0):
            cmap = plt.get_cmap('Reds')
        else:
            midpoint = 1 - vmax/(vmax + abs(vmin))
            cmap = shiftedColorMap(blue_red, midpoint=midpoint, name='shifted')

        return_fig = False
        if ax:
            ax = focal_graph_axis(ax)
        else:
            fig, ax = focal_graph()
            return_fig = True
        Image = ax.pcolor(xedges, yedges, C,
                          cmap=cmap, vmin=vmin, vmax=vmax)
        CB = fig.colorbar(Image, ax=ax)

        if return_fig:
            return fig, ax
        else:
            return ax
