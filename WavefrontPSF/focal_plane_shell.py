#!/usr/bin/env python
"""
File: focal_plane_shell.py
Author: Chris Davis
Description: Class for creating wavefronts on a generic focal plane.
"""

from __future__ import print_function, division
import numpy as np
from wavefront import Wavefront
from hexapodtoZernike_cpd import hexapodtoZernike
from donutana_cpd import donutana
from decamutil_cpd import decaminfo
from focal_plane_routines import fwhm_to_rzero, average_dictionary, \
    variance_dictionary, convert_moments


class FocalPlaneShell(Wavefront):
    """Wavefront object that now has coordinates and the ability to generate an
    entire wavefront.

    Attributes
    ----------
    path_mesh
        path to directory where the meshes are located

    mesh_name
        name of the specific types of meshes we want
        we have reference corrections for "Science20120915seq1_134239"
        but not for the latest "Science20130325s1v2_190406"

    da
        aaron's donutana object

    verbosity
        a list with strings indicating what should be saved.
        the only two currently implemented are 'stamp' for returning the stamp
        when making moments, and 'history' for updating the history

    reference_correction
        an array containing the reference corrections between the focus chips
        and the focal plane

    decaminfo
        a decamutil class object with useful routines relating the DECam focal
        plane pixels to positions

    history
        an object with all plane creation parameters of the plane

    Methods
    -------
    plane
        from a dictionary of zernike corrections and coordinates, generate a
        focal plane of moments. if the 'stamp' is in verbosity,
        the stamps of the stars are also saved.

    init_da

    zernike_corrections_from_dictionary
        convert dictionary corrections to the zernike corrections array

    zernike_dictionary_from_corrections
        Convert zernike corrections into a dictionary.

    zernikes
        from zernike corrections and coordinates, get a list of zernike
        polynomials

    zernike_corrections
        create a list of functions for getting the zernike polynomial
        coefficients on the focal plane

    random_coordinates
        A method for generating coordinates by sampling over boxes

    check_full_bounds
        Convenience method for checking whether my random sampling hits all
        possible divisions over the chip.

    """

    def __init__(self,
                 path_mesh='/u/ec/roodman/Astrophysics/Donuts/Meshes/',
                 mesh_name="Science20120915s1v3_134239",
                 verbosity=['history'],
                 **args):

        # do the old init for Wavefront
        super(FocalPlaneShell, self).__init__(**args)

        self.path_mesh = path_mesh
        self.mesh_name = mesh_name
        self.da = self.init_da(path_mesh=path_mesh, mesh_name=mesh_name)

        self.verbosity = verbosity
        self.history = []

        # TODO: incorporate some way of varying the mesh correction
        # NOTE: THIS DEPENDS ON THE MESH YOU CHOOSE TO USE.
        self.reference_correction = -1 * np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1690, 0.496, 6.14],
            [-0.0115, 0.00232, 0.000538],
            [-0.11, 0.000627, 0.000801],
            [0.0113, -0.000329, 9.68e-5],
            [-0.145, 0.000396, 0.00024],
            [-0.0884, 0.000365, 0.000158],
            [0.0452, 0.000159, 0.000191],
            [0, 0, 0]])

        # decaminfo stuff (since it's useful)
        self.decaminfo = decaminfo()

    def init_da(self, path_mesh, mesh_name):

        """Load up the donutana interpolation routine

        Parameters
        ----------

        path_mesh : str
            location of the mesh files

        mesh_name : str
            name of the meshes used

        Returns
        -------
        da : object
            Donutana object that we use to interpolate the zernike value on the
            focal plane.
        """

        #mesh_name = "Science20120915seq1_134239"
        sensorSet = "ScienceOnly"
        method = "idw"

        in_dict = {"zPointsFile": path_mesh + "z4Mesh_" +
                                 mesh_name + ".dat",
                  "sensorSet": sensorSet,
                  "doTrefoil": True,
                  "unVignettedOnly": False,
                  "interpMethod": method,
                  "histFlag": False,  # True,
                  "debugFlag": False,  # True,
                  "donutCutString": ""}

        for zi in range(5, 12):
            try:
                in_dict.update({'z{0}PointsFile'.format(zi):
                               path_mesh + 'z{0}Mesh_'.format(zi) +
                               mesh_name + '.dat'})
            except IOError:
                continue

        da = donutana(**in_dict)
        return da

    def zernike_corrections_from_hexapod(self, hexapod):
        """Brief Description

        Parameters
        ----------
        hexapod : list
            List of the five hexapod terms [dz, dx, dy, xt, yt]

        Returns
        -------
        zernike_corrections : array
            A 3 x 11 array containing the corresponding zid and zix and ziy
            terms.
        """

        # get hexapod to zernike

        hex_z5thetax, hex_z5thetay, hex_z6thetax, hex_z6thetay, \
            hex_z7delta, hex_z8delta = \
            hexapodtoZernike(hexapod[1], hexapod[2], hexapod[3], hexapod[4])

        zernike_correction = np.array([[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [hexapod[0], 0.0, 0.0],
                                       [0.0, hex_z5thetax, hex_z5thetay],
                                       [0.0, hex_z6thetax, hex_z6thetay],
                                       [hex_z7delta, 0.0, 0.0],
                                       [hex_z8delta, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]], dtype=np.float64)

        return zernike_correction

    def zernike_corrections_from_dictionary(self, in_dict):
        """Take an input dictionary, come up with an array of delta and theta
        zernike corrections.

        Parameters
        ----------
        in_dict : dictionary
            A dictionary with terms like 'dz' (for hexapod) or 'z05x' (for
            z5thetax)

        Returns
        -------
        zernike_correction : array
            A 3 x 11 array of all the zernike corrections

        """

        hexapod = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        zernike_correction = np.array([[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]], dtype=np.float64)
        ztype_dict = {'d': 0, 'x': 1, 'y': 2}
        # go through keys and add them to zernike_correction
        for key in in_dict:
            entry = in_dict[key]
            if key == 'dz':
                hexapod[0] = hexapod[0] + entry
            elif key == 'dx':
                hexapod[1] = hexapod[1] + entry
            elif key == 'dy':
                hexapod[2] = hexapod[2] + entry
            elif key == 'xt':
                hexapod[3] = hexapod[3] + entry
            elif key == 'yt':
                hexapod[4] = hexapod[4] + entry
            # else extract zernike correction
            elif (key[0] == 'z') * (len(key) == 4):
                znum = int(key[1:3]) - 1
                ztype = key[-1]
                ztype_num = ztype_dict[ztype]
                zernike_correction[znum][ztype_num] = \
                    zernike_correction[znum][ztype_num] + entry

        hexapod_correction = \
            self.zernike_corrections_from_hexapod(hexapod)
        zernike_correction = zernike_correction + hexapod_correction

        return zernike_correction

    def zernike_dictionary_from_corrections(self, zernike_correction):
        """Convert zernike corrections into a dictionary.

        Parameters
        ----------
        zernike_correction : array
            A 3 x 11 array of all the zernike corrections

        Returns
        -------
        zernike_dictionary : dictionary
            A dictionary with terms like 'dz' (for hexapod) or 'z05x' (for
            z5thetax)

        See Also
        --------
        zernike_corrections_from_dictionary : inverse procedure

        """

        ztype_dict = {'d': 0, 'x': 1, 'y': 2}

        # create empty dictionary
        zernike_dictionary = dict(z04d=0.0, z04x=0.0, z04y=0.0,
                                  z05d=0.0, z05x=0.0, z05y=0.0,
                                  z06d=0.0, z06x=0.0, z06y=0.0,
                                  z07d=0.0, z07x=0.0, z07y=0.0,
                                  z08d=0.0, z08x=0.0, z08y=0.0,
                                  z09d=0.0, z09x=0.0, z09y=0.0,
                                  z10d=0.0, z10x=0.0, z10y=0.0,
                                  z11d=0.0, z11x=0.0, z11y=0.0)
        for key in zernike_dictionary:
            znum = int(key[1:3]) - 1
            ztype = key[-1]
            ztype_num = ztype_dict[ztype]
            zernike_dictionary[key] = zernike_dictionary[key] + \
                zernike_correction[znum][ztype_num]

        return zernike_dictionary

    def plane(self, in_dict, coords,
              windowed=True, order_dict={'x2': {'p': 2, 'q': 0},
                                         'y2': {'p': 0, 'q': 2},
                                         'xy': {'p': 1, 'q': 1}}):
        """create a wavefront across the focal plane

        Parameters
        ----------
        in_dict : dictionary
            dictionary containing the zernike corrections

        coords : array
            An array of [[coordx, coordy, ext_num]] of the locations sampled

        windowed : bool, optional ; depreciated
            Do we calculate the windowed moments, or unwindowed? Default true.

        order_dict : dictionary, optional
            A dictionary of dictionaries indicating the name and the powers of
            the moments calculated.
            Default calculates the second moments.

        Returns
        -------
        moments : dictionary
            A dictionary containing all the moments calculated and
            corresponding convenient linear combinations (ellipticities, etc).

        """

        if 'history' in self.verbosity:
            self.history.append(in_dict.copy())

        if 'rzero' in in_dict.keys():
            rzero = in_dict['rzero']
        elif 'fwhm' in in_dict.keys():
            rzero = fwhm_to_rzero(in_dict['fwhm'])
        else:
            # set the atmospheric contribution to some nominal level
            rzero = 0.14

        zernikes = self.zernikes(coords, in_dict)
        N = len(zernikes)
        rzeros = [rzero] * N
        backgrounds = [self.background] * N
        if self.input_dict['randomFlag']:
            # TODO: This is not tested
            thresholds = [np.sqrt(self.number_electrons)] * N
        else:
            thresholds = [0] * N
        # make stamps
        stamps = self.stamp_factory(zernikes, rzeros, coords)
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
            self, in_dict, coords, average=np.mean, boxdiv=0, subav=False,
            windowed=True, order_dict={'x2': {'p': 2, 'q': 0},
                                       'y2': {'p': 0, 'q': 2},
                                       'xy': {'p': 1, 'q': 1}}):
        """create a wavefront across the focal plane and average into boxes

        Parameters
        ----------
        in_dict : dictionary
            dictionary containing the zernike corrections

        coords : array
            An array of [[coordx, coordy, ext_num]] of the locations sampled

        windowed : bool, optional ; depreciated
            Do we calculate the windowed moments, or unwindowed? Default true.

        order_dict : dictionary, optional
            A dictionary of dictionaries indicating the name and the powers of
            the moments calculated.
            Default calculates the second moments.

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
        moments_unaveraged = self.plane(in_dict, coords, windowed=windowed,
                                        order_dict=order_dict)

        # now average
        moments = average_dictionary(moments_unaveraged, average,
                                     boxdiv=boxdiv, subav=subav)

        poles = convert_moments(moments)

        return poles

    def zernikes(self, coords, in_dict):
        """create a list of zernikes at these coordinate locations

        Parameters
        ----------
        coords : list
            A list of [[x, y, ext_num]] detailing the locations we wish to
            sample on the focal plane

        in_dict : dictionary
            dictionary containing the zernike corrections

        Returns
        -------
        zernikes : list
            A list of the zernike polynomial coefficients [[z1, z2, z3 ...],
            ...]calculated from the interpolated reference mesh plus
            corrections.

        Notes
        -----
        Implicitly included are the reference corrections between the focus
        chips and the focal plane. I do NOT implicitly include any
        image-specific corrections.

        """

        zernike_corrections_in = self.zernike_corrections_from_dictionary(
            in_dict)
        #TODO: why the copy?
        zernike_corrections = np.copy(zernike_corrections_in)

        # add in the cross reference correction
        zernike_corrections = zernike_corrections + \
            self.reference_correction[:len(zernike_corrections)]

        # make the function
        correction = self.zernike_corrections(zernike_corrections,
                                              self.da)

        # make the final polynomials
        zernikes = []
        for coord in coords:
            # TODO: double check this new method works
            ## zernike = [
            ##     correctioni[0](
            ##         self.ccddict[int(coord[2])],
            ##         [coord[0]], [coord[1]])
            ##     + correctioni[1]
            ##     + correctioni[2] * coord[1]
            ##     + correctioni[3] * coord[0]
            ##     for correctioni in correction]
            zernike = [correctioni(self.decaminfo.ccddict[int(coord[2])],
                                   coord[0], coord[1])
                       for correctioni in correction]
            zernikes.append(zernike)
        zernikes = np.array(zernikes).tolist()
        return zernikes

    def zernike_corrections(self, zernike_corrections_in, da):
        """Create a list of functions for getting each zernike polynomial on
        the focal plane.

        Parameters
        ----------
        zernike_corrections_in : array
            An array with the z delta and theta corrections to the reference
            mesh.

        da : object
            Aaron's donutana object. Needed to access the refmesh functions.

        Returns
        -------
        zernike_corrections : list of functions
            A list of functions, ith corresponding to a fuction to calculate
            the (i+1)th zernike polynomial coefficient.

        Notes
        -----
        The way I did it (with the evals) feels really kludgey.
        """

        numfac = 0.0048481  # rad / arcsec um/mm
        zernike_corrections = []
        zdelta = zernike_corrections_in[:, 0]
        zthetax = zernike_corrections_in[:, 1]
        zthetay = zernike_corrections_in[:, 2]
        z_length = len(zdelta)

        def empty(a, b, c):
            return 0

        def ran(a, b, c):
            # for z2 and z3
            N = 30.   # to move +- 15 microns
            return N * (np.random.random() - 0.5) / 8.4

        # TODO: THIS IS AWFUL
        # a list of functions used for getting the zernike polynomials on the
        # focal plane
        def z11func(ext_name, coordx, coordy):
            z11 = da.z11RefMesh.doInterp(ext_name, [coordx], [coordy]) + \
                zdelta[10] + coordy * zthetax[10] + coordx * zthetay[10]
            return z11

        def z10func(ext_name, coordx, coordy):
            z10 = da.z10RefMesh.doInterp(ext_name, [coordx], [coordy]) + \
                zdelta[9] + coordy * zthetax[9] + coordx * zthetay[9]
            return z10

        def z9func(ext_name, coordx, coordy):
            z9 = da.z9RefMesh.doInterp(ext_name, [coordx], [coordy]) + \
                zdelta[8] + coordy * zthetax[8] + coordx * zthetay[8]
            return z9

        def z8func(ext_name, coordx, coordy):
            z8 = da.z8RefMesh.doInterp(ext_name, [coordx], [coordy]) + \
                zdelta[7] + coordy * zthetax[7] + coordx * zthetay[7]
            return z8

        def z7func(ext_name, coordx, coordy):
            z7 = da.z7RefMesh.doInterp(ext_name, [coordx], [coordy]) + \
                zdelta[6] + coordy * zthetax[6] + coordx * zthetay[6]
            return z7

        def z6func(ext_name, coordx, coordy):
            z6 = da.z6RefMesh.doInterp(ext_name, [coordx], [coordy]) + \
                zdelta[5] + coordy * zthetax[5] + coordx * zthetay[5]
            return z6

        def z5func(ext_name, coordx, coordy):
            z5 = da.z5RefMesh.doInterp(ext_name, [coordx], [coordy]) + \
                zdelta[4] + coordy * zthetax[4] + coordx * zthetay[4]
            return z5

        # include the 172. for proper unit conversions
        def z4func(ext_name, coordx, coordy):
            z4 = da.zRefMesh.doInterp(ext_name, [coordx], [coordy]) / 172. + \
                zdelta[3] / 172. + \
                zthetax[3] * numfac / 172. * coordy + \
                zthetay[3] * numfac / 172. * coordx

            return z4

        # z3 and z2 are calculated so that the artificial stars are at the same
        # centroid as the coordinates; z2 and z3 are used to shift the
        # artificial star coordinate system to line up with the DECam pixels.
        def z3func(ext_name, coordx, coordy):
            # <x_i> = -0.56477 * zi + 7.87479
            # so zi = (<x_i> - 7.87479) / -0.56477
            # where the coords are in pixel and boxed
            pixel = self.decaminfo.getPixel(ext_name, coordx, coordy)
            # convert pixel to be in center of 16 x 16 box with
            # x -> p with int(p) = 7
            # so p = x - int(x) + 7
            # the following is 7 for nPixels = 16
            pixel_offset = int(self.input_dict['nPixels'] / 2 - 1)
            coord = pixel[1] - int(pixel[1]) + pixel_offset

            # in order to incorporate z7 correction, need to calculate z7
            if z_length > 6:
                z7 = z7func(ext_name, coordx, coordy)
            else:
                z7 = 0

            # TODO:
            # this equation has only been fitted for 16 x 16
            z3 = (coord - 0.87479 - pixel_offset) / -0.56477 - 2 * z7 + \
                zdelta[2] + coordy * zthetax[2] + coordx * zthetay[2]

            return z3

        def z2func(ext_name, coordx, coordy):
            # <x_i> = -0.56477 * zi + 7.87479
            # so zi = (<x_i> - 7.87479) / -0.56477
            # where the coords are in pixel and boxed
            pixel = self.decaminfo.getPixel(ext_name, coordx, coordy)
            # convert pixel to be in center of 16 x 16 box with
            # x -> p with int(p) = 7
            # so p = x - int(x) + 7
            pixel_offset = int(self.input_dict['nPixels'] / 2 - 1)
            coord = pixel[0] - int(pixel[0]) + pixel_offset

            # in order to incorporate z8 correction, need to calculate z8
            if z_length > 7:
                z8 = z8func(ext_name, coordx, coordy)
            else:
                z8 = 0

            # TODO:
            # this equation has only been fitted for 16 x 16;
            z2 = (coord - 0.87479 - pixel_offset) / -0.56477 - 2 * z8 + \
                zdelta[1] + coordy * zthetax[1] + coordx * zthetay[1]

            return z2

        # piston contributes nothing
        def z1func(ext_name, coordx, coordy):

            z1 = 0 + zdelta[0] + coordy * zthetax[0] + coordx * zthetay[0]

            return z1

        # TODO: double check this new method works
        ## for i in range(len(zernike_corrections_in)):
        ##     if i == 0:
        ##         zcorri = [empty, zdelta[i], zthetax[i], zthetay[i]]
        ##     elif i < 3:
        ##         if self.input_dict['randomFlag']:
        ##             # if randomness is enabled, allow z2 and z3 to vary
        ##             # slightly
        ##             zcorri = [ran, zdelta[i], zthetax[i], zthetay[i]]
        ##         else:
        ##             zcorri = [empty, zdelta[i], zthetax[i], zthetay[i]]
        ##     elif i == 3:
        ##         # correct z4 term for mm input
        ##         zcorri = [z4func,
        ##                   zdelta[i] / 172.,
        ##                   zthetax[i] * numfac / 172.,
        ##                   zthetay[i] * numfac / 172.]
        ##     else:
        ##         # this is totally a terrible way of doing this,
        ##         # full of safety
        ##         # issues
        ##         zcorri = [eval("da.z{0}RefMesh.doInterp".format(i+1)),
        ##                   zdelta[i], zthetax[i], zthetay[i]]
        ##     zernike_corrections.append(zcorri)

        zernike_corrections = [eval("z{0}func".format(i+1))
                               for i in range(z_length)]

        return zernike_corrections

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
            if ext_num == 61:
                # N30 is bad
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

    def check_full_bounds(self, data, boxdiv, minimum_number, average):
        """Convenience method for checking whether my sampling hits all
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

        average : function
            The function that we use to average over.

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
        x_av, x_av2, N, _ = self.decaminfo.average_boxdiv(x, y, x, average,
                                                          boxdiv=boxdiv,
                                                          Ntrue=True)
        # check that all N >= minimum_number

        success = (len(x_av) == len(x_box)) * np.all(N >= minimum_number)

        return success

    def __add__(self, other):
        """Take other's items and append them to this one's data attribute.
        NOTE: data is NOT defined here!"""

        other_keys = other.data.keys()
        # pop the keys that are not in self:
        self_keys = self.data.keys()
        keys = [key for key in other_keys if key in self_keys]

        return_dict = {}
        for key in keys:
            return_dict.update({key: np.append(self.data[key],
                                               other.data[key])})
        return return_dict
