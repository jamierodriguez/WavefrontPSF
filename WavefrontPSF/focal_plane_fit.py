#!/usr/bin/env python
"""
File: focal_plane_fit.py
Author: Chris Davis
Description: Class for creating wavefronts on a generic focal plane.
"""

from __future__ import print_function, division
import numpy as np
from wavefront import Wavefront
from hexapodtoZernike_cpd import hexapodtoZernike
from donutlib.donutana import donutana
from routines import average_dictionary
from os import path
from analytic_moments import analytic_data
from routines_moments import convert_moments

class FocalPlaneFit(Wavefront):
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
        Use: Science-20130325s1-v1i2_All
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
                 path_mesh='/u/ec/roodman/Astrophysics/Donuts/ComboMeshes/',
                 mesh_name='Science-20130325s1-v1i2_All',
                 methodVal=(),
                 verbosity=['history'],
                 **args):

        # do the old init for Wavefront
        super(FocalPlaneFit, self).__init__(**args)

        if path.exists(path_mesh):
            self.path_mesh = path_mesh
        elif path.exists('/u/ec/roodman/Astrophysics/Donuts/Meshes/'):
            print('Your path_mesh is incorrect! Trying default ki-ls at',
                    '/u/ec/roodman/Astrophysics/Donuts/Meshes/')
            self.path_mesh = '/u/ec/roodman/Astrophysics/Donuts/Meshes/'
        elif path.exists('/Users/cpd/Desktop/Meshes/'):
            print('Your path_mesh is incorrect! Trying your computer at',
                    '/Users/cpd/Desktop/Meshes/')
            self.path_mesh = '/Users/cpd/Desktop/Meshes/'

        self.mesh_name = mesh_name
        self.methodVal = methodVal
        self.da = self.init_da(path_mesh=self.path_mesh,
                               mesh_name=self.mesh_name,
                               methodVal=self.methodVal)

        self.verbosity = verbosity
        self.history = []

        self.jitter_modes = ['e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']

        # NOTE: THIS DEPENDS ON THE MESH YOU CHOOSE TO USE.
        self.reference_correction = -1 * np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            ])
        ## # Science20120915s1v3_134239
        ## self.reference_correction = -1 * np.array([
        ##     [0, 0, 0],
        ##     [0, 0, 0],
        ##     [0, 0, 0],
        ##     [1690, 0.496, 6.14],
        ##     [-0.0115, 0.00232, 0.000538],
        ##     [-0.11, 0.000627, 0.000801],
        ##     [0.0113, -0.000329, 9.68e-5],
        ##     [-0.145, 0.000396, 0.00024],
        ##     [-0.0884, 0.000365, 0.000158],
        ##     [0.0452, 0.000159, 0.000191],
        ##     [0, 0, 0]])


    def init_da(self, path_mesh, mesh_name, methodVal=()):

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
        nInterpGrid = 32
        if len(methodVal) != 2:
            if mesh_name == 'Science-20130325s1-v1i2_All':
                #methodVal = (250, 1.0)  # use 250 NN, 1.0 mm offset distance
                methodVal = (20, 1.0)  # use 20 NN, 1.0 mm offset distance
            else:
                methodVal = (4, 1.0)

        in_dict = {"zPointsFile": path_mesh + "z4Mesh_" +
                                 mesh_name + ".dat",
                  "sensorSet": sensorSet,
                  "doTrefoil": True,
                  "doSpherical": True,
                  "doQuadrefoil": False,
                  "unVignettedOnly": False,
                  "interpMethod": method,
                  "methodVal": methodVal,
                  "nInterpGrid": nInterpGrid,
                  "histFlag": False,  # True,
                  "debugFlag": True,  # True,
                  "donutCutString": ""}

        for zi in range(4, 12):
            try:
                in_dict.update({'z{0}PointsFile'.format(zi):
                               path_mesh + 'z{0}Mesh_'.format(zi) +
                               mesh_name + '.dat'})
            except IOError:
                continue
        ## for key in sorted(in_dict.keys()):
        ##     print(key, in_dict[key])
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
        zernike_dictionary = dict(
                                  z01d=0.0, z01x=0.0, z01y=0.0,
                                  z02d=0.0, z02x=0.0, z02y=0.0,
                                  z03d=0.0, z03x=0.0, z03y=0.0,
                                  z04d=0.0, z04x=0.0, z04y=0.0,
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
              windowed=True, order_dict={}):
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
            Default calculates the second and third moments.

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
        else:
            # set the atmospheric contribution to some nominal level
            rzero = 0.14

        zernikes = self.zernikes(in_dict, coords)
        N = len(zernikes)
        rzeros = [rzero] * N
        backgrounds = [self.background] * N
        ## jitter_keys = ['e1', 'e2']
        ## jitter_dict = {}
        ## # guess e0
        ## plane = analytic_data(zernikes, rzero, coords=coords)
        ## e0 = np.median(plane['e0'])
        ## for jitter_key in jitter_keys:
        ##     if jitter_key in in_dict.keys():
        ##         jitter_dict.update({jitter_key: in_dict[jitter_key] / e0})
        ## jitters = [jitter_dict] * N  # note: if you modify one member,
        ##                              # you will modify all. thanks python!
        jitters = []
        if self.input_dict['randomFlag']:
            # TODO: This is not tested
            thresholds = [np.sqrt(self.number_electrons)] * N
        else:
            thresholds = [0] * N
        # make stamps
        stamps = self.stamp_factory(zernikes, rzeros, coords, jitters)
        # make moments
        moments = self.moment_dictionary(stamps,
                                         coords,
                                         backgrounds=backgrounds,
                                         thresholds=thresholds,
                                         verbosity=self.verbosity,
                                         windowed=windowed,
                                         order_dict=order_dict)

        ## for jitter_mode in self.jitter_modes:
        ##     if (jitter_mode in moments) * (jitter_mode in in_dict):
        ##         moments[jitter_mode] += in_dict[jitter_mode]

        return moments

    def plane_averaged(
            self, in_dict, coords, average=np.mean, boxdiv=0, subav=False,
            windowed=True, order_dict={}):
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
        moments_unaveraged = self.plane(in_dict, coords, windowed=windowed,
                                        order_dict=order_dict)

        # now average
        moments = average_dictionary(moments_unaveraged, average,
                                     boxdiv=boxdiv, subav=subav)

        return moments

    def analytic_plane(self, in_dict, coords):
        if 'history' in self.verbosity:
            self.history.append(in_dict.copy())
        if 'rzero' in in_dict.keys():
            rzero = in_dict['rzero']
        else:
            # set the atmospheric contribution to some nominal level
            rzero = 0.14

        zernikes = np.array(self.zernikes(in_dict, coords))

        plane = analytic_data(zernikes, rzero, coords=coords)

        for jitter_mode in self.jitter_modes:
            if (jitter_mode in plane) * (jitter_mode in in_dict):
                plane[jitter_mode] += in_dict[jitter_mode]

        plane = convert_moments(plane)

        return plane

    def analytic_plane_averaged(self, in_dict, coords, average=np.mean,
                                boxdiv=0, subav=False):

        plane_unaveraged = self.analytic_plane(in_dict, coords)

        plane = average_dictionary(plane_unaveraged, average,
                                   boxdiv=boxdiv, subav=subav)

        return plane

    def zernikes(self, in_dict, coords):
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
        zernike_corrections = np.copy(zernike_corrections_in)

        # add in the cross reference correction
        zernike_corrections = zernike_corrections + \
            self.reference_correction[:len(zernike_corrections)]

        numfac = 0.0048481  # rad / arcsec um/mm
        zdelta = zernike_corrections[:, 0]
        zthetax = zernike_corrections[:, 1]
        zthetay = zernike_corrections[:, 2]
        z_length = len(zdelta)

        zernikes = np.array([[0] * 3 +

            [self.da.meshDict['zMesh'].doInterp(
            self.decaminfo.ccddict[int(coord[2])], [coord[0]], [coord[1]])
            / 172. +
            zdelta[4 - 1] / 172. +
            coord[1] * numfac / 172. * zthetax[4 - 1] +
            coord[0] * numfac / 172. * zthetay[4 - 1]] +

            [self.da.meshDict['z{0}Mesh'.format(iZ)].doInterp(
            self.decaminfo.ccddict[int(coord[2])], [coord[0]], [coord[1]]) +
            zdelta[iZ - 1] +
            coord[1] * zthetax[iZ - 1] +
            coord[0] * zthetay[iZ - 1]
            for iZ in range(5, z_length + 1)]

            for coord in coords])

        return zernikes

    def adjust_center(self, x, y, zernikes):
        """A method for adjusting z2 and z3

        Parameters
        ----------
        x, y : arrays
            Arrays of the first moments that we want to make our image have

        zernikes : arrays
            Arrays of the zernikes

        Returns
        -------
        z2, z3 : 
        Notes
        -----
        Currently hardwired; TODO: create program to automatically generate the
        hardwired coefficients (as well as for rzero scaling!)

        These basically work empirically: I varied z8 or so on and found how
        the location of the first moment changed with them. I assume these guys
        are independent; that should be pretty reasonable. I guess I could
        figure these out...

        """
        # now adjust z2 and z3
        # These are currently hardwired!
        z7 = zernikes[:, 7 - 1]
        z8 = zernikes[:, 8 - 1]
        z9 = zernikes[:, 9 - 1]
        z10 = zernikes[:, 10 - 1]
        middle_value = 15.93750

        P_2_8 = -1.226 * z8 + 1.704e-1 * z8 ** 3
        P_2_10 = -1.546e-2 * z10 - 4.550e-3 * z10 ** 3
        P_3_7 = -1.19 * z7 + 1.642e-1 * z7 ** 3
        P_3_9 = -1.671e-2 * z9 - 4.908e-3 * z9 ** 3
        z2 = (x - P_2_8 - P_2_10 - middle_value) / -0.558
        z3 = (y - P_3_7 - P_3_9 - middle_value) / -0.558

        return z2, z3

    def random_coordinates(self, max_samples_box=5, boxdiv=0, chip=0,
                           border=30):
        """A method for generating coordinates by sampling over boxes

        Parameters
        ----------
        max_samples_box : int, optional
            Integer for the maximum number of stars per box that we sample
            from. Default is 5 stars per box.

        boxdiv : int, optional
            How many divisions we will put into the chip. Default is zero
            divisions on each chip.

        chip : int, optional
            What chip are we making the coordinates for? If 0, do for entire
            plane.

        border : float, optional
            How far away from the border will we sample? Unit is pixels.

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

        if chip == 0:
            ext_num_list = range(1, 63)
        else:
            ext_num_list = [chip]
        for ext_num in ext_num_list:
            ext_name = self.decaminfo.ccddict[ext_num]
            if ((ext_num == 61) * (chip != 61)):
                # N30 is bad
                continue

            ## # TODO: include the boxdiv part
            ## xpix = np.random.random_sample(max_samples_box) * \
            ##     (2048 - 2 * border) + border
            ## ypix = np.random.random_sample(max_samples_box) * \
            ##     (4096 - 2 * border) + border

            ## coords = [list(self.decaminfo.getPosition(
            ##     ext_name, xpix[i], ypix[i])) + [ext_num]
            ##     for i in xrange(max_samples_box)]

            ## coords_final += coords

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

    def compare(self, data_a, data_b, var_dict, chi_weights):
        """Compare another data object

        Parameters
        ----------
        data_a, data_b : dictionary of lists
            Entries which we will compare. Must be same size!

        var_dict : dictionary of lists
            Entries correspond to variances at each point

        chi_weights: dictionary
            What parameters are we comparing? Must be in data_a and data_b

        Returns
        -------
        chi_dict : dictionary
            Dictionary of chi2 for each point for each weight, plus a 'chi2'
            parameter that is the total value of the chi2.

        """
        chi_dict = {'chi2': 0}
        for key in chi_weights.keys():
            val_a = data_a[key]
            val_b = data_b[key]
            weight = chi_weights[key]
            if 'var_{0}'.format(key) in var_dict.keys():
                var = var_dict['var_{0}'.format(key)]
            else:
                print('No variance for', key, '!')
                var = 1

            chi2 = np.square(val_a - val_b) / var

            # update chi_dict
            chi_dict.update({key: chi2})
            chi_dict['chi2'] = chi_dict['chi2'] + np.sum(weight * chi2)

        # check whether chi_dict['chi2'] is an allowable number (finite
        # positive)
        if (chi_dict['chi2'] < 0) + (np.isnan(chi_dict['chi2'])):
            # if it isn't, make it really huge
            chi_dict['chi2'] = 1e20

        return chi_dict

