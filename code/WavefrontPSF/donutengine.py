#!/usr/bin/env python
"""
File: donutengine.py
Author: Chris Davis
Description: Specific implimentation of WavefrontPSF using donutlib to make the
             donut images.

TODO: Remove automatic overwriting of old data parameter!

TODO: makedonut stuff should actually be a PSF_Interpolator object
"""

import numpy as np
import pandas as pd
from os import path, makedirs
import pickle

from wavefront import Wavefront
from digestor import Digestor
from psf_interpolator import kNN_Interpolator, PSF_Interpolator
from psf_evaluator import PSF_Evaluator, Moment_Evaluator

from donutlib.makedonut import makedonut

class Generic_Donutengine_Wavefront(Wavefront):
    """
    Idea is you just call with a list of x, y, zernikes, and rzero and you're good to go
    """

    def __init__(self):
        evaluator = Moment_Evaluator()
        interpolator = PSF_Interpolator()
        super(Generic_Donutengine_Wavefront, self).__init__(
                PSF_Interpolator=interpolator,
                PSF_Evaluator=evaluator,
                model=None)
        self.PSF_Drawer = Zernike_to_Pixel_Interpolator()

    def draw_psf(self, xs, ys, zernikes, rzeros):
        # make the donut!
        stamps = []
        for index, inputZernikeArray in enumerate(zernikes):
            xi = xs[index]
            yi = ys[index]
            rzero = rzeros[index]
            stamp = self.PSF_Drawer(xi, yi,
                                    inputZernikeArray=inputZernikeArray,
                                    rzero=rzero)
            stamps.append(stamp)
        stamps = np.array(stamps)

        return stamps


class DECAM_Model_Wavefront(Wavefront):
    """
    Interpolate zernikes, make donuts, evaluate them
    """

    def __init__(self, data, misalignment={'rzero': 0.14}, interp_kwargs={}):
        # data here is a csv with all the zernikes
        evaluator = Moment_Evaluator()
        interpolator = kNN_Interpolator(data, **interp_kwargs)
        super(DECAM_Model_Wavefront, self).__init__(
                PSF_Interpolator=interpolator,
                PSF_Evaluator=evaluator,
                model=None)
        self.PSF_Drawer = Zernike_to_Pixel_Interpolator()

        self.input_misalignment = misalignment
        self.misalignment = self.misalign_optics(misalignment)

    def __call__(self, misalignment, overwrite=True, **kwargs):
        if overwrite:
            self.misalignment = self.misalign_optics(misalignment)
            # check that 'rzero' is in misalignment
            if 'rzero' not in self.misalignment:
                print('Warning! rzero not in misalignment. Setting to 0.14')
                self.misalignment['rzero'] = 0.14
        else:
            self.misalignment.update(self.misalign_optics(misalignment))
        # make new data
        try:
            evaluated_psfs = self.get_psf_stats(**kwargs)
        except:
            try:
                x = self.data['x']
                y = self.data['y']
                evaluated_psfs = self.get_psf_stats(x, y, **kwargs)
            except:
                print('I have no x and y coordinates, either given or in my data! Just setting the misalignment to something new.')
                evaluated_psfs = None
        return evaluated_psfs

    def save(self, out_path):
        # make_donut is a nasty offender
        self.PSF_Drawer.make_donut = None
        super(DECAM_Model_Wavefront, self).save(out_path)
        # now remake make_donut
        self.PSF_Drawer.make_donut = self.PSF_Drawer.remakedonut()

    def zernike_corrections_from_hexapod(self, dz=0, dx=0, dy=0, xt=0, yt=0):
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
            hexapodtoZernike(dx, dy, xt, yt)

        zernike_correction = np.array([[0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [dz, 0.0, 0.0],
                                       [0.0, hex_z5thetax, hex_z5thetay],
                                       [0.0, hex_z6thetax, hex_z6thetay],
                                       [hex_z7delta, 0.0, 0.0],
                                       [hex_z8delta, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0]], dtype=np.float64)

        return zernike_correction

    def misalign_optics(self, misalign_dict):
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
        for key in misalign_dict:
            entry = misalign_dict[key]
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
            self.zernike_corrections_from_hexapod(*hexapod)
        zernike_correction = zernike_correction + hexapod_correction


        # create empty dictionary
        zernike_dictionary = dict(
                                  z1d=0.0, z1x=0.0, z1y=0.0,
                                  z2d=0.0, z2x=0.0, z2y=0.0,
                                  z3d=0.0, z3x=0.0, z3y=0.0,
                                  z4d=0.0, z4x=0.0, z4y=0.0,
                                  z5d=0.0, z5x=0.0, z5y=0.0,
                                  z6d=0.0, z6x=0.0, z6y=0.0,
                                  z7d=0.0, z7x=0.0, z7y=0.0,
                                  z8d=0.0, z8x=0.0, z8y=0.0,
                                  z9d=0.0, z9x=0.0, z9y=0.0,
                                  z10d=0.0, z10x=0.0, z10y=0.0,
                                  z11d=0.0, z11x=0.0, z11y=0.0)
        ztype_dict = {0: 'd', 1: 'x', 2: 'y'}
        for znum in xrange(1, 11):
            for ztype in xrange(3):
                key = 'z{0}{1}'.format(znum, ztype_dict[ztype])
                zernike_dictionary[key] = zernike_dictionary[key] + \
                    zernike_correction[znum - 1][ztype]

        try:
            zernike_dictionary['rzero'] = misalign_dict['rzero']
        except KeyError:
            print('Warning! No rzero found! Setting to 0.14')
            zernike_dictionary['rzero'] = 0.14

        return zernike_dictionary

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
        z2, z3 : arrays

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

    def draw_zernikes(self, x, y):

        zernikes = self.PSF_Interpolator(x, y)
        for z in xrange(1, 12):
            key = 'z{0}'.format(z)
            if z != 4:
                correction = self.misalignment[key + 'd'] + \
                             self.misalignment[key + 'y'] * x + \
                             self.misalignment[key + 'x'] * y
            else:
                # z4 needs to be in waves, too!
                numfac = 0.0048481  # rad / arcsec um/mm
                wavefac = 172.  # waves / mm
                correction = self.misalignment[key + 'd'] + \
                             self.misalignment[key + 'y'] * x * numfac + \
                             self.misalignment[key + 'x'] * y * numfac
            if key not in zernikes:
                zernikes[key] = correction
            else:
                zernikes[key] += correction
            if z == 4:
                zernikes[key] /= 172.

        self.data = zernikes

        return zernikes

    def draw_psf(self, x, y,
                 **kwargs):
        # draw many PSFs from the x and y coords as well as other params
        # get input parameters
        zernikes = self.draw_zernikes(x, y)

        # make the donut!
        stamps = []
        for index, zernike in zernikes.iterrows():
            xi = zernike['x']
            yi = zernike['y']
            inputZernikeArray = np.array([zernike['z{0}'.format(i)]
                                          for i in xrange(1, 12)])
            rzero = self.misalignment['rzero']
            stamp = self.PSF_Drawer(xi, yi,
                                    inputZernikeArray=inputZernikeArray,
                                    rzero=rzero)
            stamps.append(stamp)
        stamps = np.array(stamps)
        # append stamps to the zernike data
        self.cutouts = stamps

        return stamps

    def get_psf_stats(self, x, y, **kwargs):
        evaluated_psfs = self.PSF_Evaluator(self.draw_psf(x, y, **kwargs))
        psf_keys = ['e0', 'e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2',
                    'a4', 'flux', 'Mx', 'My']
        self.data = pd.concat([self.data, evaluated_psfs[psf_keys]], axis=1)
        self.field, self.bins_x, self.bins_y = self.reduce_data_to_field(self.data, np.median, 1)

        return evaluated_psfs

class Data_Wavefront(Wavefront):
    """
    Take a database of PSF params and interpolate over that!
    """

    def __init__(self, data):
        evaluator = PSF_Evaluator()
        interpolator = kNN_Interpolator(data,
                y_keys=['e0', 'e1', 'e2'])
        super(Data_Wavefront, self).__init__(
                PSF_Interpolator=interpolator,
                PSF_Evaluator=evaluator,
                model=data)


def hexapodtoZernike(dx,dy,xt,yt):

    # input units are microns,arcsec

    # latest calibration matrix
    hexapodArray20121020 = np.array(
            ((  0.00e+00 ,  1.07e+05 ,  4.54e+03 ,  0.00e+00),
             (  1.18e+05 , -0.00e+00 ,  0.00e+00 , -4.20e+03),
             ( -4.36e+04 ,  0.00e+00 ,  0.00e+00 , -8.20e+01),
             (  0.00e+00 ,  4.42e+04 , -8.10e+01 ,  0.00e+00)
            ))

    # take its inverse
    alignmentMatrix = np.matrix(hexapodArray20121020)
    aMatrixInv = alignmentMatrix.I

    # build column vector of the hexapod dof
    hexapodList = (dx,dy,xt,yt)
    hexapodColVec = np.matrix(hexapodList).transpose()

    # calculate Zernikes
    zernikeM = aMatrixInv * hexapodColVec
    zernikeVector = zernikeM.A

    aveZern5ThetaX = zernikeVector[0][0]
    aveZern6ThetaX = zernikeVector[1][0]
    z07d = zernikeVector[2][0]
    z08d = zernikeVector[3][0]

    # output values
    z05x = aveZern5ThetaX
    z06y = aveZern5ThetaX
    z06x = aveZern6ThetaX
    z05y = -aveZern6ThetaX

    return z05x, z05y, z06x, z06y, z07d, z08d

def zerniketoHexapod(z05x, z05y, z06x, z06y, z07d, z08d):

    # inverse of hexapodtoZernike2

    # latest calibration matrix
    hexapodArray20121020 = np.array(
            ((  0.00e+00 ,  1.07e+05 ,  4.54e+03 ,  0.00e+00),
             (  1.18e+05 , -0.00e+00 ,  0.00e+00 , -4.20e+03),
             ( -4.36e+04 ,  0.00e+00 ,  0.00e+00 , -8.20e+01),
             (  0.00e+00 ,  4.42e+04 , -8.10e+01 ,  0.00e+00)
            ))
    alignmentMatrix = np.matrix(hexapodArray20121020)

    # note that z05y ~ -z06x and z05x ~ z06y
    # so we average them
    aveZern5ThetaX = np.mean([z05x, z06y])
    aveZern6ThetaX = np.mean([-z05y, z06x])
    zernikeList = (aveZern5ThetaX, aveZern6ThetaX, z07d, z08d)
    zernikeColVec = np.matrix(zernikeList).transpose()

    hexapodM = alignmentMatrix * zernikeColVec
    hexapodVector = hexapodM.A

    dx = hexapodVector[0][0]
    dy = hexapodVector[1][0]
    xt = hexapodVector[2][0]
    yt = hexapodVector[3][0]

    return dx, dy, xt, yt

class Zernike_to_Pixel_Interpolator(PSF_Interpolator):
    """PSF Interpolator that inputs zernikes, focal plane coordinates, and
    Fried parameter and returns a pixel representation of the PSF.
    """

    def __init__(self, **kwargs):
        # set up makedonut
        self.makedonut_dict = {'nbin': 256,
                               'nPixels': 32,
                               'pixelOverSample': 8,
                               'scaleFactor': 1,
                               'randomFlag': 0}
        self.makedonut_dict.update(kwargs)
        self.make_donut = makedonut(**self.makedonut_dict)


    def remakedonut(self):
        """Remake make_donut

        Notes
        -----
        No input or return.

        """
        self.make_donut = makedonut(**self.makedonut_dict)
        return

    def draw_donut(self, x, y, inputZernikeArray, rzero):
        stamp = self.make_donut.make(inputZernikeArray=inputZernikeArray,
                                     rzero=rzero,
                                     nEle=1e0,
                                     background=0,
                                     xDECam=x,
                                     yDECam=y).astype(np.float64)
        return stamp

    def interpolate(self, x, y, inputZernikeArray, rzero):
        stamp = self.draw_donut(xi, yi, inputZernikeArray, rzero)

        return stamp


if __name__ == '__main__':
    # do some basic runs to demonstrate functionality
    from digestor import Digestor
    digestor = Digestor()
    interpolation_data = digestor('/Users/cpd/Desktop/ComboMeshes2/Mesh_Science-20140212s2-v1i2_All_train.csv')
    wf = DECAM_Model_Wavefront(interpolation_data)
    comparison_data = digestor('/Users/cpd/Desktop/ComboMeshes2/Mesh_Science-20140212s2-v1i2_All_val.csv')
    wf_comparison = DECAM_Model_Wavefront(comparison_data, interp_kwargs={'n_neighbors': 1, 'weights': 'uniform', 'p': 1})
    vals = wf_comparison.PSF_Interpolator.data[:5000]
    x = vals['x']
    y = vals['y']
    import ipdb; ipdb.set_trace()
    wf({'rzero': 0.14}, x=x, y=y)


    import matplotlib.pyplot as plt
    wf.plot_field('e0')
    plt.show()
    import ipdb; ipdb.set_trace()
