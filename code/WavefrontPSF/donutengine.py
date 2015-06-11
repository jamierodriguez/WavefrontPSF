#!/usr/bin/env python
"""
File: donutengine.py
Author: Chris Davis
Description: Specific implimentation of WavefrontPSF using donutlib to make the
             donut images.
"""

import numpy as np
import pandas as pd
from os import path, makedirs
import pickle

from WavefrontPSF.wavefront import Wavefront
from WavefrontPSF.digestor import Digestor
from WavefrontPSF.psf_interpolator import kNN_Interpolator, PSF_Interpolator
from WavefrontPSF.psf_evaluator import PSF_Evaluator, Moment_Evaluator
from WavefrontPSF.decamutil import decaminfo
from donutlib.makedonut import makedonut

class Generic_Donutengine_Wavefront(Wavefront):
    """
    Idea is you just call with a list of x, y, zernikes, and rzero and you're good to go

    TODO: keep track of zernikes that are not misaligned
    """

    def __init__(self, PSF_Evaluator,
                 PSF_Interpolator,
                 PSF_Drawer, **kwargs):
        super(Generic_Donutengine_Wavefront, self).__init__(
                PSF_Interpolator=PSF_Interpolator,
                PSF_Evaluator=PSF_Evaluator,
                model=None)
        self.PSF_Drawer = PSF_Drawer

    def save(self, out_path):
        # make_donut is a nasty offender
        self.PSF_Drawer.make_donut = None
        super(Generic_Donutengine_Wavefront, self).save(out_path)
        # now remake make_donut
        self.PSF_Drawer.make_donut = self.PSF_Drawer.remakedonut()

    def draw_psf(self, data, force_interpolation=True,
                 **kwargs):
        # draw many PSFs from the x and y coords as well as other params
        # get input parameters if they are not already in the data
        #if not np.all([key in data.columns for key in self.PSF_Drawer.x_keys]):
        data = self.PSF_Interpolator(data,
                force_interpolation=force_interpolation, **kwargs)
        # for i, row in data.iterrows():
        #     print('draw_psf', i, row)
        # make the donut!
        stamps, data = self.PSF_Drawer(data, **kwargs)

        return stamps, data

    def evaluate_psf(self, data, force_interpolation=False, **kwargs):

        stamps, data = self.draw_psf(data,
                force_interpolation=force_interpolation, **kwargs)

        evaluated_psfs = self.PSF_Evaluator(stamps, **kwargs)
        # this is sick:
        combined_df = evaluated_psfs.combine_first(data)

        return combined_df

    def __call__(self, data, **kwargs):
        return self.evaluate_psf(data, **kwargs)

class DECAM_Model_Wavefront(Generic_Donutengine_Wavefront):
    """
    Includes misalignments. Convenience class
    """

    def __init__(self, PSF_Interpolator_data=None, interp_kwargs={}, **kwargs):
        # data here is a csv with all the zernikes
        #TODO: make the csv here path independent
        if type(PSF_Interpolator_data) == type(None):
            PSF_Interpolator_data=pd.read_csv('/Users/cpd/Projects/WavefrontPSF/meshes/ComboMeshes2/Mesh_Science-20140212s2-v1i2_All_train.csv', index_col=0)
        interp = {}
        interp.update(interp_kwargs)

        # take z4 and divide by 172 to put it in waves as it should be
        PSF_Interpolator_data['z4'] /= 172.

        PSF_Interpolator = kNN_Interpolator(PSF_Interpolator_data, **interp)
        PSF_Drawer = Zernike_to_Misalignment_to_Pixel_Interpolator()
        PSF_Evaluator = Moment_Evaluator()
        super(DECAM_Model_Wavefront, self).__init__(
                PSF_Evaluator=PSF_Evaluator,
                PSF_Interpolator=PSF_Interpolator,
                PSF_Drawer=PSF_Drawer,
                model=None)

class Zernike_to_Pixel_Interpolator(PSF_Interpolator):
    """PSF Interpolator that inputs zernikes, focal plane coordinates, and
    Fried parameter and returns a pixel representation of the PSF.

    """

    def __init__(self, **kwargs):
        y_keys = ['stamp']
        x_keys = ['x', 'y', 'rzero',
                  'z1', 'z2', 'z3',
                  'z4', 'z5', 'z6',
                  'z7', 'z8',
                  'z9', 'z10',
                  'z11']
        super(Zernike_to_Pixel_Interpolator, self).__init__(
                y_keys=y_keys, x_keys=x_keys, **kwargs)
        # set up makedonut
        self.makedonut_dict = {'nbin': 256,
                               'nPixels': 32,
                               'pixelOverSample': 8,
                               'scaleFactor': 1,
                               'randomFlag': 0}
        for kwarg in kwargs:
            if kwarg in self.makedonut_dict:
                self.makedonut_dict[kwarg] = kwargs[kwarg]
        self.make_donut = makedonut(**self.makedonut_dict)

    def convert_zernike_floats_to_array(self, series, columns=['z{0}'.format(i) for i in range(1, 12)]):
        zernike = []
        for column in columns:
            if column in series:
                zernike.append(series[column])
            else:
                zernike.append(0)
        return zernike

    def remakedonut(self):
        """Remake make_donut

        Notes
        -----
        No input or return.

        """
        self.make_donut = makedonut(**self.makedonut_dict)
        return

    def draw_donut(self, data, **kwargs):
        # get x, y, inputzernikearray, rzero from data
        stamps = []
        for i, row in data.iterrows():
            inputZernikeArray = self.convert_zernike_floats_to_array(row)
            rzero = row['rzero']
            if np.all(inputZernikeArray):
                print('Warning! All zernikes in row {0} are missing or equal to zero!'.format(i))
            x = row['x']
            y = row['y']
            stamp = self.make_donut.make(inputZernikeArray=inputZernikeArray,
                                         rzero=rzero,
                                         nEle=1e0,
                                         background=0,
                                         xDECam=x,
                                         yDECam=y).astype(np.float64)
            # print('draw donut', i, x, y, rzero, inputZernikeArray)
            # print(row)
            stamps.append(stamp)
        stamps = np.array(stamps)

        return stamps, data

    def interpolate(self, data, **kwargs):
        # print('interpolate', data)
        stamps, data = self.draw_donut(data, **kwargs)
        # unfortunately, 2d data cannot go into pandas dataframes

        return stamps, data

class Zernike_to_Misalignment_to_Pixel_Interpolator(Zernike_to_Pixel_Interpolator):
    def misalign_optics(self, misalignment):
        return misalign_optics(misalignment)

    # take set of zernikes, misalign them, then get pixel basis
    def misalign_zernikes(self, data, misalignment={}):
        # get the right format for the misalignment
        misalignment = self.misalign_optics(misalignment)

        # make copy of data with the misalignment information NOT modifying the
        # base zernikes!
        data_with_misalignments = {}
        ones = np.ones(len(data))
        for key in misalignment:
            data_with_misalignments[key] = misalignment[key] * ones
        data_with_misalignments = pd.DataFrame(data_with_misalignments).combine_first(data)

        # return a copy of the data with zernikes modified by misalignments
        # data_reduced = misalign_zernikes(data, misalignment)
        data_reduced = misalign_zernikes(data_with_misalignments)
        return data_reduced, data_with_misalignments

    def draw_donut(self, data, misalignment={}, **kwargs):
        # misalign zernikes
        data_reduced, data_with_misalignments = self.misalign_zernikes(data, misalignment)
        # for i, row in data_reduced.iterrows():
        #     print('draw donut reduced misalignment to pixel', i, row)
        # for i, row in data_with_misalignments.iterrows():
        #     print('draw donut misalignment to pixel', i, row)
        # now draw the donuts
        stamps, data_reduced = super(Zernike_to_Misalignment_to_Pixel_Interpolator, self).draw_donut(data_reduced, **kwargs)

        return stamps, data_with_misalignments

def generate_random_coordinates(number):
    x = np.random.random(number) * 2048
    y = np.random.random(number) * 4096
    extnum = np.random.randint(1, 62, number)
    x, y = decaminfo().getPosition_extnum(extnum, x, y)
    return x, y, extnum

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

def zernike_corrections_from_hexapod(dz=0, dx=0, dy=0, xt=0, yt=0):
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

def misalign_optics(misalign_dict):
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
        elif (key[0] == 'z'):
            znum = int(key[1:].split('d')[0].split('x')[0].split('y')[0]) - 1
            ztype = key[-1]
            ztype_num = ztype_dict[ztype]
            zernike_correction[znum][ztype_num] = \
                zernike_correction[znum][ztype_num] + entry

    hexapod_correction = \
        zernike_corrections_from_hexapod(*hexapod)
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

    return zernike_dictionary

def misalign_zernikes(data, misalignment={}):
    # returns COPY of data with ONLY the following keys:
    # [x, y, rzero, z1, z2, ... z11]
    keys = ['x', 'y', 'rzero'] + ['z{0}'.format(i) for i in xrange(1, 12)]
    # filter out keys not present
    keys = [key for key in keys if key in data.keys()]

    # if misalignment is empty trust that the keys are in data
    if len(misalignment) == 0:
        misalignment = data

    # make copy of data so we don't modify in place
    x = data['x']
    y = data['y']
    rzero = data['rzero']
    df = {'x': x, 'y': y, 'rzero': rzero}

    # modify the zernikes
    for z in xrange(1, 12):
        key = 'z{0}'.format(z)
        correction = misalignment[key + 'd'] + \
                     misalignment[key + 'y'] * x + \
                     misalignment[key + 'x'] * y

        if key not in data:
            df[key] = correction
        else:
            df[key] = data[key] + correction

    return pd.DataFrame(df)

def correct_dz(dz):
    wavefac = 172.  # waves / mm
    return dz / wavefac

def correct_dz_theta(dz):
    numfac = 0.0048481  # rad / arcsec um/mm
    wavefac = 172.  # waves / mm
    return dz * numfac / wavefac

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
