from __future__ import print_function
import numpy
'''
from  hexapodtoZernike import makeZString
import os
dz = -1680
dy = 0.
xt = 0.
yt = 0.
dx = -1500.
z5delta = 0.
z6delta = 0.
str = makeZString(dz,dx,dy,xt,yt,z5delta,z6delta)

and the str will be a string formatted the way you need for makemesh
(there is also a hexapodtoZernike method which returns the individual deltas
and thetas)
'''

def hexapodtoZernike_old(dx,dy,xt,yt,z5delta=0.0,z6delta=0.0):

    # input units are microns,arcsec

    # latest calibration matrix
    hexapodArray20121020 = numpy.array(((  0.00e+00 ,  1.07e+05 ,  4.54e+03 ,  0.00e+00),
                                        (  1.18e+05 , -0.00e+00 ,  0.00e+00 , -4.20e+03),
                                        ( -4.36e+04 ,  0.00e+00 ,  0.00e+00 , -8.20e+01),
                                        (  0.00e+00 ,  4.42e+04 , -8.10e+01 ,  0.00e+00) ))

    # take its inverse
    alignmentMatrix = numpy.matrix(hexapodArray20121020)
    aMatrixInv = alignmentMatrix.I

    # build column vector of the hexapod dof
    hexapodList = (dx,dy,xt,yt)
    hexapodColVec = numpy.matrix(hexapodList).transpose()

    # calculate Zernikes
    zernikeM = aMatrixInv * hexapodColVec
    zernikeVector = zernikeM.A

    aveZern5ThetaX = zernikeVector[0][0]
    aveZern6ThetaX = zernikeVector[1][0]
    z7delta = zernikeVector[2][0]
    z8delta = zernikeVector[3][0]

    # output values
    z5thetax = aveZern5ThetaX
    z6thetay = aveZern5ThetaX
    z6thetax = aveZern6ThetaX
    z5thetay = -aveZern6ThetaX

    # print out
    outstring = "[%.5f,%.5f,%.5f],[%.5f,%.5f,%.5f],[%.2f,0.,0.],[%.2f,0.,0.]" % (z5delta,z5thetax,z5thetay,z6delta,z6thetax,z6thetay,z7delta,z8delta)

    return outstring,z5thetax,z5thetay,z6thetax,z6thetay,z7delta,z8delta

def makeZString(dz,dx,dy,xt,yt,z5delta=0.,z6delta=0.):

    outstring,z5thetax,z5thetay,z6thetax,z6thetay,z7delta,z8delta  = hexapodtoZernike(dx,dy,xt,yt,z5delta,z6delta)

    allstring = "[[0,0,0],[0,0,0],[0,0,0],[%.1f,0.,0.],%s]" % (dz,outstring)
    print(allstring)

    return allstring

def hexapodtoZernike(dx,dy,xt,yt):

    # input units are microns,arcsec

    # latest calibration matrix
    hexapodArray20121020 = numpy.array(
            ((  0.00e+00 ,  1.07e+05 ,  4.54e+03 ,  0.00e+00),
             (  1.18e+05 , -0.00e+00 ,  0.00e+00 , -4.20e+03),
             ( -4.36e+04 ,  0.00e+00 ,  0.00e+00 , -8.20e+01),
             (  0.00e+00 ,  4.42e+04 , -8.10e+01 ,  0.00e+00)
            ))

    # take its inverse
    alignmentMatrix = numpy.matrix(hexapodArray20121020)
    aMatrixInv = alignmentMatrix.I

    # build column vector of the hexapod dof
    hexapodList = (dx,dy,xt,yt)
    hexapodColVec = numpy.matrix(hexapodList).transpose()

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
    hexapodArray20121020 = numpy.array(
            ((  0.00e+00 ,  1.07e+05 ,  4.54e+03 ,  0.00e+00),
             (  1.18e+05 , -0.00e+00 ,  0.00e+00 , -4.20e+03),
             ( -4.36e+04 ,  0.00e+00 ,  0.00e+00 , -8.20e+01),
             (  0.00e+00 ,  4.42e+04 , -8.10e+01 ,  0.00e+00)
            ))
    alignmentMatrix = numpy.matrix(hexapodArray20121020)

    # note that z05y ~ -z06x and z05x ~ z06y
    # so we average them
    aveZern5ThetaX = numpy.mean([z05x, z06y])
    aveZern6ThetaX = numpy.mean([-z05y, z06x])
    zernikeList = (aveZern5ThetaX, aveZern6ThetaX, z07d, z08d)
    zernikeColVec = numpy.matrix(zernikeList).transpose()

    hexapodM = alignmentMatrix * zernikeColVec
    hexapodVector = hexapodM.A

    dx = hexapodVector[0][0]
    dy = hexapodVector[1][0]
    xt = hexapodVector[2][0]
    yt = hexapodVector[3][0]

    return dx, dy, xt, yt

def extract_hexapod_from_zernike_corrections(zernike_corrections):
    z05x = zernike_corrections[4, 1]
    z05y = zernike_corrections[4, 2]
    z06x = zernike_corrections[5, 1]
    z06y = zernike_corrections[5, 2]
    z07d = zernike_corrections[6, 0]
    z08d = zernike_corrections[7, 0]
    dz = zernike_corrections[3, 0]

    dx, dy, xt, yt = zerniketoHexapod(z05x, z05y, z06x, z06y, z07d, z08d)

    return_dict = {'dz':dz, 'dx':dx, 'dy':dy, 'xt':xt, 'yt':yt}
    return return_dict


if __name__ == '__main__':
    print('dx = 1000', '\nz05x, z05y, z06x, z07d, z08d = \n', hexapodtoZernike(1000, 0, 0, 0))
    print('dy = 1000', '\nz05x, z05y, z06x, z07d, z08d = \n', hexapodtoZernike(0, 1000, 0, 0))
    print('xt = 100', '\nz05x, z05y, z06x, z07d, z08d = \n', hexapodtoZernike(0, 0, 100, 0))
    print('yt = 100', '\nz05x, z05y, z06x, z07d, z08d = \n', hexapodtoZernike(0, 0, 0, 100))

