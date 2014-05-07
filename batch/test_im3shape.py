#!/usr/bin/env python
"""
File: test_im3shape.py
Author: Chris Davis
Description: Compare my moments with im3shape and galsim


central assumption: my moment measurements are /all/ off by
    q_ij_mine = alpha q_ij_other
"""

from __future__ import print_function, division

import numpy as np
from os import path, makedirs, chdir, remove

from wavefront import Wavefront
from analytic_moments import analytic_data
from routines import convert_dictionary, MAD

from scipy.optimize import curve_fit

from matplotlib import cm
reds = cm.Reds
reds.set_bad('white')
reds.set_under('white')

blues_r = cm.Blues_r
blues_r.set_bad('white')
blues_r.set_under('white')

from py3shape.analyze import get_psf_params, get_FWHM
from py3shape.image import Image
import galsim

WF = Wavefront(nEle=1, background=0)

def stshow(stamp, cmap=reds):
    plt.figure()
    IM = plt.imshow(stamp, origin='lower', interpolation='nearest',
                    cmap=cmap)
    CB = plt.colorbar(IM)
    return

##############################################################################
# create zernike etc distribution
##############################################################################

N = 2000
zmax = 0.5
zmin = -zmax
zernikes = np.hstack((np.zeros((N, 3)),
                      np.random.random((N, 8)) * (zmax - zmin) + zmin))
rzeros = np.random.random(N) * (0.25 - 0.1) + 0.1
coords = np.zeros((N, 3))

##############################################################################
# expected moments from analytic and make image
##############################################################################

analytic = analytic_data(zernikes, rzeros)#, coords=coords)
plane = {'E1': analytic['e1'] / analytic['e0'],
         'E2': analytic['e2'] / analytic['e0'],
         }

##############################################################################
# make images
##############################################################################

stamps = []
for i in xrange(len(rzeros)):
    zernike = zernikes[i]
    rzero = rzeros[i]
    coord = coords[i]
    stamp = WF.stamp(zernike, rzero, coord)

    stamps.append(stamp)


wf = {'x0': [], 'y0': [], 'x2': [], 'xy': [], 'y2': []}
im3 = {'x0': [], 'y0': [], 'x2': [], 'xy': [], 'y2': [], 'e1': [], 'e2': []}
hsm = {'x0': [], 'y0': [], 'E1': [], 'E2': [], 'sigma': []}

for stamp_i in xrange(len(stamps)):
    stamp = stamps[stamp_i]

    ##############################################################################
    # measure moments
    ##############################################################################

    poles = WF.moments(stamp)
    # fwhm = (x2 y2 - xy xy) ** (1/4)
    # ie im3shape 'e0' is my e0 + 2 * fwhm ** 2
    results_wf = {'x0': poles['Mx'], 'y0': poles['My'],
                  'x2': poles['x2'], 'xy': poles['xy'],
                  'y2': poles['y2']}

    for key in wf.keys():
        wf[key].append(results_wf[key])

    ##############################################################################
    # measure moments with im3shape
    ##############################################################################

    psf = Image(stamp)
    #fwhm = get_FWHM(psf, fwxm=0.5, upsampling=1., radialProfile=True)
    moments_i = psf.weighted_moments(weight_radius=10., iterations=20)

    """
    attributes:
        qxx, qxy, qyy, x0, y0, e1, e2

    i3_flt ellipticity_denominator = qxx+qyy+2*i3_sqrt(qxx*qyy-qxy*qxy)
    moments->e1 = (qxx-qyy)/ellipticity_denominator;

    """
    results_im3shape = {'x0': moments_i.x0 - 0.5, 'y0': moments_i.y0 - 0.5,
                       'x2': moments_i.qxx, 'xy': moments_i.qxy,
                       'y2': moments_i.qyy,
                       'e1': moments_i.e1, 'e2': moments_i.e2}

    for key in im3.keys():
        im3[key].append(results_im3shape[key])

    ##############################################################################
    # measure moments with galsim
    ##############################################################################

    """
    After running the above code, `result.observed_shape` ["shape" = distortion, the
        (a^2 - b^2)/(a^2 + b^2) definition of ellipticity]

        reduced shear |g| = (a - b)/(a + b)
        distortion |e| = (a^2 - b^2)/(a^2 + b^2)
        conformal shear eta, with a/b = exp(eta)
        minor-to-major axis ratio q = b/a


    size sigma = (det M)^(1/4) from the adaptive moments, in units of pixels; -1 if
          not measured.

    detM = T^2 / 4 (1 - e_+^2 + e_x^2) = alpha (e0^2 - e1^2 - e2^2)

    but e_+ is is my e1 / e0

    Mxx = T/2 (1 + e+) = alpha / 2 (e0 + e1) etc...
    """

    psf_galsim = galsim.Image(stamp)
    result = galsim.hsm.FindAdaptiveMom(psf_galsim)
    results_galsim = {'x0': result.moments_centroid.x - 1,
                      'y0': result.moments_centroid.y - 1,
                      'E1': result.observed_shape.getE1(),
                      'E2': result.observed_shape.getE2(),
                      'sigma': result.moments_sigma}

    for key in hsm.keys():
        hsm[key].append(results_galsim[key])


##############################################################################
# make comparisons by manipulating wf
##############################################################################
# ellipticity_denominator = qxx+qyy+2*i3_sqrt(qxx*qyy-qxy*qxy)
for key in wf:
    wf[key] = np.array(wf[key])
ellipicity_denominator = wf['x2'] + wf['y2'] + \
                         2 * np.sqrt(wf['x2'] * wf['y2'] - wf['xy'] ** 2)
wf.update({'e1': (wf['x2'] - wf['y2']) / ellipicity_denominator,
           'e2': 2 * wf['xy'] / ellipicity_denominator,
           'E1': (wf['x2'] - wf['y2']) / (wf['x2'] + wf['y2']),
           'E2': 2 * wf['xy'] / (wf['x2'] + wf['y2']),
           'sigma': np.sqrt(np.sqrt(4 * wf['x2'] * wf['y2']
                            - 4 * wf['xy'] ** 2)),
           })


##############################################################################
# convert dictionaries to recarrays
##############################################################################
wf = convert_dictionary(wf)
im3 = convert_dictionary(im3)
hsm = convert_dictionary(hsm)
plane = convert_dictionary(plane)


##############################################################################
# Do linear fits
##############################################################################

def line(x, m, b):
    return m * x + b

def line_by_func(x, y, bins=50, func=np.median):
    """
    trick here is this:
    for each x bin, take median value of y!
    """
    H, xedges, yedges = np.histogram2d(x, y, bins)
    x_out = []
    y_out = []
    y_sigma = []
    for xi in range(len(xedges) - 1):
        xedges_lower = xedges[xi]
        xedges_upper = xedges[xi + 1]
        conds = (x > xedges_lower) * (x < xedges_upper)
        xs = x[conds]
        ys = y[conds]
        x_out.append(func(xs))
        y_out.append(func(ys))
        y_sigma.append(np.sqrt(func(np.square(ys - func(ys)))))
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    y_sigma = np.array(y_sigma)

    # tune y_sigma
    conds = np.where((y_sigma == y_sigma) * (y_sigma != 0))
    y_sigma = y_sigma[conds]
    x_out = x_out[conds]
    y_out = y_out[conds]

    return x_out, y_out, y_sigma


##############################################################################
# make graph outputs
##############################################################################

out_dir = ''
SAVE = False

bins = np.int(np.sqrt(N) / 2)
bins = 50
for key in sorted(im3.dtype.names):
    plt.figure()
    outs = plt.hist2d(im3[key], wf[key], bins=bins, cmap=reds)
    plt.xlabel('im3shape ' + key)
    plt.ylabel('WavefrontPSF ' + key)
    plt.colorbar()

    xs = np.linspace(outs[1][0], outs[1][-1], 50)
    plt.plot(xs, xs, 'b--', linewidth=4)


    xfit = im3[key]
    yfit = wf[key]
    ysigma = 1

    xfit, yfit, ysigma = line_by_func(xfit, yfit, bins=bins)
    plt.errorbar(xfit, yfit, yerr=ysigma, fmt='ko',)

    popt, pcov = curve_fit(line, xfit, yfit, sigma=ysigma)
    plt.plot(xs, line(xs, *popt), 'k-', linewidth=2)
    plt.title('{0:.2e}, {1:.2e}'.format(*popt))

    if SAVE:
        plt.savefig(out_dir + 'im3_{0}.png'.format(key))

for key in sorted(hsm.dtype.names):
    plt.figure()
    outs = plt.hist2d(hsm[key], wf[key], bins=bins, cmap=reds)
    plt.xlabel('hsm ' + key)
    plt.ylabel('WavefrontPSF ' + key)
    plt.colorbar()

    xs = np.linspace(outs[1][0], outs[1][-1], 50)
    plt.plot(xs, xs, 'b--', linewidth=4)

    xfit = hsm[key]
    yfit = wf[key]
    ysigma = 1

    xfit, yfit, ysigma = line_by_func(xfit, yfit, bins=bins)
    plt.errorbar(xfit, yfit, yerr=ysigma, fmt='ko',)

    popt, pcov = curve_fit(line, xfit, yfit, sigma=ysigma)
    plt.plot(xs, line(xs, *popt), 'k-', linewidth=2)
    plt.title('{0:.2e}, {1:.2e}'.format(*popt))

    if SAVE:
        plt.savefig(out_dir + 'hsm_{0}.png'.format(key))

for key in sorted(plane.dtype.names):
    plt.figure()
    outs = plt.hist2d(plane[key], wf[key], bins=bins, cmap=reds)
    plt.xlabel('plane ' + key)
    plt.ylabel('WavefrontPSF ' + key)
    plt.colorbar()

    xs = np.linspace(outs[1][0], outs[1][-1], 50)
    plt.plot(xs, xs, 'b--', linewidth=4)

    xfit = plane[key]
    yfit = wf[key]
    ysigma = 1

    xfit, yfit, ysigma = line_by_func(xfit, yfit, bins=bins)
    plt.errorbar(xfit, yfit, yerr=ysigma, fmt='ko',)

    popt, pcov = curve_fit(line, xfit, yfit, sigma=ysigma)
    plt.plot(xs, line(xs, *popt), 'k-', linewidth=2)
    plt.title('{0:.2e}, {1:.2e}'.format(*popt))

    if SAVE:
        plt.savefig(out_dir + 'plane_{0}.png'.format(key))


