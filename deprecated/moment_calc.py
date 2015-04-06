from __future__ import print_function, division
import numpy as np
from scipy.optimize import leastsq
from math import atan2, cos, sin, sqrt

"""
8 Jan 2014: I have decided the window function will be a gaussian with sigma =
            5 pixels

TODO: should I change that window size? How many sigma out should I go?
      If I make it such that the window cuts off at exactly 32/2 pixels out,
      how many sigma shoudl I make it?

TODO: implement im3shape method for getting fwhm and moments
    - radial profile
    - fwhm
    - weight method
    - moments
    - background levels and threshold?
"""

SIGMA_WINDOW = 5.

def centroid(data, indices=None):
    """Calculate first and second moments for a given dataset

    Parameters
    ----------

    data : array
        2d image array

    indices : length 2 list, optional
        length 2 list of 2d arrays Y and X such that Y and X indicate the
        index values (ie indices[0][i,j] = i, indices[1][i,j] = j)
        Default is None; constructs the indices with each call.

    Returns
    -------

    x, y : floats
        The unweighted first moment (or mean) of the image (ie the expectation
        value). This is otherwise known as the centroid.

    Notes
    -----
    Image convention is data[Y,X]

    See Also
    --------
    centered_moment
        Given the centroid and a weighting scheme, calculate a desired moment
        centered about the centroid.

    """

    tot = np.sum(data)
    if not indices:
        Y, X = np.indices(data.shape) + 0.5  # data is Y,X!!
    else:
        Y, X = indices
    x = np.sum(X * data) / tot
    y = np.sum(Y * data) / tot

    return x, y

def centered_moment(data, q=0, p=0, centroid=None, indices=None, amplitude=None):
    """Calculate the centered moment
    Mpq = sum(data * (X - x) ** p * (Y - y) ** q) / sum(data)
    where x and y are the centroid locations and X and Y are the indices of the
    data array

    Parameters
    ----------
    data : array
        2d image array

    p, q : floats, optional
        The moments we are interested in. p,q for x,y
        Default for each is 0. (should return close to zero)

    centroid : list, optional
        A length 2 list with centroid[0] = y and centroid[1] = x.
        If this is not given, then the unweighted centroid is calculated.

    indices : length 2 list, optional
        length 2 list of 2d arrays Y and X such that Y and X indicate the
        index values (ie indices[0][i,j] = i, indices[1][i,j] = j)
        Default is None; constructs the indices with each call.

    amplitude : float, optional
        sum(data) (in case you are calculating this multiple times)

    Returns
    -------
    Mpq : float
        The centered moment M_pq centered about the centroid

    Notes
    -----
    Image convention is data[Y,X]

    If you want to /find/ xbar, ybar, just set them to 0 and (p,q)=(1,0) or
    (0,1)

    If you want to include weights, just multiply your data by them.
    """

    if not indices:
        Y, X = np.indices(data.shape) + 0.5  # data is Y,X!!
    else:
        Y, X = indices
    if not centroid:
        x, y = centroid(data, indices=[Y, X])
    else:
        y, x = centroid
    if not amplitude:
        amplitude = np.sum(data)
    Mpq = np.sum(data * (X - x) ** p * (Y - y) ** q) / amplitude
    return Mpq

def fit_gaussian(data, indices=None, verbose=False):
    """Fit a gaussian model plus background to the stamp

    Parameters
    ----------
    data : array
        2d image array

    indices : length 2 list, optional
        length 2 list of 2d arrays Y and X such that Y and X indicate the
        index values (ie indices[0][i,j] = i, indices[1][i,j] = j)
        Default is None; constructs the indices with each call.

    verbose : Bool
        If true, return the fit parameters

    Returns
    -------
    popt : list
        List of the fit parameters:
        [background, normalization, sigma, centroid_y, centroid_x]
    """

    shape = data.shape
    if not indices:
        Y, X = np.indices(data.shape) + 0.5
        indices = [Y, X]
    centroid = [shape_i / 2 for shape_i in shape]
    if verbose:
        history = dict(model=[], value=[], y=[])

    def f(p, y, indices):
        fx = p[0] + p[1] * np.exp(-0.5 / p[2] ** 2 *
                    (np.square(indices[0] - p[3]) +
                     np.square(indices[1] - p[4])))
        return fx

    def model(p, y, indices):
        ## if np.any(p) < 0:
        ##     return_val = 1e20 * np.ones(data.size)
        ## elif ((p[3] > shape[0] * 0.75) + (p[3] < shape[0] * 0.25) +
        ##       (p[4] > shape[1] * 0.75) + (p[4] < shape[1] * 0.25)):
        ##     return_val = 1e20 * np.ones(data.size)
        ## else:
        if True:
            w = gaussian_window(shape, centroid=[p[3], p[4]],
                                indices=[indices[0], indices[1]],
                                sigma=SIGMA_WINDOW)
            fx = f(p, y, indices)
            return_val = (w * (fx - y)).flatten()

        if verbose:
            history['model'].append(p.copy())
            history['value'].append(return_val.sum() ** 2)
            history['y'].append(y.copy().mean())
        return return_val

    def dmodel(p, y, indices):
        # account for centroid in w

        w = gaussian_window(shape, centroid=[p[3], p[4]],
                            indices=[indices[0], indices[1]],
                            sigma=SIGMA_WINDOW)

        chi = -0.5 / p[2] ** 2 * (np.square(indices[0] - p[3]) +
                                  np.square(indices[1] - p[4]))
        expchi = np.exp(chi)

        dwdp3 = SIGMA_WINDOW ** -2 * (indices[0] - p[3])
        dwdp4 = SIGMA_WINDOW ** -2 * (indices[1] - p[4])

        dchidp2 = -2 * p[2] ** -1 * chi
        dchidp3 = p[2] ** -2 * (indices[0] - p[3])
        dchidp4 = p[2] ** -2 * (indices[1] - p[4])


        dmodeldp0 = (w * 1).flatten()
        dmodeldp1 = (w * expchi).flatten()
        dmodeldp2 = (w * p[1] * expchi * dchidp2).flatten()
        dmodeldp3 = ((p[0] + p[1] * expchi - y) * dwdp3 +
                    p[1] * w * expchi * dchidp3).flatten()
        dmodeldp4 = ((p[0] + p[1] * expchi - y) * dwdp4 +
                    p[1] * w * expchi * dchidp4).flatten()

        return [dmodeldp0, dmodeldp1, dmodeldp2, dmodeldp3, dmodeldp4]


    mindata = np.min(data)
    databack = np.append(data[0, :], data[-1, :])
    databack = np.append(databack, data[:, 0])
    databack = np.append(databack, data[:, -1])
    background = np.median(databack)
    thresh = np.max(data - background) / 5.0
    x0 = np.array([background, 2 * mindata, 1.0, centroid[0], centroid[1]])
    #popt, ier = \
    popt,cov,infodict,mesg,ier = \
        leastsq(model, x0=x0,
                       col_deriv=1,
                       Dfun=dmodel,
                       maxfev=35,
                       xtol=1e-5,
                       full_output=True,
                       args=(data, indices),
                       diag=[1e-3, 1e-3, 10, 1, 1])
    popt_fin = np.copy(popt)
    # print(popt)
    # print(ier)
    # correct fwhm with empirical correction
    # if popt[2] > 1:
    #     popt[2] -= 1. / (4 * popt[2])

    #print(popt)
    #print(pcov)

    # TODO: implement backup method using old ways in case this one fails...
    # basically, if you hit the cap in function calls, just move on and
    # calculate the old-fashioned way:
    failure = False
    if (ier > 4) + (not np.any(np.isfinite(popt))) + (np.any(popt < 0)):
        failure = True
        y, x = windowed_centroid(data,
                                 centroid=centroid,
                                 indices=indices,
                                 sigma=SIGMA_WINDOW)
        fwhm = FWHM(data, centroid=[y,x], indices=indices)
        popt = np.array([background, 2 * background,
                         fwhm / np.sqrt(8 * np.log(2)), y, x])
        # print('p prime')
        # print(popt)
    if verbose:
        fit_dict = dict(popt=popt,
                        cov=cov,
                        infodict=infodict,
                        mesg=mesg,
                        ier=ier,
                        model=model,
                        dmodel=dmodel,
                        f=f,
                        failure=failure,
                        history=history,
                        centroid=centroid,
                        background=background,
                        thresh=thresh,
                        x0=x0,
                        popt_fin=popt_fin)
        return popt, fit_dict
    else:
        return popt

def FWHM(data, centroid=None, indices=None):
    """Calculate the FWHM via least squares fit, weighted by (data value -
    background).

    Parameters
    ----------
    data : array
        2d image array

    centroid : list, optional
        A length 2 list with centroid[0] = y and centroid[1] = x.
        If this is not given, then the unweighted centroid is calculated.

    indices : length 2 list, optional
        length 2 list of 2d arrays Y and X such that Y and X indicate the
        index values (ie indices[0][i,j] = i, indices[1][i,j] = j)
        Default is None; constructs the indices with each call.

    Returns
    -------
    fwhm : float
        The full width half max of the data.

    Notes
    -----
    fit least squares to a 1d Gaussian

    .. math:: P_i = A \exp(-((x - xbar)^2 + (y - ybar)^2) / (2 \sigma^2))

    we can convert this to form :math:`y = a + b xi + ei`:

    .. math:: \ln P_i = \ln A + -1/(2 \sigma^2) ((x - xbar)^2 + (y - ybar)^2)

    where `xi` is now :math:`((x - xbar)^2 + (y - ybar)^2)`
    and `b` is :math:`-1 / (2 \sigma^2)`

    .. math::

        b = (\sum(w_i) \sum(w_i x_i y_i) - \sum(w_i x_i) \sum(w_i y_i)) /
            (\sum(w_i) \sum(w_i x_i x_i) - \sum(w_i x_i) \sum(w_i x_i))

        b = (s sxy - sx sy) / (s sxx - sx sx)

    """

    if not indices:
        Y, X = np.indices(data.shape) + 0.5  # data is Y,X!!
    else:
        Y, X = indices
    # calculate centroids if not given
    if not centroid:
        tot = data.sum()
        # get centroids
        x = np.sum(X * data) / tot
        y = np.sum(Y * data) / tot
    else:
        # else, centroids are listed (y, x)
        y, x = centroid

    # get x_i = r^2
    dx = X - x
    dy = Y - y
    d2 = dx * dx + dy * dy

    ## if np.sum(conds_negative) != len(conds_negative):
    ##     print('Error in background calculation. Some points ended up' +
    ##           'negative!')

    lpix = np.log(data)  # this is the y parameter
    # in sextractor, inverr2 = data**2, which is weird since the error should
    # go as err propto 1/sqrt(data), or in other words inverr2 = data. maybe
    # emanuel meant err as in var = std^2 instead of err as std... but then
    # this doesn't fit with the least squares regression.
    inverr2 = data * data  # otherwise known as weight w

    s = np.sum(inverr2)
    sx = np.sum(d2 * inverr2)
    sxx = np.sum(d2 * d2 * inverr2)
    sy = np.sum(lpix * inverr2)
    sxy = np.sum(lpix * d2 * inverr2)

    d = s * sxx - sx * sx
    b = (s * sxy - sx * sy) / d
    # b is the best fit to the slope of a least squares line

    # convert slope to FWHM
    fwhm = 1.6651 / np.sqrt(-b)

    ## # undersample correction; this is from the sextractor code and is
    ## # apparently an empirical correction
    ## if fwhm > 0.5:
    ##     fwhm -= 1 / (4 * fwhm)

    ## # check for nan's from pathological cases
    ## if not np.isfinite(fwhm):
    ##     #print('Warning! non-fininite fwhm calculated!')
    ##     fwhm = 4.  # some number; any number!

    return fwhm

def adaptive_second_moment_window(y, x,
                                  max_moment_nsig2=25.,
                                  My=8., Mx=8.,
                                  Myy=9., Mxy=0., Mxx=9.):
    """Calculate adaptive second moment window

    Parameters
    ----------
    x, y : array
        Array of x and y coordinate values.

    max_moment_nsig2 : float, optional
        Maximal radius for the window squared. Defaults to 25.

    Mx, My : float, optional
        Location of centroid of image.

    Mxx, Mxy, Myy : float, optional
        Second moment matrix used for window.

    Returns
    -------
    window : array
        Window of x.shape shape.

    """

    x_Mx = x - Mx
    y_My = y - My

    detM = Mxx * Myy - Mxy * Mxy
    Minv_xx = Myy / detM
    Minv_xy = -Mxy / detM
    Minv_yy = Mxx / detM

    rho2 = Minv_xx * np.square(x_Mx) + \
        2 * Minv_xy * x_Mx * y_My + \
        Minv_yy * np.square(y_My)

    window = np.where(rho2 <= max_moment_nsig2, np.exp(-0.5 * rho2), 0)

    return window

def adaptive_moments(data):
    """Calculate the adaptive moments ala Hirata et al 2004

    Parameters
    ----------
    data : array
        The image.

    Returns
    -------
    Mx, My : float
        Centroids

    Mxx, Mxy, Myy : float
        Second moment matrix

    amplitude : float
        Total windowed intensity sum(w data)

    window : array
        Window used for calculations.


    Not Implemented
    ---------------

    rho4 : float
        Radial fourth moment rho4 = 2(1 + a_4) which corresponds to the
        nongaussianity, such that a_4 approx 0.046 for kolmogorov turbulance.

    ## # Now that we have converged, find the final moments
    ## end_order_dict = {
    ##         'Mxxx': {'p':3, 'q':0},
    ##         'Mxxy': {'p':2, 'q':1},
    ##         'Mxyy': {'p':1, 'q':2},
    ##         'Myyy': {'p':0, 'q':3},
    ##         'Mxxxx': {'p':4, 'q':0},
    ##         'Mxxxy': {'p':3, 'q':1},
    ##         'Mxxyy': {'p':2, 'q':2},
    ##         'Mxyyy': {'p':1, 'q':3},
    ##         'Myyyy': {'p':0, 'q':4},
    ##         }
    ## # Convert 4th moments into rho4 Parameter

    Notes
    -----
    rho2 = (r-r0)Minv(r-r0)
    so rho4 = (x-x0)^4 Minv_xx^2 +
              (x-x0)^3 (y-y0) 4 Minv_xx Minv_xy +
              (x-x0)^2 (y-y0)^2 (2 Minv_xx Minv_yy + 4 Minv_xy^2)
              + obvious perms


    """
    shape = data.shape

    Y, X = np.indices(shape) + 0.5  # data is Y,X ; 0.5 ?

    # initial guesses
    Mx, My = np.array(shape) / 2
    Mxx, Mxy, Myy = 9., 0., 9.

    convergence_factor = 1.0
    bound_correct_wt = 0.25  # Maximum shift in centroids and sigma between
                             # iterations for adaptive moments.
    max_moment_nsig2 = 25.
    epsilon = 1e-6
    num_iter = 0
    max_num_iter = 1000

    # Iterate until we converge
    while (convergence_factor > epsilon) * (num_iter < max_num_iter):

        window = adaptive_second_moment_window(
                Y, X,
                max_moment_nsig2=max_moment_nsig2,
                Mx=Mx, My=My,
                Mxx=Mxx, Mxy=Mxy, Myy=Myy
                )
        amplitude = np.sum(window * data)

        # Compute configuration of the weight function
        two_psi = atan2(2 * Mxy, Mxx - Myy)
        semi_a2 = 0.5 * ((Mxx + Myy) + (Mxx - Myy) * cos(two_psi)) + \
                         Mxy * sin(two_psi)
        semi_b2 = Mxx + Myy - semi_a2

        if semi_b2 <= 0:
            print("Error: non positive-definite weight in adaptive_moments.\n")

        shiftscale = sqrt(semi_b2)
        if num_iter == 0:
            shiftscale0 = shiftscale

        # Compute moments
        Cx = centered_moment(
                window * data,
                centroid = [My, Mx],
                indices=[Y, X],
                amplitude=amplitude,
                p=1, q=0)
        Cy = centered_moment(
                window * data,
                centroid = [My, Mx],
                indices=[Y, X],
                amplitude=amplitude,
                p=0, q=1)
        Cxx = centered_moment(
                window * data,
                centroid = [My, Mx],
                indices=[Y, X],
                amplitude=amplitude,
                p=2, q=0)
        Cxy = centered_moment(
                window * data,
                centroid = [My, Mx],
                indices=[Y, X],
                amplitude=amplitude,
                p=1, q=1)
        Cyy = centered_moment(
                window * data,
                centroid = [My, Mx],
                indices=[Y, X],
                amplitude=amplitude,
                p=0, q=2)

        # Now compute changes to x0
        dx = 2. * Cx / (shiftscale)
        dy = 2. * Cy / (shiftscale)
        dxx = 4. * (Cxx - 0.5 * Mxx) / semi_b2
        dxy = 4. * (Cxy - 0.5 * Mxy) / semi_b2
        dyy = 4. * (Cyy - 0.5 * Myy) / semi_b2

        # Bound correction tests
        if (dx     >  bound_correct_wt): dx     =  bound_correct_wt
        if (dx     < -bound_correct_wt): dx     = -bound_correct_wt
        if (dy     >  bound_correct_wt): dy     =  bound_correct_wt
        if (dy     < -bound_correct_wt): dy     = -bound_correct_wt
        if (dxx    >  bound_correct_wt): dxx    =  bound_correct_wt
        if (dxx    < -bound_correct_wt): dxx    = -bound_correct_wt
        if (dxy    >  bound_correct_wt): dxy    =  bound_correct_wt
        if (dxy    < -bound_correct_wt): dxy    = -bound_correct_wt
        if (dyy    >  bound_correct_wt): dyy    =  bound_correct_wt
        if (dyy    < -bound_correct_wt): dyy    = -bound_correct_wt

        # Convergence tests
        convergence_factor = abs(dx) ** 2
        if (abs(dy) > convergence_factor):
            convergence_factor = abs(dy) ** 2
        if (abs(dxx) > convergence_factor):
            convergence_factor = abs(dxx)
        if (abs(dxy) > convergence_factor):
            convergence_factor = abs(dxy)
        if (abs(dyy) > convergence_factor):
            convergence_factor = abs(dyy)
        convergence_factor = sqrt(convergence_factor)
        if (shiftscale < shiftscale0):
            convergence_factor *= shiftscale0 / shiftscale

        # Update moments
        Mx += dx * shiftscale
        My += dy * shiftscale
        Mxx += dxx * semi_b2
        Mxy += dxy * semi_b2
        Myy += dyy * semi_b2

        num_iter += 1

    if num_iter > max_num_iter:
        print('Warning, maximum number of iterations', max_num_iter,
              'reached with at', num_iter,
              'with convergence factor', convergence_factor,
              'less than epsilon', epsilon)

    return Mx, My, Mxx, Mxy, Myy, amplitude, window


def gaussian_window(shape, centroid=None, indices=None,
                    sigma=SIGMA_WINDOW):
    """Calculate the gaussian window.

    Parameters
    ----------
    shape : tuple
        Shape of the 2d image array

    centroid : list, optional
        A length 2 list with centroid[0] = y and centroid[1] = x.
        If this is not given, then the unweighted centroid is calculated.

    indices : length 2 list, optional
        length 2 list of 2d arrays Y and X such that Y and X indicate the
        index values (ie indices[0][i,j] = i, indices[1][i,j] = j)
        Default is None; constructs the indices with each call.

    sigma : float, optional
        The size of the window. Default is SIGMA_WINDOW.

    Returns
    -------
    w : array
        Image array (same size of data) of weight at each pixel.

    Notes
    -----
    STD for the gaussian window is determined by least squares fit to image
    from function FWHM.

    """

    if not indices:
        Y, X = np.indices(shape) + 0.5  # data is Y,X!!
    else:
        Y, X = indices
    centroid_guess = [data_shape_i / 2 for data_shape_i in shape]
    if not centroid:
        y, x = centroid_guess
    else:
        y, x = centroid

    # make the window
    r2 = (X - x) ** 2 + (Y - y) ** 2
    w = np.exp(-0.5 * r2 / sigma ** 2)

    # TODO: the oversampling stuff from sextractor?
    # oversample some areas
    WINPOS_NSIG = 3
    raper = WINPOS_NSIG * sigma
    ## WINPOS_OVERSAMP = 3
    ## scaley = scalex = 1.0 / WINPOS_OVERSAMP

    ## rintlim = raper - 0.75
    ## if rintlim < 0:
    ##     rintlim2 = 0
    ## else:
    ##     rintlim2 = rintlim ** 2
    ## rextlim2 = (raper + 0.75) ** 2
    ## locarea = np.zeros(w.shape)

    # if you are outside the window, set weight to zero
    xmin = x - raper
    xmax = x + raper
    ymin = y - raper
    ymax = y + raper

    w = np.where((X > xmin) * (X < xmax) * (Y > ymin) * (Y < ymax), w, 0)

    return w


def windowed_centroid(data, centroid=None, indices=None, sigma=SIGMA_WINDOW):
    """calculate windowed centroid ala sextractor WIN parameter with iterative
    fit

    Parameters
    ----------
    data : array
        2d image array

    centroid : list, optional
        A length 2 list with centroid[0] = y and centroid[1] = x.
        This is an initial guess.

    indices : length 2 list, optional
        length 2 list of 2d arrays Y and X such that Y and X indicate the
        index values (ie indices[0][i,j] = i, indices[1][i,j] = j)
        Default is None; constructs the indices with each call.

    sigma : float, optional
        Size of the window used. Default is SIGMA_WINDOW

    Returns
    -------
    y, x : float
        Windowed centroid coordinates.

    Notes
    -----
    Unlike from what I can tell in the sextractor manual, I recalculate the
    gaussian window at each iteration with the new centroid. (I don't think
    sextractor does this.) This actually speeds up the convergence such that
    the overall process is actually faster.
    16.12.13: This is no longer true; see commented segment below if you wish
    to turn it back on.

    See Also
    --------
    Sextractor manual.

    """

    # get unwindowed moments and indices
    if not indices:
        Y, X = np.indices(data.shape) + 0.5  # data is Y,X!!
    else:
        Y, X = indices
    # calculate centroids if not given
    if not centroid:
        tot = np.sum(data)
        # get centroids
        x = np.sum(X * data) / tot
        y = np.sum(Y * data) / tot
    else:
        # else, centroids are listed (y, x)
        y, x = centroid

    # calculate windowed centroid via iterative process
    maxiter = 100
    #minepsilon = 2.e-4
    minepsilon = 2.e-9
    i = 0
    epsilon = 1e5

    w = gaussian_window(data.shape, centroid=[y, x], indices=[Y, X],
                        sigma=sigma)

    while (i < maxiter) * (epsilon > minepsilon):
        tot = np.sum(w * data)
        xp = x + 2 * np.sum(w * data * (X - x)) / tot
        yp = y + 2 * np.sum(w * data * (Y - y)) / tot

        epsilon = np.sqrt((np.square(xp - x) + np.square(yp - y))) / sigma
        i = i + 1
        x, y = xp, yp

        # iterate window
        w = gaussian_window(data.shape, centroid=[y, x], indices=[Y, X],
                            sigma=sigma)

    return y, x
