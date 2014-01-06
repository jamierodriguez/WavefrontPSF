from __future__ import print_function, division
import numpy as np
from scipy.optimize import leastsq

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

def centered_moment(data, q=0, p=0, centroid=None, indices=None):
    """Calculate the centered moment
    Mpq = sum(w * data * (X - x) ** p * (Y - y) ** q) / sum(w * data)
    where x and y are the centroid locations and X and Y are the indices of the
    data array

    Parameters
    ----------
    data : array
        2d image array

    p, q : floats, optional
        The moments we are interested in.
        Default for each is 0. (should return close to zero)

    centroid : list, optional
        A length 2 list with centroid[0] = y and centroid[1] = x.
        If this is not given, then the unweighted centroid is calculated.

    indices : length 2 list, optional
        length 2 list of 2d arrays Y and X such that Y and X indicate the
        index values (ie indices[0][i,j] = i, indices[1][i,j] = j)
        Default is None; constructs the indices with each call.

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
    Mpq = np.sum(data * (X - x) ** p * (Y - y) ** q) / \
        np.sum(data)
    return Mpq

def fit_gaussian(data, indices=None):
    """Fit a gaussian model plus background to the stamp

    Parameters
    ----------
    data : array
        2d image array

    Returns
    -------
    popt : list
        List of the fit parameters:
        [background, normalization, sigma2, centroid_y, centroid_x]

    """

    shape = data.shape
    if not indices:
        indices = np.indices(data.shape) + 0.5
    centroid = [shape_i / 2 for shape_i in shape]
    maxval2 = (shape[0] * 0.75) ** 2
    def model(p):
        if np.any(p) < 0:
            return 1e20
        elif np.any(np.square(p[2:5])) > maxval2:
            return 1e20
        else:

            w = gaussian_window(data, centroid=[p[3], p[4]],
                                indices=[indices[0], indices[1]],
                                background=p[0],
                                sigma2=np.square(p[2]) / (8 * np.log(2)))

            y = data
            fx = p[0] + p[1] * np.exp(-0.5 / p[2] * (np.square(indices[0] -
                                                               p[3]) +
                                                     np.square(indices[1] -
                                                               p[4])))
            return (w * (fx - y)).flatten()
    def dmodel(p):
        term = -0.5 / p[2] * (np.square(indices[0] - p[3]) +
                              np.square(indices[1] - p[4]))
        expterm = np.exp(term)
        length = shape[0] * shape[1]

        w = gaussian_window(data, centroid=[p[3], p[4]],
                            indices=[indices[0], indices[1]],
                            background=p[0],
                            sigma2=np.square(p[2]) / (8 * np.log(2)))

        dmdp0 = (w * 1).flatten()
        dmdp1 = (w * expterm).flatten()
        dmdp2 = (w * expterm * p[1] * term * -1 / p[2]).flatten()
        dmdp3 = (w * expterm * p[1] * (indices[0] - p[3]) / p[2]).flatten()
        dmdp4 = (w * expterm * p[1] * (indices[1] - p[4]) / p[2]).flatten()
        return [dmdp0, dmdp1, dmdp2, dmdp3, dmdp4]

    #popt, ier = \
    popt,cov,infodict,mesg,ier = \
        leastsq(model, x0=[0, 100, 4.,
                           centroid[0], centroid[1]],
                       col_deriv=1,
                       Dfun=dmodel,
                       maxfev=35,
                       full_output=True)
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
    if (ier > 4) + (not np.any(np.isfinite(popt))) + (np.any(popt < 0)):
        y, x, fwhm = windowed_centroid(data)
        popt = [0, 1, np.square(fwhm) / (8 * np.log(2)), y, x]
        # print('p prime')
        # print(popt)


    return popt

def FWHM(data, centroid=None, indices=None, background=0, thresh=-1):
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

    background : float, optional
        Background data value. This is subtracted out.
        If not given, estimate by an annulus around the edge of the image.

    thresh : float, optional
        Threshold value in the data array for pixels to consider in this fit.
        If no value is specified, then it is set to be max(data) / 5 .

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

    # also estimate background via annulus around edge
    if background == 0:
        databack = np.append(data[0, :], data[-1, :])
        databack = np.append(databack, data[:, 0])
        databack = np.append(databack, data[:, -1])
        background_median = np.median(databack)
        background_mean = np.mean(databack)
        background = background_median


    #print(background)

    data_use = data - background

    # get x_i = r^2
    dx = X - x
    dy = Y - y
    d2 = dx * dx + dy * dy

    # filter by threshold
    if thresh == -1:
        thresh = np.max(data - background) / 5.0
    conds = (data_use > thresh)
    d2 = d2[conds]
    data_use = data_use[conds]

    ## if np.sum(conds_negative) != len(conds_negative):
    ##     print('Error in background calculation. Some points ended up' +
    ##           'negative!')

    lpix = np.log(data_use)  # this is the y parameter
    # in sextractor, inverr2 = data**2, which is weird since the error should
    # go as err propto 1/sqrt(data), or in other words inverr2 = data. maybe
    # emanuel meant err as in var = std^2 instead of err as std... but then
    # this doesn't fit with the least squares regression.
    inverr2 = data_use * data_use  # otherwise known as weight w

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

    return fwhm


def gaussian_window(data, centroid=None, indices=None, background=0,
                    sigma2=None):
    """Calculate the gaussian window.

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

    background : float, optional
        Background data value. This is subtracted out.
        If not given, estimate by an annulus around the edge of the image.

    sigma2 : float, optional
        The size of the window.
        If no value is specified, then find it yourself.

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
        Y, X = np.indices(data.shape) + 0.5  # data is Y,X!!
    else:
        Y, X = indices
    centroid_guess = [data_shape_i / 2 for data_shape_i in data.shape]
    if not sigma2:
        fit_out = fit_gaussian(data, centroid=centroid_guess, indices=indices,
                               background=background)
        sigma2 = fit_out[2]
        centroid_guess = [fit_out[3], fit_out[4]]
        background = fit_out[0]
    if not centroid:
        y, x = centroid_guess
    else:
        y, x = centroid

    ## # calculate centroids if not given
    ## if not centroid:
    ##     tot = data.sum()
    ##     # get centroids
    ##     x = np.sum(X * data) / tot
    ##     y = np.sum(Y * data) / tot
    ## else:
    ##     # else, centroids are listed (y, x)
    ##     y, x = centroid

    ## # get the scale for the window
    ## if not sigma2:
    ##     d502, background = np.square(FWHM(data, centroid=[y, x], indices=[Y, X],
    ##                           background=background, thresh=thresh))
    ##     swin2 = d502 / (8 * np.log(2))
    ##     sigma2 = swin2
    # make the window
    r2 = (X - x) ** 2 + (Y - y) ** 2
    w = np.exp(-0.5 * r2 / sigma2)

    # TODO: the oversampling stuff from sextractor?
    # oversample some areas
    WINPOS_NSIG = 4
    raper = WINPOS_NSIG * np.sqrt(sigma2)
    ## WINPOS_OVERSAMP = 3
    ## scaley = scalex = 1.0 / WINPOS_OVERSAMP

    rintlim = raper - 0.75
    if rintlim < 0:
        rintlim2 = 0
    else:
        rintlim2 = rintlim ** 2
    rextlim2 = (raper + 0.75) ** 2
    locarea = np.zeros(w.shape)

    # if you are outside the window, set weight to zero
    xmin = x - raper
    xmax = x + raper
    ymin = y - raper
    ymax = y + raper

    w = np.where((X > xmin) * (X < xmax) * (Y > ymin) * (Y < ymax), w, 0)

    return w


def windowed_centroid(data, centroid=None, indices=None, background=0,
                      thresh=-1):
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

    background : float, optional
        Background data value. This is subtracted out.
        If not given, estimate by an annulus around the edge of the image.

    thresh : float, optional
        Threshold value in the data array for pixels to consider in this fit.
        If no value is specified, then it is set to be max(data) / 5 .

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
        tot = data.sum()
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


    d502 = np.square(FWHM(data, centroid=[y, x], indices=[Y, X],
                          background=background, thresh=thresh))
    swin2 = d502 / (8 * np.log(2))
    sigma2 = swin2
    w = gaussian_window(data, centroid=[y, x], indices=[Y, X],
                        background=background,
                        sigma2=sigma2)

    while (i < maxiter) * (epsilon > minepsilon):
        tot = np.sum(w * data)
        xp = x + 2 * np.sum(w * data * (X - x)) / tot
        yp = y + 2 * np.sum(w * data * (Y - y)) / tot

        epsilon = np.sqrt((np.square(xp - x) + np.square(yp - y)) / sigma2)
        i = i + 1
        x, y = xp, yp

        # iterate window
        # update sigma?
        ## sigma2 = np.square(FWHM(data, centroid=[y, x], indices=[Y, X],
        ##                         background=background, thresh=thresh))
        ## sigma2 /= (8 * np.log(2))
        w = gaussian_window(data, centroid=[y, x], indices=[Y, X],
                            background=background,
                            sigma2=sigma2)

    fwhm = np.sqrt(sigma2 * 8 * np.log(2))
    # make final fwhm calculation
    fwhm = FWHM(data, centroid=[y, x], indices=[Y, X],
                background=background, thresh=thresh)
    return y, x, fwhm
