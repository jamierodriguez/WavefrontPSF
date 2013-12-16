from __future__ import print_function, division
import numpy as np


def moments(data, indices=None):
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

    tot = data.sum()
    if not indices:
        Y, X = np.indices(data.shape)  # data is Y,X!!
    else:
        Y, X = indices
    x = np.sum(X * data) / tot
    y = np.sum(Y * data) / tot
    #x2 = np.sum(X * X * data) / tot - x * x
    #y2 = np.sum(Y * Y * data) / tot - y * y
    #xy = np.sum(X * Y * data) / tot - x * y

    return x, y


def centered_moment(data, w=1, q=0, p=0, centroid=None, indices=None):
    """Calculate the centered moment
    Mpq = sum(w * data * (X - x) ** p * (Y - y) ** q) / sum(w * data)
    where x and y are the centroid locations and X and Y are the indices of the
    data array

    Parameters
    ----------
    data : array
        2d image array

    w : array, optional
        weight array. Gives the weight at each pixel.
        Defaults to 1: unweighted

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
    """

    if not indices:
        Y, X = np.indices(data.shape)  # data is Y,X!!
    else:
        Y, X = indices
    if not centroid:
        tot = np.sum(w * data)
        x = np.sum(X * data) / tot
        y = np.sum(Y * data) / tot
    else:
        y, x = centroid
    Mpq = np.sum(w * data * (X - x) ** p * (Y - y) ** q) / \
        np.sum(w * data)
    return Mpq


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
        Y, X = np.indices(data.shape)  # data is Y,X!!
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

    # filter by threshold
    if thresh == -1:
        thresh = np.max(data) / 5.0
    conds = (data > thresh)
    d2 = d2[conds]

    # also estimate background via annulus around edge
    if background == 0:
        databack = np.append(data[0, :], data[-1, :])
        databack = np.append(databack, data[:, 0])
        databack = np.append(databack, data[:, -1])
        background = np.median(databack)
    data_use = data[conds] - background
    # add additional filter for data_use < 0
    conds_negative = (data_use > 0)
    data_use = data_use[conds_negative.tolist()]
    d2 = d2[conds_negative.tolist()]
    ## if np.sum(conds_negative) != len(conds_negative):
    ##     print('Error in background calculation. Some points ended up' +
    ##           'negative!')

    lpix = np.log(data_use)  # this is the y parameter
    # in sextractor, inverr2 = data**2, which is weird since the error should
    # go as err propto 1/sqrt(data), or in other words inverr2 = data. maybe
    # emanuel meant err as in var = std^2 instead of err as std... but then
    # this doesn't fit with the least squares regression.
    inverr2 = data_use   * data_use  # otherwise known as weight w

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

    # undersample correction; this is from the sextractor code and is
    # apparently an empirical correction
    if fwhm > 0.5:
        fwhm -= 1 / (4 * fwhm)

    return fwhm


def gaussian_window(data, centroid=None, indices=None, background=0,
                    thresh=-1):
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

    thresh : float, optional
        Threshold value in the data array for pixels to consider in this fit.
        If no value is specified, then it is set to be max(data) / 5 .

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
        Y, X = np.indices(data.shape)  # data is Y,X!!
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

    # get the scale for the window
    d502 = np.square(FWHM(data, centroid=[y, x], indices=[Y, X],
                          background=background, thresh=thresh))
    swin2 = d502 / (8 * np.log(2))
    # make the window
    w = np.exp(-0.5 * (np.square(X - x) + np.square(Y - y)) / swin2)

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

    See Also
    --------
    Sextractor manual.

    """

    # get unwindowed moments and indices
    if not indices:
        Y, X = np.indices(data.shape)  # data is Y,X!!
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
    minepsilon = 2.e-4
    i = 0
    epsilon = 1e5

    w = gaussian_window(data, centroid=[y, x], indices=[Y, X],
                        background=background, thresh=thresh)
    tot = np.sum(w * data)

    while (i < maxiter) * (epsilon > minepsilon):
        xp = x + 2 * np.sum(w * data * (X - x)) / tot
        yp = y + 2 * np.sum(w * data * (Y - y)) / tot

        epsilon = np.sqrt(np.square(xp - x) + np.square(yp - y))
        i = i + 1
        x, y = xp, yp

        # iterate window
        w = gaussian_window(data, centroid=[y, x], indices=[Y, X],
                            background=background, thresh=thresh)
        tot = np.sum(w * data)

    #print(epsilon, i)
    return y, x
