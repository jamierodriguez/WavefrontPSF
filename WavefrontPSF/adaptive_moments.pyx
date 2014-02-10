from __future__ import print_function, division
from math import atan2, cos, sin
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
from libc.math cimport sqrt, ceil, floor, exp

"""
Adaptation of GalSim's adaptive moments code, which is itself an adaptation of
some other code.

441 mus vs 338 mus for galsim python interface + cpp code (and I only cythonized ellipmom_1 !)
but: that is for a weaker epsilon of 1e-3. 1e-4 makes it take much longer (hits num_iter max)

TODO: why is it that I need that /2 in my ellipticity calculations to
recover roughly sextractor's ellipticity values? (for when I switch from
moments to ellipticities...) BUT not in e0!
"""

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

"A parameter for optimizing calculations of adaptive moments by\n"
"cutting off profiles. This parameter is used to decide how many\n"
"sigma^2 into the Gaussian adaptive moment to extend the moment\n"
"calculation, with the weight being defined as 0 beyond this point.\n"
"i.e., if max_moment_nsig2 is set to 25, then the Gaussian is\n"
"extended to (r^2/sigma^2)=25, with proper accounting for elliptical\n"
"geometry.  If this parameter is set to some very large number, then\n"
"the weight is never set to zero and the exponential function is\n"
"always called. Note: GalSim script devel/modules/test_mom_timing.py\n"
"was used to choose a value of 25 as being optimal, in that for the\n"
"cases that were tested, the speedups were typically factors of\n"
"several, but the results of moments and shear estimation were\n"
"changed by <10^-5.  Not all possible cases were checked, and so for\n"
"use of this code for unusual cases, we recommend that users check\n"
"that this value does not affect accuracy, and/or set it to some\n"
"large value to completely disable this optimization.\n"
cdef double MAX_MOMENT_NSIG2 = 25.

# cpdef NOT cdef !!
cpdef np.ndarray[DTYPE_t, ndim=1] find_ellipmom_1(
        np.ndarray[DTYPE_t, ndim=2] data,
        double Mx, double My,
        double Mxx, double Mxy, double Myy):
    """C routine to obtain estimates of corrections to Mxx etc. Taken from
    galsim."""


    cdef DTYPE_t A = 0
    cdef DTYPE_t Bx = 0
    cdef DTYPE_t By = 0
    cdef DTYPE_t Cxx = 0
    cdef DTYPE_t Cxy = 0
    cdef DTYPE_t Cyy = 0
    cdef DTYPE_t Cxxx = 0
    cdef DTYPE_t Cxxy = 0
    cdef DTYPE_t Cxyy = 0
    cdef DTYPE_t Cyyy = 0
    cdef DTYPE_t rho4w = 0

    cdef int xmin = 0
    cdef int ymin = 0
    cdef int ymax = data.shape[0]
    cdef int xmax = data.shape[1]

    cdef double detM = Mxx * Myy - Mxy * Mxy
    if (detM <= 0) + (Mxx <= 0) + (Myy <= 0):
        print("Error: non positive definite adaptive moments!\n")

    cdef double Minv_xx = Myy / detM
    cdef double TwoMinv_xy = -Mxy / detM * 2.0
    cdef double Minv_yy = Mxx / detM
    cdef double Inv2Minv_xx = 0.5 / Minv_xx  # Will be useful later...

    # rho2 = Minv_xx(x-Mx)^2 + 2Minv_xy(x-Mx)(y-My) + Minv_yy(y-My)^2
    # The minimum/maximum y that have a solution rho2 = max_moment_nsig2 is at:
    #   2*Minv_xx*(x-Mx) + 2Minv_xy(y-My) = 0
    # rho2 = Minv_xx (Minv_xy(y-My)/Minv_xx)^2
    #           - 2Minv_xy(Minv_xy(y-My)/Minv_xx)(y-My)
    #           + Minv_yy(y-My)^2
    #      = (Minv_xy^2/Minv_xx - 2Minv_xy^2/Minv_xx + Minv_yy) (y-My)^2
    #      = (Minv_xx Minv_yy - Minv_xy^2)/Minv_xx (y-My)^2
    #      = (1/detM) / Minv_xx (y-My)^2
    #      = (1/Myy) (y-My)^2
    #
    # we are finding the limits for the iy values and then the ix values.
    cdef double y_My = sqrt(MAX_MOMENT_NSIG2 * Myy)
    cdef double y1 = -y_My + My
    cdef double y2 = y_My + My

    # stay within image bounds
    cdef int iy1 = max(int(ceil(y1)), ymin)
    cdef int iy2 = min(int(floor(y2)), ymax)
    cdef int y

    if iy1 > iy2:
        print('iy1 > iy2', y1, ymin, y2, ymax, iy1, iy2)

    cdef double a, b, c, d, sqrtd, inv2a, x1, x2, x_Mx, \
        Minv_xx__x_Mx__x_Mx, rho2, intensity, TwoMinv_xy__y_My, Minv_yy__y_My__y_My
    cdef int ix1, ix2, x

    for y in xrange(iy1, iy2):

        y_My = float(y) - My
        TwoMinv_xy__y_My = TwoMinv_xy * y_My
        Minv_yy__y_My__y_My = Minv_yy * y_My ** 2

        # Now for a particular value of y, we want to find the min/max x that satisfy
        # rho2 < max_moment_nsig2.
        #
        # 0 = Minv_xx(x-Mx)^2 + 2Minv_xy(x-Mx)(y-My) + Minv_yy(y-My)^2 - max_moment_nsig2
        # Simple quadratic formula:

        a = Minv_xx
        b = TwoMinv_xy__y_My
        c = Minv_yy__y_My__y_My - MAX_MOMENT_NSIG2
        d = b * b - 4 * a * c
        sqrtd = sqrt(d)
        inv2a = Inv2Minv_xx
        x1 = inv2a * (-b - sqrtd) + Mx
        x2 = inv2a * (-b + sqrtd) + Mx

        # stay within image bounds
        ix1 = max(int(ceil(x1)), xmin)
        ix2 = min(int(floor(x2)), xmax)
        # in the following two cases, ask if we somehow wanted to find
        # pixels outside the image
        if (ix1 > xmax) * (ix2 == xmax):
            continue
        elif (ix1 == xmin) * (ix2 < xmin):
            continue
        elif ix1 > ix2:
            # print('ix1 > ix2', y, x1, xmin, x2, xmax, ix1, ix2)
            # usually what happens is you want to take only one pixel and you
            # end up due to the ceil and floor funcs with e.g. 15, 14 instead
            # of 14, 15
            # ix1, ix2 = ix2, ix1
            # ix1 = max(ix1, xmin)
            # ix2 = min(ix2, xmax)
            continue

        for x in xrange(ix1, ix2):

            x_Mx = float(x) - Mx

            # Compute displacement from weight centroid, then get elliptical
            # radius and weight.
            Minv_xx__x_Mx__x_Mx = Minv_xx * x_Mx ** 2
            rho2 = Minv_yy__y_My__y_My + \
                TwoMinv_xy__y_My * x_Mx + \
                Minv_xx__x_Mx__x_Mx

            # this shouldn't happen by construction
            if (rho2 > MAX_MOMENT_NSIG2 + 1e8):
                print('rho2 > max_moment_nsig2 !')
                continue

            intensity = exp(-0.5 * rho2) * data[y, x]  # y,x order!

            A += intensity
            Bx += intensity * x_Mx
            By += intensity * y_My
            Cxx += intensity * x_Mx ** 2
            Cxy += intensity * x_Mx * y_My
            Cyy += intensity * y_My ** 2
            Cxxx += intensity * x_Mx ** 3
            Cxxy += intensity * x_Mx ** 2 * y_My
            Cxyy += intensity * x_Mx * y_My ** 2
            Cyyy += intensity * y_My ** 3
            rho4w += intensity * rho2 * rho2

    cdef np.ndarray[DTYPE_t, ndim=1] return_array = \
        np.array([A, Bx, By, Cxx, Cxy, Cyy, rho4w,
                  Cxxx, Cxxy, Cxyy, Cyyy
                  ], dtype=DTYPE)
    return return_array

def adaptive_moments(data):

    epsilon = 1e-6
    convergence_factor = 1.0
    guess_sig = 3.0
    bound_correct_wt = 0.25  # Maximum shift in centroids and sigma between
                             # iterations for adaptive moments.
    num_iter = 0
    num_iter_max = 100

    # Set Amp = -1000 as initial value just in case the while() block below is
    # never triggered; in this case we have at least *something* defined to
    # divide by, and for which the output will fairly clearly be junk.
    Amp = -1000.
    Mx = data.shape[1] / 2.
    My = data.shape[0] / 2.
    Mxx = guess_sig ** 2
    Mxy = 0
    Myy = guess_sig ** 2

    # Iterate until we converge
    while (convergence_factor > epsilon) * (num_iter < num_iter_max):
        # print(Mx, My, Mxx, Mxy, Myy, num_iter)

        # Get moments
        Amp, Bx, By, Cxx, Cxy, Cyy, rho4, \
            Cxxx, Cxxy, Cxyy, Cyyy \
            = find_ellipmom_1(data, Mx, My, Mxx, Mxy, Myy)
        # print(num_iter)
        # print(Amp, Bx / Amp, By / Amp, Cxx / Amp, Cxy / Amp, Cyy / Amp, rho4 / Amp)
        # print(Mx, My, Mxx, Mxy, Myy)
        # Compute configuration of the weight function
        two_psi = atan2(2 * Mxy, Mxx - Myy)
        semi_a2 = 0.5 * ((Mxx + Myy) + (Mxx - Myy) * cos(two_psi)) + \
                         Mxy * sin(two_psi)
        semi_b2 = Mxx + Myy - semi_a2

        if semi_b2 <= 0:
            print("Error: non positive-definite weight in find_ellipmom_2.\n")

        shiftscale = sqrt(semi_b2)
        if num_iter == 0:
            shiftscale0 = shiftscale

        # Now compute changes to Mx, etc
        dx = 2. * Bx / (Amp * shiftscale)
        dy = 2. * By / (Amp * shiftscale)
        dxx = 4. * (Cxx / Amp - 0.5 * Mxx) / semi_b2
        dxy = 4. * (Cxy / Amp - 0.5 * Mxy) / semi_b2
        dyy = 4. * (Cyy / Amp - 0.5 * Myy) / semi_b2

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
        convergence_factor = np.sqrt(convergence_factor)
        if (shiftscale < shiftscale0):
            convergence_factor *= shiftscale0 / shiftscale

        # Now update moments
        Mx += dx * shiftscale
        My += dy * shiftscale
        Mxx += dxx * semi_b2
        Mxy += dxy * semi_b2
        Myy += dyy * semi_b2

        num_iter += 1

    A = Amp
    rho4 /= Amp
    x2 = Cxx / Amp
    xy = Cxy / Amp
    y2 = Cyy / Amp
    x3 = Cxxx / Amp
    x2y = Cxxy / Amp
    xy2 = Cxyy / Amp
    y3 = Cyyy / Amp

    return Mx, My, Mxx, Mxy, Myy, A, rho4, x2, xy, y2, x3, x2y, xy2, y3


cpdef double centered_moment(
        np.ndarray[DTYPE_t, ndim=2] data,
        int p, int q,
        double Mx, double My,
        double Mxx, double Mxy, double Myy):

    """Calculate centered moments

    Parameters
    ----------
    data : array
        2d image array

    p, q : int
        The moments we are interested in. p,q for x,y

    Mx, My : floats
        The centroids of the image.

    Mxx, Mxy, Myy : floats
        The second moment weight matrix. See the notes below on the extra
        factor of two compared to the actual centered moment.

    Returns
    -------
    mu_pq : float
        The centered adaptive moment.

    Notes
    -----
    Because of the 2 in eq (5) of Hirata et al 2004, the second moment Ms in
    the input are /twice/ the value of the centered moments I would obtain
    otherwise. When you measure ellipticities it generally doesn't matter since
    you normalize by Mxx + Myy, thus cancelling the factor of two. In my
    analysis, however, I have kept these separate so this is good to
    understand.

    ie Mxx = 2 <x^2>, Mxy = 2 <xy>, Myy = 2 <y^2>
    but Mx = <x>, My = <y>

    consequently the T/2 in (4) = e0 = <x^2> + <y^2>

    """

    cdef double A = 0
    cdef double mu_pq = 0

    cdef int xmin = 0
    cdef int ymin = 0
    cdef int ymax = data.shape[0]
    cdef int xmax = data.shape[1]

    cdef double detM = Mxx * Myy - Mxy * Mxy
    if (detM <= 0) + (Mxx <= 0) + (Myy <= 0):
        print("Error: non positive definite adaptive moments!\n")

    cdef double Minv_xx = Myy / detM
    cdef double TwoMinv_xy = -Mxy / detM * 2.0
    cdef double Minv_yy = Mxx / detM
    cdef double Inv2Minv_xx = 0.5 / Minv_xx  # Will be useful later...

    # rho2 = Minv_xx(x-Mx)^2 + 2Minv_xy(x-Mx)(y-My) + Minv_yy(y-My)^2
    # The minimum/maximum y that have a solution rho2 = max_moment_nsig2 is at:
    #   2*Minv_xx*(x-Mx) + 2Minv_xy(y-My) = 0
    # rho2 = Minv_xx (Minv_xy(y-My)/Minv_xx)^2
    #           - 2Minv_xy(Minv_xy(y-My)/Minv_xx)(y-My)
    #           + Minv_yy(y-My)^2
    #      = (Minv_xy^2/Minv_xx - 2Minv_xy^2/Minv_xx + Minv_yy) (y-My)^2
    #      = (Minv_xx Minv_yy - Minv_xy^2)/Minv_xx (y-My)^2
    #      = (1/detM) / Minv_xx (y-My)^2
    #      = (1/Myy) (y-My)^2
    #
    # we are finding the limits for the iy values and then the ix values.
    cdef double y_My = sqrt(MAX_MOMENT_NSIG2 * Myy)
    cdef double y1 = -y_My + My
    cdef double y2 = y_My + My

    # stay within image bounds
    cdef int iy1 = max(int(ceil(y1)), ymin)
    cdef int iy2 = min(int(floor(y2)), ymax)
    cdef int y
    if iy1 > iy2:
        print('iy1 > iy2', y1, ymin, y2, ymax, iy1, iy2)

    cdef double a, b, c, d, sqrtd, inv2a, x1, x2, x_Mx, \
        Minv_xx__x_Mx__x_Mx, rho2, intensity, \
        TwoMinv_xy__y_My, Minv_yy__y_My__y_My, \
        y_My_q
    cdef int ix1, ix2, x

    for y in xrange(iy1, iy2):

        y_My = float(y) - My
        y_My_q = y_My ** q

        TwoMinv_xy__y_My = TwoMinv_xy * y_My
        Minv_yy__y_My__y_My = Minv_yy * y_My ** 2

        # Now for a particular value of y, we want to find the min/max x that satisfy
        # rho2 < max_moment_nsig2.
        #
        # 0 = Minv_xx(x-Mx)^2 + 2Minv_xy(x-Mx)(y-My) + Minv_yy(y-My)^2 - max_moment_nsig2
        # Simple quadratic formula:

        a = Minv_xx
        b = TwoMinv_xy__y_My
        c = Minv_yy__y_My__y_My - MAX_MOMENT_NSIG2
        d = b * b - 4 * a * c
        sqrtd = sqrt(d)
        inv2a = Inv2Minv_xx
        x1 = inv2a * (-b - sqrtd) + Mx
        x2 = inv2a * (-b + sqrtd) + Mx

        # stay within image bounds
        ix1 = max(int(ceil(x1)), xmin)
        ix2 = min(int(floor(x2)), xmax)
        # in the following two cases, ask if we somehow wanted to find
        # pixels outside the image
        if (ix1 > xmax) * (ix2 == xmax):
            continue
        elif (ix1 == xmin) * (ix2 < xmin):
            continue
        elif ix1 > ix2:
            # print('ix1 > ix2', y, x1, xmin, x2, xmax, ix1, ix2)
            # usually what happens is you want to take only one pixel and you
            # end up due to the ceil and floor funcs with e.g. 15, 14 instead
            # of 14, 15
            # ix1, ix2 = ix2, ix1
            # ix1 = max(ix1, xmin)
            # ix2 = min(ix2, xmax)
            continue

        for x in xrange(ix1, ix2):

            x_Mx = float(x) - Mx

            # Compute displacement from weight centroid, then get elliptical
            # radius and weight.
            Minv_xx__x_Mx__x_Mx = Minv_xx * x_Mx ** 2
            rho2 = Minv_yy__y_My__y_My + \
                TwoMinv_xy__y_My * x_Mx + \
                Minv_xx__x_Mx__x_Mx

            # this shouldn't happen by construction
            if (rho2 > MAX_MOMENT_NSIG2 + 1e8):
                print('rho2 > max_moment_nsig2 !')
                continue

            intensity = exp(-0.5 * rho2) * data[y, x]  # y,x order!

            A += intensity

            if (p == 0) + (q == 0):
                if p == 0:
                    mu_pq += intensity * y_My_q
                else:
                    mu_pq += intensity * x_Mx ** p
            else:
                mu_pq += intensity * x_Mx ** p * y_My_q

    return mu_pq / A


