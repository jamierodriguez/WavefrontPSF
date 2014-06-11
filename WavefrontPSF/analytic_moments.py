#!/usr/bin/env python
"""
File: analytic_moments.py
Author: Chris Davis
Description: Class for creating wavefronts analytically from a given zernike.
"""

from __future__ import print_function, division
import numpy as np
import sympy as sp

###############################################################################
# create the analytic arrays
###############################################################################

w, R = sp.symbols('w, R')
ws = sp.conjugate(w)

# put in the gradient by hand
# TODO: double check 0th components
# TODO: also double check effects of conjugation
#       this extends to the 2 and 3 components
grad = [0,
        0,
        0,
        0,
        (4 * np.sqrt(3) * w),
        (2 * np.sqrt(6) * sp.I * ws),
        (2 * np.sqrt(6) * ws),
        (-sp.I * np.sqrt(8) * (3 * w ** 2 - 6 * w * ws + 2)),
        (np.sqrt(8) * (3 * w ** 2 + 6 * w * ws - 2)),
        (sp.I * 3 * np.sqrt(8) * ws ** 2),
        (3 * np.sqrt(8) * ws ** 2),
        24 * np.sqrt(5) * w * (w * ws - 1/2)]

# jarvis, schechter, and jain grad w
grad_jsj = [0,
            0,
            0,
            0,
            2 * w,
            2 * sp.I * ws,
            2 * ws,
            -sp.I * (w ** 2 - 2 * w * ws + 1),
            w ** 2 + 2 * w * ws - 1,
            3 * sp.I * ws ** 2,
            3 * ws ** 2,
            4 * w ** 2 * ws]

estimate = {}
z_list = sp.symbols('z_0:12', real = True)
expression = {}

# look at e0 term
real = []
imaginary = []
expr_re = 0
expr_im = 0
for i in range(1, 12):
    for j in range(1, 12):
        expr = grad[i] * sp.conjugate(grad[j])
        expr = sp.expand(expr)
        expr = expr.subs(w * sp.conjugate(w), R)

        # now replace any other w or sp.conjugate(w) with 0
        expr = expr.subs(w, 0)

        # now replace R^4 with 1/5, etc
        for rpow in range(8, 0, -1):
            expr = expr.subs(R ** rpow, 1 / (rpow + 1))
        if sp.re(expr) != 0:
            real.append([float(sp.re(expr)), i, j])
            expr_re = expr_re + z_list[i] * z_list[j] * sp.re(expr)
        if sp.im(expr) != 0:
            imaginary.append([float(sp.im(expr)), i, j])
            expr_im = expr_im + z_list[i] * z_list[j] * sp.im(expr)
real = np.array(real)
imaginary = np.array(imaginary)
estimate.update(dict(e0 = real))
expression.update(dict(e0 = expr_re))


# look at e term
real = []
imaginary = []
expr_re = 0
expr_im = 0
for i in range(1, 12):
    for j in range(1, 12):
        expr = grad[i] * grad[j]
        expr = sp.expand(expr)
        expr = expr.subs(w * sp.conjugate(w), R)

        # now replace any other w or sp.conjugate(w) with 0
        expr = expr.subs(w, 0)

        # now replace R^4 with 1/5, etc
        for rpow in range(8, 0, -1):
            expr = expr.subs(R ** rpow, 1 / (rpow + 1))
        if sp.re(expr) != 0:
            real.append([float(sp.re(expr)), i, j])
            expr_re = expr_re + z_list[i] * z_list[j] * sp.re(expr)
        if sp.im(expr) != 0:
            imaginary.append([float(sp.im(expr)), i, j])
            expr_im = expr_im + z_list[i] * z_list[j] * sp.im(expr)
real = np.array(real)
imaginary = np.array(imaginary)
estimate.update(dict(e1 = real, e2 = imaginary))
expression.update(dict(e1 = expr_re, e2 = expr_im))

# look at delta term
real = []
imaginary = []
expr_re = 0
expr_im = 0
for i in range(1, 12):
    for j in range(1, 12):
        for k in range(1, 12):
            expr = grad[i] * grad[j] * grad[k]
            expr = sp.expand(expr)
            expr = expr.subs(w * sp.conjugate(w), R)

            # now replace any other w or sp.conjugate(w) with 0
            expr = expr.subs(w, 0)

            # now replace R^4 with 1/5, etc
            for rpow in range(8, 0, -1):
                expr = expr.subs(R ** rpow, 1 / (rpow + 1))
            if sp.re(expr) != 0:
                real.append([float(sp.re(expr)), i, j, k])
                expr_re = expr_re + z_list[i] * z_list[j] * z_list[k] * \
                        sp.re(expr)

            if sp.im(expr) != 0:
                imaginary.append([float(sp.im(expr)), i, j, k])
                expr_im = expr_im + z_list[i] * z_list[j] * z_list[k] * \
                        sp.im(expr)
real = np.array(real)
imaginary = np.array(imaginary)
estimate.update(dict(delta1 = real, delta2 = imaginary))
expression.update(dict(delta1 = expr_re, delta2 = expr_im))

# look at zeta term
real = []
imaginary = []
expr_re = 0
expr_im = 0
for i in range(1, 12):
    for j in range(1, 12):
        for k in range(1, 12):
            expr = grad[i] * grad[j] * sp.conjugate(grad[k])
            expr = sp.expand(expr)
            expr = expr.subs(w * sp.conjugate(w), R)

            # now replace any other w or sp.conjugate(w) with 0
            expr = expr.subs(w, 0)

            # now replace R^4 with 1/5, etc
            for rpow in range(8, 0, -1):
                expr = expr.subs(R ** rpow, 1 / (rpow + 1))
            if sp.re(expr) != 0:
                real.append([float(sp.re(expr)), i, j, k])
                expr_re = expr_re + z_list[i] * z_list[j] * z_list[k] * \
                        sp.re(expr)
            if sp.im(expr) != 0:
                imaginary.append([float(sp.im(expr)), i, j, k])
                expr_im = expr_im + z_list[i] * z_list[j] * z_list[k] * \
                        sp.im(expr)
real = np.array(real)
imaginary = np.array(imaginary)
estimate.update(dict(zeta1 = real, zeta2 = imaginary))
expression.update(dict(zeta1 = expr_re, zeta2 = expr_im))

# construct derivative
derivative = {}
for key in estimate.keys():
    derivative_key = []
    for zi in range(12):
        derivative_i = []
        # choose only the ones with zi in it
        chosen = np.any(estimate[key][:, 1:] == zi, axis=1)
        # now create an entry for each zi
        temp = estimate[key][chosen].tolist()
        # go through each entry in temp
        for temp_i in temp:
            for ij in xrange(1, len(temp_i)):
                temp_ij = temp_i[ij]
                if temp_ij == zi:
                    # append the temp list sans this item
                    derivative_i.append(temp_i[:ij] + temp_i[ij + 1:])
        derivative_key.append(np.array(derivative_i))

    derivative.update({key: derivative_key})


# now we also need to include the scalings with rzero
# [oneover?, slope, intercept]
rzero_scaling = {'e0': {'slope': [1.0601e-04, 3.2515e-03, -4.6234e-06],
                        'intercept': [9.3828e-04, 6.5692e-03, 4.2024e-03],},
                 'e1': {'slope': [8.7110e-05, 2.7806e-03, -3.5211e-06],
                        'intercept': [1.6411e-05, 3.8464e-04, 6.5707e-07],},
                 'e2': {'slope': [1.0975e-04, 2.7569e-03, -3.3518e-06],
                        'intercept': [3.6041e-05, -3.7639e-04, -6.8314e-07],},
                 'delta1': {'slope': [-7.7465e-06, -3.8174e-05, 7.8885e-08],
                        'intercept': [-3.1853e-07, 7.0394e-07, 7.4074e-08],},
                 'delta2': {'slope': [-1.0437e-05, -3.8929e-05, 1.7061e-07],
                        'intercept': [-5.5116e-07, 5.7043e-06, 8.2532e-08],},
                 'zeta1': {'slope': [-1.6626e-06, -3.4517e-05, 2.4426e-08],
                        'intercept': [-3.3082e-07, -2.8350e-05, 1.2584e-07],},
                 'zeta2': {'slope': [-1.3047e-06, -3.7087e-05, 3.8996e-09],
                        'intercept': [7.5179e-07, -2.7431e-05, 1.6107e-08],},
                 }

def analytic_data_DEPRECATED(zernikes, rzero, coords=[]):
    return_dict = {}

    for key in estimate.keys():
        # TODO: should be possible to do this numpy-like?
        # Do estimate
        val = 0
        for term in estimate[key]:
            val_i = term[0]
            for term_i in xrange(1, len(term)):
                if term[term_i] <= zernikes.shape[1]:
                    val_i *= zernikes[:, int(term[term_i]) - 1]
                else:
                    val_i *= 0
            val += val_i

        # incorporate rzero slope and intercept
        rzero_scaling_items = rzero_scaling[key]['slope']

        slope = rzero_scaling_items[0] / rzero + \
                rzero_scaling_items[1] + \
                rzero_scaling_items[2] / rzero ** 2


        rzero_scaling_items = rzero_scaling[key]['intercept']

        intercept = rzero_scaling_items[0] / rzero + \
                    rzero_scaling_items[1] + \
                    rzero_scaling_items[2] / rzero ** 2



        val = val * slope + intercept

        return_dict.update({key: val})

    if len(coords) > 0:
        return_dict.update({'x': coords[:,0],
                            'y': coords[:,1],
                            'chip': coords[:,2]})

    return return_dict

def analytic_data(zernikes, rzero, coords=[]):

    return_dict = {}

    if type(rzero) == int:
        rzeros = [rzero] * len(zernikes)
    else:
        rzeros = rzero

    if len(coords) == 0:
        coords = np.array([[0, 0, 0]] * len(zernikes))
    return_dict.update({'x': coords[:,0],
                        'y': coords[:,1],
                        'chip': coords[:,2]})


    for key in estimate.keys():
        deg = fit_dict['deg']
        c = fit_dict[key]
        # TODO: should be possible to do this numpy-like?
        # Do estimate
        val = 0
        for term in estimate[key]:
            val_i = term[0]
            for term_i in xrange(1, len(term)):
                if term[term_i] <= zernikes.shape[1]:
                    val_i *= zernikes[:, int(term[term_i]) - 1]
                else:
                    val_i *= 0
            val += val_i

        val_fin = interpolate_4d(val, 1. / rzeros, coords[:,0], coords[:,1],
                                 deg, c)

        return_dict.update({key: val_fin})

    return return_dict


def analytic_data_no_rzero(zernikes, coords=[]):
    return_dict = {}

    for key in estimate.keys():
        # TODO: should be possible to do this numpy-like?
        # Do estimate
        val = 0
        for term in estimate[key]:
            val_i = term[0]
            for term_i in xrange(1, len(term)):
                if term[term_i] <= zernikes.shape[1]:
                    val_i *= zernikes[:, int(term[term_i]) - 1]
                else:
                    val_i *= 0
            val += val_i

        return_dict.update({key: val})

    if len(coords) > 0:
        return_dict.update({'x': coords[:,0],
                            'y': coords[:,1],
                            'chip': coords[:,2]})

    return return_dict

def analytic_derivative_data_DEPRECATED(zernikes, rzero, coords=[]):
    return_dict = {}
    for key in derivative.keys():
        return_dict.update({key: {}})
        for zi in range(12):
            val = 0
            for term in derivative[key][zi]:
                if len(term) > 0:
                    val_i = term[0]
                    for term_i in xrange(1, len(term)):
                        if term[term_i] <= zernikes.shape[1]:
                            val_i *= zernikes[:, int(term[term_i]) - 1]
                        else:
                            val_i *= 0
                    val += val_i

            # incorporate rzero slope. No intercept for derivative
            rzero_scaling_items = rzero_scaling[key]['slope']

            slope = rzero_scaling_items[0] / rzero + \
                    rzero_scaling_items[1] + \
                    rzero_scaling_items[2] / rzero ** 2

            val = val * slope

            return_dict[key].update({'z{0:02d}'.format(zi): val})

        # throw in rzero derivative
        val = 0
        for term in estimate[key]:
            val_i = term[0]
            for term_i in xrange(1, len(term)):
                if term[term_i] <= zernikes.shape[1]:
                    val_i *= zernikes[:, int(term[term_i]) - 1]
                else:
                    val_i *= 0
            val += val_i

        rzero_scaling_items = rzero_scaling[key]['slope']

        slope = -rzero_scaling_items[0] / rzero ** 2 + \
                -2 * rzero_scaling_items[2] / rzero ** 3


        rzero_scaling_items = rzero_scaling[key]['intercept']

        intercept = - rzero_scaling_items[0] / rzero ** 2 + \
                    -2 * rzero_scaling_items[2] / rzero ** 3

        val = val * slope + intercept

        return_dict[key].update({'rzero': val})


    if len(coords) > 0:
        return_dict.update({'x': coords[:,0],
                            'y': coords[:,1],
                            'chip': coords[:,2]})

    return return_dict

def get_derivative(key, zi):
    return get_expression(key).diff(z_list[zi])

def get_expression(key):
    # give sympy expression

    estimate_array = estimate[key]
    expr = 0
    for term in estimate_array:
        val_i = term[0]
        for term_i in xrange(1, len(term)):
            val_i = val_i * z_list[np.int(term[term_i])]
        expr = expr + val_i
    return expr

def print_expression(key):
    # same idea but list of terms

    estimate_array = estimate[key]
    vals = []
    zs = []
    for term in estimate_array:
        val = term[0]
        zs_i = sorted(list(term[1:]))

        # find if permutation of zs is already in expr
        same = False
        same_i = 0
        while (not same) * (same_i < len(zs)):
            zs_same = sorted(zs[same_i])
            same = (zs_same == zs_i)
            same_i += 1
        # if so, add val to it!
        if same:
            vals[same_i - 1] += val
        else:
            # if not create a new one!
            zs.append(zs_i)
            vals.append(val)

    # sort them; this is pretty fancy (thanks stackoverflow!)
    # http://stackoverflow.com/questions/9543211/sorting-a-list-in-python-using-the-result-from-sorting-another-list
    zs, vals = zip(*sorted(zip(zs, vals), key=lambda x: sorted(x[0])))

    # now construct strings for printing this
    string_expression = []
    total_expression = ''
    for i in xrange(len(vals)):
        value = '{0:.2f} '.format(vals[i])
        for zi in zs[i]:
            value += 'z_{' + '{0}'.format(int(zi)) + '} '
        string_expression.append(value)
        total_expression += value + '+ '
    string_expression.append(total_expression[:-3])
    return string_expression


###############################################################################
# poly interp helper functions
###############################################################################

def polyvander4d(a, r, x, y, deg):
    a, r, x, y = np.asarray((a, r, x, y)) + 0.0
    dega, degr, degx, degy = deg
    va = np.polynomial.polynomial.polyvander(a, dega)
    vr = np.polynomial.polynomial.polyvander(r, degr)
    vx = np.polynomial.polynomial.polyvander(x, degx)
    vy = np.polynomial.polynomial.polyvander(y, degy)
    v = va[..., None, None, None] * \
        vr[..., None, :, None, None] * \
        vx[..., None, None, :, None] * \
        vy[..., None, None, None, :]
    v = v.reshape(v.shape[:-4] + (-1,))
    return v

def interpolate_4d(a, r, x, y, deg, c):
    v = polyvander4d(a, r, x, y, deg)
    z = np.dot(v, c)

    return z

def fit_analytic_rzero_coords(a, r, x, y, z, deg):
    z = np.asarray(z) + 0.0
    v = polyvander4d(a, r, x, y, deg)
    lhs = v.T
    rhs = z.T

    scl = np.sqrt(np.square(lhs).sum(1))
    scl[scl == 0] = 1

    # Solve the least squares problem.
    c, resids, rank, s = np.linalg.lstsq(lhs.T/scl, rhs.T)
    c = (c.T/scl).T

    # c are the answers
    # dot(v, c) is the interpolated fit
    return c, np.dot(v, c), resids, rank, s

###############################################################################
# fit_dict for polyinterp
###############################################################################


fit_dict = {'deg': [1, 2, 3, 3],
 'delta1': np.array([ -5.28356246e-05,  -1.21048533e-07,   3.18701701e-10,
          7.52591866e-12,  -3.33603018e-07,  -1.07522186e-08,
          1.49043751e-11,   5.41971638e-13,   8.51965726e-10,
          7.70048653e-13,   1.02686643e-13,   4.90229832e-16,
         -1.09817461e-11,   5.24940183e-13,   1.22602269e-15,
         -2.86233265e-17,   1.31952674e-05,   1.10995768e-07,
          1.62008952e-10,  -2.98168039e-12,   1.18410768e-07,
          3.97225021e-09,   3.54243690e-11,  -1.74665609e-13,
          2.12401542e-11,  -4.98667964e-13,  -6.09041999e-14,
         -2.78786406e-16,  -9.80291541e-12,  -1.60791775e-13,
         -6.76890849e-16,   6.59475165e-18,  -8.86361132e-07,
         -6.31490554e-09,  -1.25592156e-11,   2.71036192e-13,
         -8.25316032e-09,  -2.84976879e-10,  -6.47993708e-13,
          1.27349027e-14,   1.03090093e-11,  -1.79170114e-14,
          3.13424599e-15,   2.97752275e-17,   1.07159582e-13,
          1.19122427e-14,   3.75889723e-17,  -5.14421007e-19,
         -1.91601024e-05,  -3.33136902e-08,  -5.91962287e-10,
          1.76072314e-12,   3.18327289e-08,   4.03619000e-10,
         -9.67580063e-13,  -1.72291274e-14,  -1.97825465e-10,
         -1.72667561e-12,  -6.94920404e-16,  -6.24607700e-17,
         -1.05516825e-12,  -4.53456381e-14,   1.75910930e-16,
          1.67404133e-18,  -1.09346090e-05,   2.44284511e-08,
          2.32329630e-10,  -8.91526781e-13,  -5.53781456e-09,
         -1.36007981e-10,   3.16714160e-13,   4.26034679e-15,
          1.47066828e-10,  -3.50949684e-13,   2.03777835e-15,
          3.01066011e-17,   1.05457355e-13,   1.40504106e-14,
         -4.59084267e-17,  -5.91316711e-19,   3.72966798e-07,
         -1.57910604e-09,  -1.09053140e-11,   5.83976215e-14,
         -2.72933651e-10,   1.33645440e-11,  -5.70997682e-15,
         -3.57814198e-16,  -4.60633360e-12,   2.01664126e-14,
         -8.04204532e-17,  -2.61607188e-18,   2.94101943e-15,
         -1.06133255e-15,   3.24068684e-18,   3.80397348e-20]),
 'delta2': np.array([  4.35338345e-05,  -1.47051705e-07,  -7.12779020e-10,
          2.54269686e-11,   2.02305572e-07,  -3.95167229e-09,
          7.43747401e-12,   2.97345973e-13,  -5.68176841e-10,
         -8.29148189e-13,   1.56554456e-13,  -2.87244560e-15,
          5.69425325e-12,   4.30222308e-13,  -1.72048065e-15,
         -3.28625876e-17,  -1.10612801e-05,   1.04697765e-07,
          3.08271739e-10,   3.96367977e-12,  -5.08014311e-08,
          1.45501071e-09,   3.34970623e-13,  -1.05987664e-13,
          9.12113917e-11,  -3.86412642e-11,  -6.40546752e-14,
          7.82856428e-16,  -9.83171096e-13,  -9.52835651e-14,
          4.26451941e-16,   9.34718054e-18,   8.90378192e-07,
         -4.62968719e-09,  -3.43446584e-11,   2.83799706e-13,
          5.00976766e-09,  -9.96715844e-11,   6.79450899e-14,
          7.69324654e-15,  -2.06691463e-11,   8.62584799e-13,
          6.54992087e-15,  -5.18681781e-17,   6.11259687e-14,
          7.69429838e-15,  -3.55344204e-17,  -6.92797607e-19,
         -2.75651583e-05,  -6.21670076e-08,   1.91315190e-11,
          1.63536657e-12,  -3.97794570e-08,  -9.46860799e-10,
          5.26371420e-12,   2.20946117e-14,  -1.36580327e-10,
         -3.04750645e-12,   2.79833267e-14,   2.96939996e-16,
          7.99302116e-13,   6.31481647e-15,  -1.04468170e-16,
          2.98350937e-18,  -8.42704482e-06,   2.04152007e-08,
          6.38755059e-11,  -5.23408972e-13,   1.53849701e-08,
          2.98720222e-10,  -1.68103562e-12,  -9.12834849e-15,
          1.29226452e-10,   5.79499814e-13,  -7.73997916e-15,
         -7.35843733e-17,  -4.08896311e-13,  -1.62349885e-15,
          3.41385314e-17,  -9.45161384e-19,   1.89338304e-07,
         -1.14781408e-09,   9.89738240e-13,   2.65874575e-14,
         -1.12717060e-09,  -2.41716345e-11,   1.24962589e-13,
          7.64113583e-16,  -2.84167336e-12,  -6.01616973e-14,
          7.44303897e-16,   5.48350925e-18,   2.39670164e-14,
          1.71486469e-16,  -2.31322684e-18,   6.97831477e-20]),
 'e0': np.array([  6.95430994e-03,  -3.03097348e-06,   9.63116838e-08,
          9.25113870e-11,   8.95491002e-07,  -2.13881621e-08,
         -3.73319912e-10,   4.74623747e-12,   1.04755623e-07,
          3.55964080e-10,   4.10552176e-12,  -9.87178885e-15,
         -1.97928199e-10,  -4.14660906e-12,   2.61298012e-14,
         -7.49518442e-18,   9.37615622e-04,   7.24830535e-07,
         -4.53477438e-09,  -2.78487896e-11,   1.43544142e-07,
          3.65361726e-09,   8.56021282e-11,  -8.81455872e-13,
         -6.81670409e-09,  -8.51714858e-11,  -6.49984406e-13,
          4.71337795e-15,   3.81238527e-11,   7.40627642e-13,
         -9.05794426e-15,   3.25742831e-17,   4.20084319e-03,
         -4.44261179e-08,   3.85965023e-11,   1.82476356e-12,
         -1.94557501e-08,  -6.13956237e-10,  -5.16160849e-12,
          7.57732668e-14,   2.25739179e-10,   4.56909885e-12,
          2.26180347e-14,  -3.43007730e-16,  -2.37471519e-12,
         -2.17135268e-14,   6.30115486e-16,  -4.58034324e-18,
          1.74221183e-03,   1.43187104e-06,   2.22716144e-08,
         -2.54771442e-11,   8.04438936e-07,   1.10926690e-08,
          1.64633297e-10,  -3.41415039e-13,   1.30984101e-08,
         -7.02169456e-11,  -1.10532966e-12,   5.17359513e-16,
         -4.00893177e-11,   5.06876858e-13,  -1.07004992e-14,
         -4.16076961e-17,   2.65349399e-04,  -3.48404152e-07,
         -2.88264761e-09,   8.75753697e-12,  -4.14716495e-07,
         -2.41255840e-09,  -2.71577410e-11,   1.08298879e-13,
         -6.55044947e-10,   1.31500935e-11,   1.53687140e-13,
         -6.88806661e-16,   1.18240504e-11,  -9.00451709e-14,
          2.54506909e-15,   2.14916254e-19,  -1.27521177e-05,
          1.92605243e-08,   1.82788630e-10,  -5.54851325e-13,
          2.65359202e-08,   2.29833068e-10,   1.50590356e-12,
         -1.17505481e-14,   4.03163665e-11,  -4.44077504e-13,
         -6.60308211e-15,   5.49114046e-17,  -5.62372087e-13,
         -3.24549845e-16,  -1.60826343e-16,   7.09556599e-19]),
 'e1': np.array([  8.69427438e-05,  -1.35386101e-06,   2.83641187e-08,
          7.08640412e-11,  -6.80877860e-07,  -1.31523360e-09,
          1.25143586e-10,   9.73462025e-13,  -2.04206431e-08,
         -1.28702966e-10,  -2.16175257e-13,   5.28153630e-15,
          5.20950902e-11,  -1.22232127e-13,   2.20474268e-15,
         -1.09975416e-16,   6.50418929e-05,   4.00065432e-07,
         -7.78391873e-09,  -1.51441716e-11,   3.08064725e-07,
         -5.33522866e-09,  -5.33039324e-11,   3.70833434e-14,
          1.46190410e-09,   9.96196834e-12,   2.75770099e-13,
         -6.56539891e-16,  -2.03596687e-11,   3.38773209e-13,
          1.04282942e-15,   1.29916375e-17,  -1.58015295e-06,
         -3.67619630e-08,   4.29412652e-10,   1.28146428e-12,
         -2.95936508e-08,   4.38866470e-10,   4.11636307e-12,
         -9.64038193e-15,  -1.31420283e-10,   2.81894892e-13,
         -1.41787494e-14,  -1.64299466e-17,   1.52629103e-12,
         -2.21660783e-14,  -8.76154706e-17,  -7.60999886e-19,
          2.12260325e-03,  -2.18483516e-06,   2.17208999e-08,
          7.00037486e-11,   1.55424061e-06,  -1.30512116e-08,
          1.85098349e-10,   1.70600092e-12,   2.11997570e-08,
          1.25071048e-11,  -2.32784479e-13,   7.41584992e-15,
         -4.70592495e-11,   2.02515936e-12,  -2.16117508e-14,
         -2.30177182e-16,   1.65806086e-04,   5.71791974e-07,
         -3.03023408e-09,  -1.68871073e-11,  -8.75619165e-07,
          1.00302910e-08,  -4.58173285e-11,  -5.11729226e-13,
         -2.06758733e-09,  -7.86124399e-12,   1.03961390e-13,
         -1.10400519e-15,   2.43169557e-11,  -7.71499845e-13,
          5.98657677e-15,   6.64444264e-17,  -9.32902974e-06,
         -5.25828299e-08,   2.47431128e-10,   1.58802189e-12,
          6.31623837e-08,  -1.00108350e-09,   3.10859568e-12,
          4.24542182e-14,   1.90910942e-10,   1.09120763e-12,
         -9.88586983e-15,   5.57296943e-17,  -1.45209820e-12,
          6.13944783e-14,  -4.18793496e-16,  -4.74388327e-18]),
 'e2': np.array([ -8.28993146e-05,   1.05598234e-06,   2.84495561e-09,
         -3.61199856e-11,   4.48637966e-07,   4.54935375e-08,
         -1.51320423e-10,  -4.19990916e-12,  -7.29478540e-09,
         -1.02469585e-10,  -9.82948642e-13,   1.44820671e-14,
          1.11813213e-10,  -3.77121943e-12,  -2.42418073e-14,
          3.00185535e-16,   8.65098770e-06,  -7.34794309e-07,
         -9.96255635e-10,   2.40295190e-11,  -3.39092959e-07,
         -5.02926748e-09,   4.64509948e-11,   7.37577021e-13,
          1.40266207e-09,   7.01555777e-11,   1.80811164e-13,
         -6.18997694e-15,  -9.00363005e-12,   3.56650549e-13,
          4.46203090e-15,  -4.39049732e-17,  -9.03041493e-07,
          5.13676779e-08,   6.06082925e-11,  -1.78619799e-12,
          2.11922294e-08,   7.55364721e-11,  -2.89696709e-12,
         -3.25555404e-14,  -7.47539420e-11,  -5.65165960e-12,
         -6.73972538e-15,   5.17360382e-16,   6.21176055e-13,
         -1.64799816e-15,  -2.90220174e-16,   2.01553510e-18,
          1.73072663e-03,  -5.34169638e-07,   5.46705137e-08,
          6.01907841e-12,   3.36013747e-06,   4.43088774e-09,
         -2.89160348e-10,  -8.57967967e-13,   3.22744775e-08,
          1.13816866e-10,  -3.47621366e-12,   3.88508694e-15,
         -1.93090370e-10,  -2.38666958e-14,   1.45215602e-14,
         -6.69527727e-18,   1.81331553e-04,  -2.32501230e-07,
         -4.08332965e-09,   9.08960844e-12,  -8.56802348e-07,
         -2.78086670e-09,   8.05292191e-11,   2.35662663e-13,
          1.55678134e-09,  -1.31188011e-11,   2.93834970e-13,
         -1.35647946e-15,   3.94814057e-11,   1.63636169e-14,
         -3.29870866e-15,   1.00937940e-17,  -8.44384032e-06,
          3.94497168e-09,   2.23856653e-10,  -3.87468720e-13,
          4.39972885e-08,   7.41293034e-11,  -5.65979006e-12,
         -1.28102574e-14,  -2.10827486e-10,   1.51911331e-12,
         -1.17542473e-14,   9.33166691e-17,  -2.11036591e-12,
         -1.01348319e-15,   2.40594478e-16,  -4.68707851e-19]),
 'zeta1': np.array([ -6.21526013e-05,  -2.93044559e-07,   1.05555361e-09,
          9.95410910e-12,  -1.19044361e-07,  -1.90318099e-09,
         -2.08455445e-11,   3.65992635e-14,   3.24089650e-10,
          1.39929912e-11,  -2.50514869e-14,  -4.35699343e-16,
          1.53831484e-12,   1.08848578e-13,  -1.10197231e-16,
         -1.56179846e-18,   1.79527863e-05,   7.40778559e-08,
         -3.08215340e-10,  -2.51289380e-12,   5.97301572e-08,
          6.34707607e-10,   1.07755012e-12,  -2.04777569e-14,
         -2.18119637e-10,  -4.52348000e-12,   9.83766004e-15,
          6.93171013e-17,  -2.70440977e-12,  -4.63905559e-14,
          2.57346359e-17,   1.43638005e-18,  -1.25240419e-06,
         -6.35870532e-09,   2.15139146e-11,   1.85943364e-13,
         -5.85369714e-09,  -6.56719493e-11,  -5.06436388e-14,
          1.57565969e-15,   1.67114546e-11,   3.93202800e-13,
         -6.46779878e-16,  -4.33444870e-18,   2.39840159e-13,
          3.60713223e-15,  -3.61681965e-18,  -5.38778495e-20,
         -1.52886635e-05,  -1.68826231e-09,  -2.06406678e-10,
          2.83335624e-13,   5.01285093e-08,   3.41164366e-10,
          3.30169202e-14,  -7.28750969e-15,  -2.78730991e-10,
         -1.85291231e-12,   1.90280891e-14,   1.02020857e-16,
         -7.66118994e-13,  -2.28633908e-14,  -2.26521051e-17,
          1.03458230e-18,  -3.24209099e-06,   5.59425599e-09,
          2.97244254e-11,  -2.67952066e-13,  -6.91369951e-09,
         -9.77781353e-11,  -1.72165485e-13,   3.38242197e-15,
          3.63621032e-11,   3.43156496e-13,  -6.35438369e-15,
         -1.43658550e-17,  -3.06795327e-14,   7.05724744e-15,
          9.56759631e-18,  -3.99738380e-19,  -6.68784669e-09,
         -2.86726329e-10,   3.81146791e-13,   1.92603331e-14,
          3.07792825e-10,   6.75288367e-12,   2.17126081e-14,
         -2.67878289e-16,   4.45861940e-13,  -3.39252623e-14,
          4.52013518e-16,   6.70843250e-19,   1.15905048e-15,
         -5.38865183e-16,  -6.41885191e-19,   3.04722088e-20]),
 'zeta2': np.array([ -5.86050879e-05,  -2.46153510e-08,   8.51958685e-10,
         -1.08474826e-12,  -2.77208742e-07,  -2.28609058e-09,
          1.98234810e-11,   1.09323433e-13,   1.00155610e-09,
         -1.68188359e-11,   5.98901023e-14,  -3.20374413e-16,
          8.59389118e-12,   5.63875998e-14,  -8.91481459e-16,
         -3.00320838e-18,   1.56224742e-05,   2.11322583e-08,
         -2.72717326e-10,  -2.49116245e-12,   8.15582979e-08,
          8.29985671e-10,  -8.00830653e-12,  -4.66567778e-14,
         -2.69001311e-10,   4.10790020e-12,  -2.09069228e-14,
          3.59183945e-17,  -2.90186581e-12,  -3.16001903e-14,
          3.57355230e-16,   2.00780403e-18,  -1.09183859e-06,
         -2.78187157e-09,   2.04847759e-11,   1.98033053e-13,
         -7.11027763e-09,  -7.21228347e-11,   6.73106855e-13,
          3.76708671e-15,   1.93565957e-11,  -2.24860110e-13,
          1.57402975e-15,  -7.67511107e-18,   2.36856907e-13,
          2.79179062e-15,  -2.76286695e-17,  -1.63406329e-19,
         -1.05601193e-05,  -5.09500096e-08,  -5.09382592e-10,
          1.33716379e-12,   2.32575476e-08,   1.84797492e-10,
         -3.20809523e-12,  -2.10081591e-14,  -2.43687258e-10,
          6.13273910e-13,   6.26213544e-15,   6.58714176e-18,
         -3.48891415e-13,  -6.39021405e-15,   1.38010085e-16,
          1.16069231e-18,  -5.33043296e-06,   2.17399394e-08,
          9.71736024e-11,  -5.61738904e-13,  -4.61354977e-09,
         -2.42722752e-11,   8.77215455e-13,   4.97228551e-15,
          4.66464413e-11,  -1.64716540e-13,  -1.95122335e-15,
         -1.00979464e-17,   5.56536152e-14,   2.15672652e-15,
         -3.73620581e-17,  -4.28707429e-19,   1.33309069e-07,
         -1.50332514e-09,  -4.14413611e-12,   3.68935947e-14,
          3.73204633e-10,   3.61336316e-12,  -5.98513419e-14,
         -4.40218375e-16,  -1.47471870e-12,   5.79172246e-15,
          2.25153688e-16,   6.74875209e-19,  -7.48850078e-15,
         -2.10580592e-16,   2.40894237e-18,   3.01177147e-20])}
