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

def analytic_data(zernikes, rzero, coords=[]):
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

def analytic_derivative_data(zernikes, rzero, coords=[]):
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
