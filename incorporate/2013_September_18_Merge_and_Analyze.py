# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# take those images I generated earlier and analyze them!

# <codecell>

%pylab inline
%load_ext autoreload
%autoreload 2
#%install_ext https://raw.github.com/lyonsquark/ipython_ext/master/ipythonRoot.py
#%load_ext ipythonRoot
%connect_info  # display the connection info
%config InlineBackend.figure_format='retina'
#plt.xkcd()
from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
from pylab import *
from minuit_fit import Minuit_Fit
from chi_class_plot import FocalPlane_Plotter
from chi_class import create_minuit_dictionary
from iminuit import Minuit, describe, Struct
from os import path, makedirs

# <codecell>

#%qtconsole # open up qtconsole to this kernel

# <markdowncell>

# ### first merge all the outputs and collect the results

# <codecell>

#our tool
FP_dict = dict(
    verbosity=1,
    subav=False,
    calculate_comparison=False,
    randomFlag=False,
    apply_image_correction=True,
    path_enter='merged_158980s',
    path_base='/nfs/slac/g/ki/ki18/cpd/focus/september_25_earlier/')

FP = FocalPlane_Plotter(**FP_dict)
#image_number_range = range(190699, 190918)
image_number_range = range(158981, 159000, 2)
boxdiv = 1
#image_number_range = range(180700, 180721)

# <markdowncell>

# # don't do this one!

# <codecell>

folders = ['fit', '3sigfit', '3sigfit_vs_fit', 'initial']
graphs_list = ['whisker', 'E1E2', 'whisker_rotated', 'fwhm', 'e0', 'chi2s']

for folder in folders:
    output_directory = FP.path_base + FP.path_enter + '/{0}/'.format(folder)
    history = {'output_directory': []}
    for image_number in image_number_range:
        output_directory_i = FP.path_base + '{0:08d}/{1}/'.format(image_number, folder)
        if path.exists(output_directory_i + 'whisker.pdf'):
            history['output_directory'].append(output_directory_i)
    FP.merge_history_images(history=history, graphs_list=graphs_list, output_directory=output_directory,
                            make_movie=False, make_graphs=False)

# <markdowncell>

# #### now collect the results
# 
# ## potentially exclude high chi2 values as being the 'failed fits' ?
# 
# *NOTE*: the chi2s that are unnormalized are (61 - 22) * 3 x larger than the ones given by amin!

# <codecell>

#FP_keys = ['rzero', 'e1', 'e2', 'dz', 'dx', 'dy', 'xt', 'yt', 'z04x', 'z04y',
#           'z05d', 'z06d', 'z07x', 'z07y', 'z08x', 'z08y',
#           'z09d', 'z09x', 'z09y', 'z10d', 'z10x', 'z10y']
FP_keys = ['rzero', 'e1', 'e2', 'dz', 'dx', 'dy', 'xt', 'yt', 'z05d', 'z06d', 'z07x', 'z07y', 'z08x', 'z08y']
mnstat_keys = ['amin', 'edm', 'nCalls', 'nCallsDerivative']
args = []
errors = []
image_numbers = []
zc_image = []
zc_fit = []
mnstat = []
fwhm = []
r50 = []
fwhm_numbers = []
qrfwhm = []
qrfwhm_numbers = []

for image_number in image_number_range:
    # recreate the FP
    if path.exists(FP.path_base + '{0:08d}'.format(image_number)):
        image_numbers.append(image_number)
        item = np.load(FP.path_base + '{0:08d}/minuit_results.npy'.format(image_number)).item()
        args.append([item['args'][key] for key in FP_keys])
        errors.append([item['errors'][key] for key in FP_keys])
        mnstat.append([item['mnstat'][key] for key in mnstat_keys])
        
        image_data = FP.get_image_data(FP.csv, image_number)
        # get the fwhm data
        if not np.ma.getmaskarray(image_data['fwhm']):
            fwhm_numbers.append(image_number)
        fwhm.append(image_data['fwhm'])
        r50.append(image_data['r50'])
        # repeat for qrfwhm
        if not np.ma.getmaskarray(image_data['qrfwhm']):
            qrfwhm_numbers.append(image_number)        
        qrfwhm.append(image_data['qrfwhm'])

        zc_image.append(FP.get_image_correction(image_data)[:FP.length])
       
        zc_fit.append(FP.create_zernike_corrections_from_dictionary(item['args']))
args = np.array(args)
errors = np.array(errors)
zc_image = np.array(zc_image)
zc_fit = np.array(zc_fit)
mnstat = np.array(mnstat)
image_numbers = np.array(image_numbers)
fwhm_numbers = np.ma.array(fwhm_numbers)
fwhm = np.squeeze(np.ma.array(fwhm))
qrfwhm_numbers = np.ma.array(qrfwhm_numbers)
qrfwhm = np.squeeze(np.ma.array(qrfwhm))
r50 = np.squeeze(np.ma.array(r50))

# <markdowncell>

# look at inputs as function of time

# <codecell>

fig_time, axers_time = plt.subplots(figsize=(10, 8 * len(FP_keys)), nrows=len(FP_keys), ncols=1, squeeze=True, sharex=True)
minuit_dict = create_minuit_dictionary(FP_keys)
cutoff_top = 8e4
cutoff_bottom = 3. * (61 * (1 + boxdiv) - len(FP_keys))

zeros = []
for i in range(len(FP_keys)):
    key = FP_keys[i]
    if key == 'rzero':
        zeros.append(FP.fwhm_to_rzero(fwhm))
    else:
        zeros.append(np.zeros(len(FP.fwhm_to_rzero(fwhm))))
zeros = np.array(zeros)
        
for i in range(len(FP_keys)):
    key = FP_keys[i]
    ax = axers_time[i]
    ax.set_xlabel('image_number')
    ax.set_ylabel(key)
    ax.set_ylim(minuit_dict['limit_{0}'.format(key)])
    ax.set_xlim(image_number_range[0], image_number_range[-1])
    filterd_OK = np.where((mnstat[:,0] < cutoff_top / cutoff_bottom))[0]
    ax.errorbar(image_numbers[filterd_OK], args[:,i][filterd_OK], yerr=errors[:,i][filterd_OK], fmt='bo', label=r'OK fit')
    #ax.plot(image_numbers[filterd_OK], np.average(args[:,i][filterd_OK], weights=1./errors[:,i][filterd_OK]) * np.ones(filterd_OK.size), 'b-')
    filterd_fail = np.where((mnstat[:,0] > cutoff_top / cutoff_bottom))[0]
    ax.plot(image_numbers[filterd_fail], args[:,i][filterd_fail], 'ro', label=r'Bad fit')
    if key == 'rzero':
        # plot where IQ gave the rzero...
        ax.plot(image_numbers, FP.fwhm_to_rzero(fwhm), 'm+-', label='from fwhm')
        ax.plot(image_numbers, FP.fwhm_to_rzero(qrfwhm), 'g+-', label='from qrfwhm')
        ax.plot(image_numbers, FP.fwhm_to_rzero(2 * r50), 'k+-', label='from r50')
        #ax.plot(fwhm_numbers, FP.fwhm_to_rzero(fwhm), 'm+-', label='from fwhm')
        #ax.plot(qrfwhm_numbers, FP.fwhm_to_rzero(qrfwhm), 'g+-', label='from qrfwhm')
        #ax.plot(fwhm_numbers, FP.fwhm_to_rzero(2 * r50), 'k+-', label='from r50')
        ax.legend()

fig_time.tight_layout()
if not(path.exists(FP.path_base + FP.path_enter + '/pics/')):
    makedirs(FP.path_base + FP.path_enter + '/pics/')
fig_time.savefig(FP.path_base + FP.path_enter + '/pics/scatter-time.pdf')

# <codecell>

from scipy.optimize import curve_fit
def fitter(rzero, m, b):
    '''
    fwhm = m / rzero + b
    so
    rzero = m / (fwhm - b) + c (extra corrective)
    '''
    #return m / (fwhm - b)
    return m / rzero + b

# correlate rzero with r50 and fwhm
fig = plt.figure(figsize=(10,8))
i = 0
key = FP_keys[i]
ax = fig.add_subplot('111')
ax.set_ylabel('fitted ' + key)
ax.set_xlabel('FWHM')
# for fitting purposes:
fwhm_mask = np.ma.getmaskarray(fwhm)
if np.sum(fwhm_mask) - fwhm.size < 0:
    ax.errorbar(fwhm[filterd_OK], args[:,i][filterd_OK], yerr=errors[:,i][filterd_OK], fmt='ob', label='fwhm')
    #popt, pcov = curve_fit(fitter, fwhm[filterd_OK], args[:,i][filterd_OK], sigma=errors[:,i][filterd_OK])
    #ax.plot(fitter(fwhm[filterd_OK], *popt), fwhm[filterd_OK], 'b-', label='fwhm {0}'.format(popt))
    popt, pcov = curve_fit(fitter, args[:,i][filterd_OK][~fwhm_mask], fwhm[filterd_OK][~fwhm_mask])
    ax.plot(fitter(sort(args[:,i][filterd_OK]), *popt), sort(args[:,i][filterd_OK]), 'b-', label='fwhm {0}'.format(popt))
    
    ax.errorbar(2 * r50[filterd_OK], args[:,i][filterd_OK], yerr=errors[:,i][filterd_OK], fmt='r+', label='2$r_{50}$')
    #popt, pcov = curve_fit(fitter, 2 * r50[filterd_OK], args[:,i][filterd_OK], sigma=errors[:,i][filterd_OK])
    #ax.plot(fitter(2 * r50[filterd_OK], *popt), 2 * r50[filterd_OK], 'r-', label='r50 {0}'.format(popt))
    # for fitting purposes:
    r50_mask = np.ma.getmaskarray(r50)
    popt, pcov = curve_fit(fitter, args[:,i][filterd_OK][~r50_mask], 2 * r50[filterd_OK][~r50_mask])
    ax.plot(fitter(sort(args[:,i][filterd_OK]), *popt), sort(args[:,i][filterd_OK]), 'r:', label='2r50 {0}'.format(popt))

# for fitting purposes:
qrfwhm_mask = np.ma.getmaskarray(qrfwhm)
if np.sum(qrfwhm_mask) - qrfwhm.size < 0:
    ax.errorbar(qrfwhm[filterd_OK], args[:,i][filterd_OK], yerr=errors[:,i][filterd_OK], fmt='mo', label='qrfwhm')
    #popt, pcov = curve_fit(fitter, fwhm[filterd_OK], args[:,i][filterd_OK], sigma=errors[:,i][filterd_OK])
    #ax.plot(fitter(fwhm[filterd_OK], *popt), fwhm[filterd_OK], 'b-', label='fwhm {0}'.format(popt))

    popt, pcov = curve_fit(fitter, args[:,i][filterd_OK][~qrfwhm_mask], qrfwhm[filterd_OK][~qrfwhm_mask])
    ax.plot(fitter(sort(args[:,i][filterd_OK]), *popt), sort(args[:,i][filterd_OK]), 'm-', label='qrfwhm {0}'.format(popt))   
ax.plot(FP.rzero_to_fwhm(sort(args[:,i][filterd_OK])), sort(args[:,i][filterd_OK]), 'k--', label='empirical rzero_to_fwhm')

ax.legend()
fig.savefig(FP.path_base + FP.path_enter + '/pics/fwhm_vs_rzero.pdf')

# correlate rzero with r50 and fwhm
fig = plt.figure(figsize=(10,8))
i = 0
key = FP_keys[i]
ax = fig.add_subplot('111')
ax.set_xlabel('fitted ' + key)
ax.set_ylabel('1/FWHM')
if len(fwhm) > 0:
    ax.plot(args[:,i][filterd_OK], 1. / fwhm[filterd_OK], 'bo', label='fwhm')
    ax.plot(args[:,i][filterd_OK], 1. / (2 * r50[filterd_OK]), 'r+', label='2$r_{50}$')
if len(qrfwhm) > 0:
    ax.plot(args[:,i][filterd_OK], 1. / qrfwhm[filterd_OK], 'mo', label='qrfwhm')
ax.plot(sort(args[:,i][filterd_OK]), 1. / FP.rzero_to_fwhm(sort(args[:,i][filterd_OK])), 'k--', label='empirical rzero_to_fwhm')
ax.legend()
fig.savefig(FP.path_base + FP.path_enter + '/pics/oneoverfwhm.pdf')

# <markdowncell>

# # nor this one!

# <markdowncell>

# #### now look at the zc corrections as well as the ones you were initially given;
# #### also consider the effects of the reference correction?

# <codecell>

import numpy.ma as ma
        
def MAD(a, c=0.6745, axis=None):
    """
    Median Absolute Deviation along given axis of an array:

    median(abs(a - median(a))) / c

    c = 0.6745 is the constant to convert from MAD to std; it is used by
    default

    """

    a = ma.masked_where(a!=a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = ma.median(ma.fabs(a - d) / c)
    else:
        d = ma.median(a, axis=axis)
        # I don't want the array to change so I have to copy it?
        if axis > 0:
            aswp = ma.swapaxes(a,0,axis)
        else:
            aswp = a
        m = ma.median(ma.fabs(aswp - d) / c, axis=0)

    return m

def nanmedian(arr, **kwargs):
    """
    Returns median ignoring NAN
    """
    return ma.median( ma.masked_where(arr!=arr, arr), **kwargs )

# <codecell>

nrows = 7
fig_time, axers_time = plt.subplots(figsize=(7 * 3, 5 * nrows), nrows=nrows, ncols=3, squeeze=False)

zeros = []
for i in range(len(FP_keys)):
    key = FP_keys[i]
    if key == 'rzero':
        zeros.append(FP.fwhm_to_rzero(fwhm))
    else:
        zeros.append(np.zeros(len(FP.fwhm_to_rzero(fwhm))))
zeros = np.array(zeros)

zdict = ['d', 'x', 'y']
for row_i in range(0, nrows):
    for col_i in range(3):
        ax = axers_time[row_i, col_i]
        ax.set_xlabel('image_number')
        ax.set_ylabel('z{0:02d}{1} image'.format(row_i + 4, zdict[col_i]))
        filterd_OK = np.where(mnstat[:,0] < cutoff_top / cutoff_bottom)[0]
        ax.plot(image_numbers[filterd_OK], zc_image[filterd_OK,row_i + 3,col_i], 'bo-')
        # plot mean
        ax.plot(image_numbers[filterd_OK], np.median(zc_image[filterd_OK,row_i + 3,col_i]) * np.ones(filterd_OK.size), 'k--', linewidth=2)
        ax.fill_between(image_numbers[filterd_OK], 
                        (np.median(zc_image[filterd_OK,row_i + 3,col_i]) - MAD(zc_image[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        (np.median(zc_image[filterd_OK,row_i + 3,col_i]) + MAD(zc_image[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        color='blue', alpha=0.3)
fig_time.tight_layout()
fig_time.savefig(FP.path_base + FP.path_enter + '/pics/zgiven_time.pdf')

fig_time, axers_time = plt.subplots(figsize=(7 * 3, 5 * nrows), nrows=nrows, ncols=3, squeeze=False)
for row_i in range(0, nrows):
    for col_i in range(3):
        ax = axers_time[row_i, col_i]
        ax.set_xlabel('image_number')
        ax.set_ylabel('z{0:02d}{1} fit'.format(row_i + 4, zdict[col_i]))
        filterd_OK = np.where(mnstat[:,0] < cutoff_top / cutoff_bottom)[0]
        ax.plot(image_numbers[filterd_OK], zc_fit[filterd_OK,row_i + 3,col_i], 'ro-')
        # plot mean
        ax.plot(image_numbers[filterd_OK], np.median(zc_fit[filterd_OK,row_i + 3,col_i]) * np.ones(filterd_OK.size), 'k--', linewidth=2)
        ax.fill_between(image_numbers[filterd_OK], 
                        (np.median(zc_fit[filterd_OK,row_i + 3,col_i]) - MAD(zc_fit[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        (np.median(zc_fit[filterd_OK,row_i + 3,col_i]) + MAD(zc_fit[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        color='red', alpha=0.3)
fig_time.tight_layout()
fig_time.savefig(FP.path_base + FP.path_enter + '/pics/zfit_time.pdf')

fig_time, axers_time = plt.subplots(figsize=(7 * 3, 5 * nrows), nrows=nrows, ncols=3, squeeze=False)
for row_i in range(0, nrows):
    for col_i in range(3):
        ax = axers_time[row_i, col_i]
        ax.set_xlabel('image_number')
        ax.set_ylabel('z{0:02d}{1} image - fit'.format(row_i + 4, zdict[col_i]))
        filterd_OK = np.where(mnstat[:,0] < cutoff_top / cutoff_bottom)[0]
        ax.plot(image_numbers[filterd_OK], zc_image[filterd_OK,row_i + 3,col_i] - zc_fit[filterd_OK,row_i + 3,col_i], 'go-')
        # plot mean
        ax.plot(image_numbers[filterd_OK], np.median(zc_image[filterd_OK,row_i + 3,col_i] - zc_fit[filterd_OK,row_i + 3,col_i]) * np.ones(filterd_OK.size), 'k--', linewidth=2)
        ax.fill_between(image_numbers[filterd_OK], 
                        (np.median(zc_image[filterd_OK,row_i + 3,col_i] - zc_fit[filterd_OK,row_i + 3,col_i]) - MAD(zc_image[filterd_OK,row_i + 3,col_i] - zc_fit[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        (np.median(zc_image[filterd_OK,row_i + 3,col_i] - zc_fit[filterd_OK,row_i + 3,col_i]) + MAD(zc_image[filterd_OK,row_i + 3,col_i] - zc_fit[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        color='green', alpha=0.3)
fig_time.tight_layout()
fig_time.savefig(FP.path_base + FP.path_enter + '/pics/zgiven-zfit_time.pdf')

fig_time, axers_time = plt.subplots(figsize=(7 * 3, 5 * nrows), nrows=nrows, ncols=3, squeeze=False)
for row_i in range(0, nrows):
    for col_i in range(3):
        ax = axers_time[row_i, col_i]
        ax.set_xlabel('image_number')
        ax.set_ylabel('z{0:02d}{1} image + fit'.format(row_i + 4, zdict[col_i]))
        filterd_OK = np.where(mnstat[:,0] < cutoff_top / cutoff_bottom)[0]
        ax.plot(image_numbers[filterd_OK], zc_image[filterd_OK,row_i + 3,col_i] + zc_fit[filterd_OK,row_i + 3,col_i], 'mo-')
        # plot mean
        ax.plot(image_numbers[filterd_OK], np.median(zc_image[filterd_OK,row_i + 3,col_i] + zc_fit[filterd_OK,row_i + 3,col_i]) * np.ones(filterd_OK.size), 'k--', linewidth=2)
        ax.fill_between(image_numbers[filterd_OK], 
                        (np.median(zc_image[filterd_OK,row_i + 3,col_i] + zc_fit[filterd_OK,row_i + 3,col_i]) - MAD(zc_image[filterd_OK,row_i + 3,col_i] + zc_fit[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        (np.median(zc_image[filterd_OK,row_i + 3,col_i] + zc_fit[filterd_OK,row_i + 3,col_i]) + MAD(zc_image[filterd_OK,row_i + 3,col_i] + zc_fit[filterd_OK,row_i + 3,col_i])) * np.ones(filterd_OK.size),
                        color='magenta', alpha=0.3)
fig_time.tight_layout()
fig_time.savefig(FP.path_base + FP.path_enter + '/pics/zgiven_plus_zfit_time.pdf')

# <codecell>


