# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

expid = 232697
path_mesh = '/Users/cpd/Desktop/Meshes/'

from focal_plane_routines import print_command
import pyfits
import argparse
from subprocess import call
from os import path, makedirs, chdir, system, remove

from focal_plane import FocalPlane
from decamutil_cpd import decaminfo

from focal_plane_routines import MAD
from decam_csv_routines import download_cat, download_image
from routines_plot import process_image, data_focal_plot, data_hist_plot, focal_graph_axis, plot_star
# remake the catalogs...
expids = np.arange(231046, 231053)
expids = np.append(expids, range(231089, 231096))
expids = np.append(expids, range(232608, 232849))
expids = np.append(expids, range(233377, 233571))
expids = np.append(expids, range(233584, 233642))

rids = [20130906105326] * (231053 - 231046) + \
       [20130906105326] * (231096 - 231089) + \
       [20130911103044] * (232849 - 232608) + \
       [20130913151017] * (233571 - 233377) + \
       [20130913151017] * (233642 - 233584)

dates = [20130905] * (231053 - 231046) + \
        [20130905] * (231096 - 231089) + \
        [20130910] * (232849 - 232608) + \
        [20130912] * (233571 - 233377) + \
        [20130912] * (233642 - 233584)

ith = np.argwhere(expid == expids)[0]
rid = rids[ith]
date = dates[ith]

dataDirectory = "/Volumes/Seagate/Images/"
dataDirectory = "/Users/cpd/Desktop/Images/"

args_dict = {'expid': expid,
             'path_mesh': path_mesh,
             'mesh_name': "Science20120915s1v3_134239",
             'output_directory': dataDirectory + '{0:08d}/'.format(expid),
             'catalogs': dataDirectory,
             'rid': rid,
             'date': date,
             'conds': 'default',
             'size': 16,
             }

##############################################################################
# download the image and catalog
##############################################################################

# from wgetscript:
dataDirectoryExp = path.expandvars(dataDirectory + "{0:08d}".format(args_dict['expid']))

# make directory if it doesn't exist
if not path.exists(dataDirectoryExp):
    makedirs(dataDirectoryExp)

# move there!
chdir(dataDirectoryExp)

from __future__ import print_function, division
import numpy as np
import pyfits
import argparse
from subprocess import call
from os import path, makedirs, chdir, system, remove

from focal_plane import FocalPlane
from decamutil_cpd import decaminfo

from focal_plane_routines import MAD
from decam_csv_routines import download_cat, download_image
from routines_plot import process_image

nPixels = args_dict['size']

# <codecell>

def check_centered_peak(stamp_in,
                        background, threshold,
                        xwin_image, ywin_image,
                        x2win_image, xywin_image, y2win_image,
                        return_stamps = False):

    # filter via background and threshold
    stamp = stamp_in.copy()
    stamp -= background
    stamp = np.where(stamp > threshold, stamp, 0)

    stamp = stamp.reshape(nPixels, nPixels)

    # do window
    max_nsig2 = 25
    y, x = np.indices(stamp.shape)
    Mx = nPixels / 2 + xwin_image % int(xwin_image) - 1
    My = nPixels / 2 + ywin_image % int(ywin_image) - 1

    r2 = (x - Mx) ** 2 + (y - My) ** 2

    Mxx = 2 * x2win_image
    Myy = 2 * y2win_image
    Mxy = 2 * xywin_image
    detM = Mxx * Myy - Mxy * Mxy
    Minv_xx = Myy / detM
    TwoMinv_xy = -Mxy / detM * 2.0
    Minv_yy = Mxx / detM
    Inv2Minv_xx = 0.5 / Minv_xx
    rho2 = Minv_xx * (x - Mx) ** 2 + TwoMinv_xy * (x - Mx) * (y - My) + Minv_yy * (y - My) ** 2
    stamp = np.where(rho2 < max_nsig2, stamp, 0)

    # take central 14 x 14 pixels and 5 x 5
    center = nPixels / 2
    if center > 10:
        cutout = stamp[center - 10: center + 10, center - 10: center + 10]
    else:
        cutout = stamp
    central_cutout = stamp[center - 3: center + 3, center - 3: center + 3]

    if return_stamps:
        return cutout, central_cutout
    else:
        return (np.max(cutout) == np.max(central_cutout))

def check_recdata(recdata, return_stamps=True):
    val = check_centered_peak(recdata['STAMP'],
                        recdata['BACKGROUND'], recdata['THRESHOLD'],
                        recdata['XWIN_IMAGE'], recdata['YWIN_IMAGE'],
                        recdata['X2WIN_IMAGE'], recdata['XYWIN_IMAGE'], recdata['Y2WIN_IMAGE'],
                        return_stamps)
    return val

def get_data(data, i):
    ret = {}
    for key in data.keys():
        ret.update({key: data[key][i]})
    return ret

# <codecell>

##############################################################################
# create the new catalogs
##############################################################################
for i in xrange(1, 63):

    if i == 61:
        # screw N30
        continue

    # get cat
    download_cat(args_dict['rid'], args_dict['expid'], args_dict['date'], i)
    # get image
    download_image(args_dict['rid'], args_dict['expid'], args_dict['date'], i)


    # create an FP object
    path_base = args_dict['catalogs']
    list_catalogs_base = \
        path_base + '{0:08d}/DECam_{0:08d}_'.format(args_dict['expid'])
    list_catalogs = [list_catalogs_base + '{0:02d}_cat.fits'.format(i)]
    list_chip = [[decaminfo().ccddict[i]]]
    list_fits_extension = [[2]]

    FP = FocalPlane(list_catalogs=list_catalogs,
                    list_fits_extension=list_fits_extension,
                    list_chip=list_chip,
                    path_mesh=args_dict['path_mesh'],
                    mesh_name=args_dict['mesh_name'],
                    boxdiv=0,
                    max_samples_box=20000,
                    conds=args_dict['conds'],
                    nPixels=nPixels,
                    )

    # now use the catalog here to make a new catalog
    image = pyfits.getdata(list_catalogs_base + '{0:02d}.fits'.format(i),
                           ext=0)
    header = pyfits.getheader(list_catalogs_base + '{0:02d}.fits'.format(i),
                              ext=0)
    gain = np.mean([header['GAINA'], header['GAINB']])
    gain_std = np.std([header['GAINA'], header['GAINB']])
    # generate dictionary of columns
    pyfits_dict = {}
    pyfits_dict.update({
        'XWIN_IMAGE': dict(name='XWIN_IMAGE',
                           array=FP.recdata['XWIN_IMAGE'],
                           format='1D',
                           unit='pixel',),
        'YWIN_IMAGE': dict(name='YWIN_IMAGE',
                           array=FP.recdata['YWIN_IMAGE'],
                           format='1D',
                           unit='pixel',),
        'FLAGS':      dict(name='FLAGS',
                           array=FP.recdata['FLAGS'],
                           format='1I',),
        'BACKGROUND':      dict(name='BACKGROUND',
                           array=FP.recdata['BACKGROUND'],
                           format='1E',
                           unit='count',),
        'FLUX_AUTO':   dict(name='FLUX_AUTO',
                           array=FP.recdata['FLUX_AUTO'],
                           format='1E',
                           unit='counts',),
        'FLUXERR_AUTO':   dict(name='FLUXERR_AUTO',
                           array=FP.recdata['FLUXERR_AUTO'],
                           format='1E',
                           unit='counts',),
        'MAG_AUTO':   dict(name='MAG_AUTO',
                           array=FP.recdata['MAG_AUTO'],
                           format='1E',
                           unit='mag',),
        'MAGERR_AUTO':   dict(name='MAGERR_AUTO',
                           array=FP.recdata['MAGERR_AUTO'],
                           format='1E',
                           unit='mag',),
        'MAG_PSF':   dict(name='MAG_PSF',
                           array=FP.recdata['MAG_PSF'],
                           format='1E',
                           unit='mag',),
        'MAGERR_PSF':   dict(name='MAGERR_PSF',
                           array=FP.recdata['MAGERR_PSF'],
                           format='1E',
                           unit='mag',),
        'FLUX_RADIUS':   dict(name='FLUX_RADIUS',
                           array=FP.recdata['FLUX_RADIUS'],
                           format='1E',
                           unit='pixel',),
        'CLASS_STAR': dict(name='CLASS_STAR',
                           array=FP.recdata['CLASS_STAR'],
                           format='1E',),
        'SPREAD_MODEL':dict(name='SPREAD_MODEL',
                            array=FP.recdata['SPREAD_MODEL'],
                            format='1E',),
        'SPREADERR_MODEL':dict(name='SPREADERR_MODEL',
                               array=FP.recdata['SPREADERR_MODEL'],
                               format='1E',),
        'FWHMPSF_IMAGE': dict(name='FWHMPSF_IMAGE',
                              array=FP.recdata['FWHMPSF_IMAGE'],
                              format='1D',
                              unit='pixel',),

        # include the sextractor ones too
        'X2WIN_IMAGE_SEX': dict(name='X2WIN_IMAGE_SEX',
                            array=FP.recdata['X2WIN_IMAGE'],
                           format='1D',
                           unit='pixel**2',),
        'XYWIN_IMAGE_SEX': dict(name='XYWIN_IMAGE_SEX',
                            array=FP.recdata['XYWIN_IMAGE'],
                           format='1D',
                           unit='pixel**2',),
        'Y2WIN_IMAGE_SEX': dict(name='Y2WIN_IMAGE_SEX',
                            array=FP.recdata['Y2WIN_IMAGE'],
                           format='1D',
                           unit='pixel**2',),
        'FWHM_WORLD_SEX': dict(name='FWHM_WORLD_SEX',
                            array=FP.recdata['FWHM_WORLD'],
                           format='1D',
                           unit='deg',),

        'SN_FLUX': dict(name='SN_FLUX',
                           array=2.5 / np.log(10) / FP.recdata['MAGERR_AUTO'],
                           format='1E',),

        # to chip number
        'CHIP': dict(name='CHIP',
                     array=np.ones(FP.recdata['MAGERR_AUTO'].size) * i,
                     format='1I'),

        'X2WIN_IMAGE': dict(name='X2WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**2',),
        'XYWIN_IMAGE': dict(name='XYWIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**2',),
        'Y2WIN_IMAGE': dict(name='Y2WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**2',),
        'X3WIN_IMAGE': dict(name='X3WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'X2YWIN_IMAGE': dict(name='X2YWIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'XY2WIN_IMAGE': dict(name='XY2WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'Y3WIN_IMAGE': dict(name='Y3WIN_IMAGE',
                           array=[],
                           format='1D',
                           unit='pixel**3',),
        'FWHM_WORLD': dict(name='FWHM_WORLD',
                           array=[],
                           format='1D',
                           unit='deg',),

        'THRESHOLD':      dict(name='THRESHOLD',
                           array=[],
                           format='1E',
                           unit='count',),
        'WHISKER': dict(name='WHISKER',
                           array=[],
                           format='1D',
                           unit='pixel',),
        'FLUX_ADAPTIVE': dict(name='FLUX_ADAPTIVE',
                           array=[],
                           format='1D',
                           unit='counts',),
        'A4_ADAPTIVE': dict(name='A4_ADAPTIVE',
                           array=[],
                           format='1D',
                           unit='pixel**4',),

        'STAMP':      dict(name='STAMP',
                           array=[],
                           format='{0}D'.format(nPixels ** 2),
                           unit='count')
        })

    # go through each entry and make stamps and other parameters
    for recdata in FP.recdata:

        x_center = recdata['X' + FP.coord_name]
        y_center = recdata['Y' + FP.coord_name]

        # coords are y,x ordered
        y_start = int(y_center - nPixels / 2)
        y_end = y_start + nPixels
        x_start = int(x_center - nPixels / 2)
        x_end = x_start + nPixels

        stamp = image[y_start:y_end, x_start:x_end].astype(np.float64)

        background = recdata['BACKGROUND']
        # only cut off at one std; it seems like 2 std actually biases the
        # data...
        # 1 std and esp 2 std are plenty for 16 x 16...
        threshold = np.sqrt(background / gain)
        moment_dict = FP.moments(stamp,
                                 background=background,
                                 threshold=threshold,
                                 )

        # append to pyfits_dict
        pyfits_dict['STAMP']['array'].append(
            stamp.flatten())
        pyfits_dict['THRESHOLD']['array'].append(
            threshold)
        pyfits_dict['X2WIN_IMAGE']['array'].append(
            moment_dict['x2'])
        pyfits_dict['Y2WIN_IMAGE']['array'].append(
            moment_dict['y2'])
        pyfits_dict['XYWIN_IMAGE']['array'].append(
            moment_dict['xy'])
        pyfits_dict['X3WIN_IMAGE']['array'].append(
            moment_dict['x3'])
        pyfits_dict['Y3WIN_IMAGE']['array'].append(
            moment_dict['y3'])
        pyfits_dict['X2YWIN_IMAGE']['array'].append(
            moment_dict['x2y'])
        pyfits_dict['XY2WIN_IMAGE']['array'].append(
            moment_dict['xy2'])
        pyfits_dict['FWHM_WORLD']['array'].append(
            moment_dict['fwhm'])

        pyfits_dict['WHISKER']['array'].append(
            moment_dict['whisker'])
        pyfits_dict['FLUX_ADAPTIVE']['array'].append(
            moment_dict['flux'])
        pyfits_dict['A4_ADAPTIVE']['array'].append(
            moment_dict['a4'])

    # almost universally, any a4 > 0.3 is either a completely miscentered
    # image or a cosmic (though there are some of both that have a4 < 0.3!)
    conds = ((np.array(pyfits_dict['A4_ADAPTIVE']['array']) < 0.15) *
             (np.array(pyfits_dict['A4_ADAPTIVE']['array']) > -0.01) *
             (np.array(pyfits_dict['FLUX_ADAPTIVE']['array']) > 10))
    # cull crazy things
    mad_keys =  ['X2WIN_IMAGE', 'XYWIN_IMAGE', 'Y2WIN_IMAGE',
                 'X3WIN_IMAGE', 'X2YWIN_IMAGE', 'XY2WIN_IMAGE', 'Y3WIN_IMAGE',
                 'A4_ADAPTIVE', 'WHISKER', 'FWHM_WORLD', 'FLUX_RADIUS']
    for key in mad_keys:
        conds *= MAD(pyfits_dict[key]['array'], sigma=5)[0]
    # cut in the stamps
    for ij in xrange(conds.size):
        conds[ij] *= check_centered_peak(pyfits_dict['STAMP']['array'][ij],
                                        pyfits_dict['BACKGROUND']['array'][ij],
                                        pyfits_dict['THRESHOLD']['array'][ij],
                                        pyfits_dict['XWIN_IMAGE']['array'][ij],
                                        pyfits_dict['YWIN_IMAGE']['array'][ij],
                                        pyfits_dict['X2WIN_IMAGE']['array'][ij],
                                        pyfits_dict['XYWIN_IMAGE']['array'][ij],
                                        pyfits_dict['Y2WIN_IMAGE']['array'][ij],)
    # create the columns
    columns = []
    for key in pyfits_dict:
        pyfits_dict[key]['array'] = np.array(pyfits_dict[key]['array'])[conds]
        columns.append(pyfits.Column(**pyfits_dict[key]))
    tbhdu = pyfits.new_table(columns)

    tbhdu.writeto(args_dict['output_directory'] + \
                  'DECam_{0:08d}_'.format(args_dict['expid']) + \
                  '{0:02d}_cat_cpd.fits'.format(i),
                  clobber=True)

    if i > 1:
        # remove image
        remove("DECam_{0:08d}_{1:02d}.fits".format(args_dict['expid'], i))

# <codecell>

# make FocalPlaneShell object
from focal_plane import FocalPlane
from focal_plane_routines import minuit_dictionary, mean_trim
from decam_csv_routines import generate_hdu_lists
from minuit_fit import Minuit_Fit
from routines_plot import data_focal_plot, data_hist_plot

path_mesh = '/Users/cpd/Desktop/Meshes/'
list_catalogs, list_fits_extension, list_chip = \
        generate_hdu_lists(expid,
                           #path_base='/Users/cpd/Desktop/Catalogs/')
                           path_base='/Users/cpd/Desktop/Images/')

boxdiv = 0
max_samples_box = 5
average = mean_trim

FP = FocalPlane(list_catalogs=list_catalogs,
                list_fits_extension=list_fits_extension,
                list_chip=list_chip,
                boxdiv=boxdiv,
                max_samples_box=5,
                conds='default',
                average=average,
                path_mesh=path_mesh,
                nPixels=nPixels,
                )
poles = FP.data

# make coords
edges = FP.decaminfo.getEdges(boxdiv=boxdiv)

coords = FP.random_coordinates(max_samples_box=max_samples_box, boxdiv=boxdiv)
coords = coords[coords[:,2] <= np.max(FP.coords[:,2])]
coords = FP.coords

p = {'dx': -234.491545,
     'dy':  293.39127,
     'dz':  29.452629,
     'xt':  0.772277,
     'yt':  9.760292,
     'z05d':  0.049454,
     'z06d': -0.246456,
     'z07x': -0.000812,
     'z07y':   7.50000000e-05,
     'z08x':   3.00000000e-06,
     'z08y': -0.000618,
     'z09d': -0.06377,
     'z10d':  0.049792,
     'rzero': 0.125,
     'e1': 0,
     'e2': 0,
     }
FP.chi_weights = {
    'e0': 1.,
    'e1': 1.,
    'e2': 1.,
    'delta1': 1.,
    'delta2': 1.,
    'zeta1': 1.,
    'zeta2': 1.,
    }

chi2hist = {key:[] for key in FP.chi_weights}
p_hist = {key:[] for key in p}
f_list = []
FP.temp = {}
FP.temp_old = {}
output_directory = '/Users/cpd/Desktop/fits2/'

# <codecell>

# make the fit function
def FP_func(dz, e1, e2, rzero,
            dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y, z09d, z10d,
            ):

    in_dict_FP_func = locals().copy()

    for key in in_dict_FP_func:
        p_hist[key].append(in_dict_FP_func[key])
    
    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict_FP_func.keys():
        if np.isnan(in_dict_FP_func[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FPS.remakedonut()
            return 1e20

    poles_i = FP.plane_averaged(in_dict_FP_func, coords=coords, average=average, boxdiv=boxdiv)
    poles_i['e1'] += e1
    poles_i['e2'] += e2
    
    FP.temp, FP.temp_old = poles_i, FP.temp
    chi2 = 0
    for key in FP.chi_weights:
        val_a = poles[key]
        val_b = poles_i[key]
        var = poles['var_{0}'.format(key)]
        weight = FP.chi_weights[key]
        
        chi2_i = np.square(val_a - val_b) / var
        chi2hist[key].append(chi2_i)
        chi2 += np.sum(weight * chi2_i)
    
    if (chi2 < 0) + (np.isnan(chi2)):
        chi2 = 1e20
        
    # update the chi2 by *= 1. / (Nobs - Nparam)
    chi2 *= 1. / (len(poles_i['e1']) * sum([FP.chi_weights[i] for i in FP.chi_weights]) -
                  len(in_dict_FP_func.keys()))
    return chi2

# <codecell>

# let's plot all the stars because why not
for i in xrange(0, FP.recdata.size, int(max_samples_box / 2)):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    fig, ax = plot_star(FP.recdata[i], nPixels=nPixels)
    ax.set_title('{0}, {1}'.format(i, FP.recdata[i]['CHIP']))
    fig.savefig(output_directory + 'stars_{0:04d}.png'.format(i))
    plt.close('all')

# <codecell>

def SaveFunc(steps,
             dz, e1, e2, rzero,
            dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y, z09d, z10d,
            ):
    
    in_dict_FP_func = locals().copy()
    
    poles_i = FP.temp
    # compare
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex='all', sharey='all', figsize=(12*4, 12*2))
    figures = {'e': fig, 'w': fig, 'zeta': fig, 'delta': fig}
    focal_keys = figures.keys()
    for ij in range(axs.shape[0]):
        for jk in range(axs.shape[1]):
            axs[ij][jk] = focal_graph_axis(axs[ij][jk])
    axes = {'e': axs[0,0], 'w': axs[0,1], 'delta': axs[1,0], 'zeta': axs[1,1]}
    figures_hist = {'e0subs': fig, 'e0': fig, 'e0prime': fig}
    axes_hist = {'e0subs': axs[1,3], 'e0': axs[0,2], 'e0prime': axs[1,2]}
    figures, axes, scales = data_focal_plot(poles,
                                            color='r', boxdiv=boxdiv,
                                            figures=figures, axes=axes,
                                            keys=focal_keys,
                                            )
    # plot the comparison
    figures, axes, scales = data_focal_plot(poles_i,
                                            color='b', boxdiv=boxdiv,
                                            figures=figures, axes=axes, scales=scales,
                                            keys=focal_keys,
                                            )

    poles_i.update({
                  'e0subs': (poles['e0'] - poles_i['e0']) / poles['e0'],
                  'e0prime': poles_i['e0'],
                  'e0': poles['e0']
                  })
    figures_hist, axes_hist, scales_hist = data_hist_plot(poles_i, edges,
                                                          figures=figures_hist,
                                                          axes=axes_hist,
                                                          keys=['e0subs', 'e0', 'e0prime'])
    
    # make tables for param and delta and chi2
    colLabels = ("Parameter", "Value", "Delta")
    cellText = [[key, '{0:.3e}'.format(p_hist[key][-1]),
                 '{0:.3e}'.format(p_hist[key][-1] - p_hist[key][-2])] for key in p.keys()]
    # add in chi2s
    cellText += [[key, 
                  '{0:.3e}'.format(np.sum(chi2hist[key][-1])), 
                  '{0:.3e}'.format(np.sum(chi2hist[key][-1] - chi2hist[key][-2]))] 
                 for key in chi2hist.keys()]
    chi2 = 0
    chi2old = 0
    for key in FP.chi_weights:
        weight = FP.chi_weights[key]
        chi2 += np.sum(weight * chi2hist[key][-1])
        chi2old += np.sum(weight * chi2hist[key][-2])
    chi2delta = chi2 - chi2old
    cellText += [['total chi2', '{0:.3e}'.format(chi2), '{0:.3e}'.format(chi2delta)]]
    axs[0,3].axis('off')
    table = axs[0,3].table(cellText=cellText, colLabels=colLabels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1, 2)
    

    plt.tight_layout()
    fig.savefig(output_directory + '{0:04d}.pdf'.format(steps))
    plt.close('all')
    del fig, axs, figures, axes, figures_hist, axes_hist

# <codecell>

# fit
# TODO: set option to let minuit do the derivative itself
par_names = p.keys()
verbosity = 3
force_derivatives = 1
strategy = 1
tolerance = 40
h_base = 1e-3
save_iter = 1
max_iterations = len(par_names) * 1000
minuit_results_list = []
# set up initial guesses
minuit_dict, h_dict = minuit_dictionary(par_names, h_base=h_base)
# for pkey in p:
#     minuit_dict[pkey] = p[pkey]
# do fit
minuit_fit = Minuit_Fit(FP_func, minuit_dict, par_names=par_names,
                        SaveFunc=SaveFunc,
                        save_iter=save_iter,
                        h_dict=h_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)

# <codecell>

# do e0
# TODO: maybe cut the dz out of here? the other params matter too
# (though a lil less than dz) so maybe no need to include it here...
for key in par_names:
    if (key != 'rzero') * (key != 'dz'):
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
    else:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False

for key in FP.chi_weights:
    if (key != 'e0'):
        FP.chi_weights[key] = 0
    else:
        FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.doFit()

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)
# do delta1
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    else:
        # use the errors from our fixed ones.
        minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]

    minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
        
    if (key != 'z10d'):
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
    else:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False

for key in FP.chi_weights:
    if (key != 'delta1'):
        FP.chi_weights[key] = 0
    else:
        FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls + 1000
minuit_fit.doFit()

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)
# do delta2
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    else:
        # use the errors from our fixed ones.
        minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]

    minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]

    if (key != 'z09d'):
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
    else:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False

for key in FP.chi_weights:
    if (key != 'delta2'):
        FP.chi_weights[key] = 0
    else:
        FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls + 1000
minuit_fit.doFit()

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)
# do zeta1
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    else:
        # use the errors from our fixed ones.
        minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]

    minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]

    if (key != 'z08d'):
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
    else:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False

for key in FP.chi_weights:
    if (key != 'zeta1'):
        FP.chi_weights[key] = 0
    else:
        FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls + 1000
minuit_fit.doFit()

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)
# do zeta2
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    else:
        # use the errors from our fixed ones.
        minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]

    minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]

    if (key != 'z07d'):
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
    else:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False

for key in FP.chi_weights:
    if (key != 'zeta2'):
        FP.chi_weights[key] = 0
    else:
        FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls + 1000
minuit_fit.doFit()

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)
# do e's
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    else:
        # use the errors from our fixed ones.
        minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]

    minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]

    if (key != 'dz') * (key != 'dx') * (key != 'dy') * (key != 'xt') * (key != 'yt') * (key != 'e1') * (key != 'e2'):
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
    else:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False

for key in FP.chi_weights:
    if (key != 'e1') * (key != 'e2'):
        FP.chi_weights[key] = 0
    else:
        FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls + 1000
minuit_fit.doFit()

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)
# do the generic fit:
# defix and update the unfixed
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    else:
        # use the errors from our fixed ones.
        minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]

    minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
for key in FP.chi_weights:
    FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls + 1000
minuit_fit.doFit()

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)

# <codecell>

minuit_results = minuit_fit.outFit()

# <codecell>

for key in sorted(minuit_results['args']):
    print(key, p[key], minuit_results['args'][key], minuit_results['errors'][key])

# <codecell>

for key in sorted(minuit_results):
    print(key, minuit_results[key])

# <codecell>

for key in sorted(minuit_dict):
    print(key, minuit_dict[key])

# <codecell>

# compare
figures, axes, scales = data_focal_plot(poles,
                                        color='r', boxdiv=boxdiv,
                                        )

# plot the comparison
poles_i = FP.plane_averaged(minuit_results['args'], coords=coords, average=average, boxdiv=boxdiv)
poles_i['e1'] += minuit_results['args']['e1']
poles_i['e2'] += minuit_results['args']['e2']
figures, axes, scales = data_focal_plot(poles_i,
                                        color='b', boxdiv=boxdiv,
                                        figures=figures, axes=axes, scales=scales,
                                        )
poles_comp = {'x_box': poles['x_box'], 'y_box': poles['y_box'],
              'e0_diff': poles['e0'] - poles_i['e0'],
              }
other_keys = ['e0_diff']
figures_hist, axes_hist, scales_hist = data_hist_plot(poles_comp, edges, keys=other_keys,
                                        )

# <codecell>

# plot chi2s
plt.figure(figsize=(12,12), dpi=300)
for key in sorted(chi2hist.keys()):
    chis = chi2hist[key]
    x = range(len(chis))
    y = []
    for i in x:
        y.append(np.sum(chis[i]) * FP.chi_weights[key])
    plt.plot(x, np.log10(y), label=key)
plt.legend()

# <codecell>

# plot values
for key in sorted(p_hist.keys()):
    plt.figure(figsize=(12,12), dpi=300)
    x = range(len(p_hist[key]))
    y = []
    for i in x:
        y.append(np.sum(p_hist[key][i]))
    plt.plot(x, y, label=key)
    plt.title('{0}'.format(key))

# <codecell>


