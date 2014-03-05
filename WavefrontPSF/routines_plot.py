#!/usr/bin/env python
"""
File: routines_plot.py
Author: Chris Davis
Description: Methods for plotting various wavefront configurations.

"""

from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from subprocess import call
from os import path, makedirs
from routines_moments import ellipticity_to_whisker

# color scheme:
# blues_r for all <0 data, reds for all >0 data
from colors import blue_red, blues_r, reds
BASE_SIZE = 12
def focal_graph():
    """Convenience for the creation of my focal plane graphics

    Returns
    -------
    focal_figure : figure object

    focal_axis : axis object

    """
    focal_figure = plt.figure(figsize=(BASE_SIZE, BASE_SIZE), dpi=300)
    focal_axis = focal_figure.add_subplot(111,
                                          aspect='equal')
    focal_axis = focal_graph_axis(focal_axis)
    return focal_figure, focal_axis

def focal_graph_axis(focal_axis):
    """Convenience for the creation of my focal plane graphics

    Returns
    -------
    focal_axis : axis object

    """
    focal_axis.set_xlabel('$X$ [mm] (East)')
    focal_axis.set_ylabel('$Y$ [mm] (South)')
    focal_axis.set_xlim(-250, 250)
    focal_axis.set_ylim(-250, 250)
    return focal_axis

def histogramize(x, y, z, bins):
    counts, xedges_counts, yedges_counts = np.histogram2d(x, y, bins=bins)

    weighted_counts, xedges, yedges = np.histogram2d(x, y,
        weights=z, bins=bins)

    C = weighted_counts.T / counts.T
    C = np.where(counts.T == 0, np.nanmin(C), C)
    #C = np.ma.masked_invalid(weighted_counts.T / counts.T)


    X, Y = np.meshgrid(0.5 * (xedges[:-1] + xedges[1:]),
                       0.5 * (yedges[:-1] + yedges[1:]))

    return C, X, Y

def contour3d(x, y, z, bins, stride=2):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z, X, Y = histogramize(x, y, z, bins)
    ax = focal_graph_axis(ax)

    ax.plot_surface(X, Y, Z, rstride=stride, cstride=stride,
                    alpha=0.3, cmap=blue_red)

    # add other plots
    ## cset = ax.contour(X, Y, Z, zdir='x', offset=-250, cmap=blue_red)
    ## cset = ax.contour(X, Y, Z, zdir='y', offset=250, cmap=blue_red)
    cset = ax.add_collection3d(plt.pcolor(X, Y, Z, cmap=blue_red),
                               zs=Z.min(), zdir='z')
    ax.set_zlim(Z.min(), Z.max())
    ax.view_init(elev=40, azim=-10)
    return fig, ax

def wedge_collection(X, Y, U, V,
                     U_var, V_var,
                     scale=1,
                     artpatch=0,
                     patch_dict={},
                     color='b'):

    """A fairly kludgy way of plotting uncertainties on whisker plots.

    Parameters
    ----------
    X, Y : array
        Location on focal plane (in mm) of each whisker.

    U, V : array
        Cartesian whisker length.

    U_var, V_var : array or float
        The variance in your U and V whiskers, either one value for all, or
        one value per whisker.

    scale : float, optional
        Conversion from UVunits to mm.
        Default is 1.

    artpatch : integer, optional
        Sets the spin of a given plot. IE if artpatch = 2, then we have a
        spin-2 object like a whisker and there will be wedges on both sides.
        Note: ONLY if artpatch = 1 will the arrows be on the quivers.

    patch_dict : dictionary, optional
        Set of parameters to pass to PatchCollection. In case you want to.
        Default just adds some alpha (not sure if this actually works).

    color : string, optional
        Color of the whiskers. Default is blue 'b'

    Returns
    -------
    collection : PatchCollection
        A matplotlib object containing the wedges. This can then be added
        to your figure via ax.add_collection(collection)

    """

    R2 = U ** 2 + V ** 2
    R = np.sqrt(R2)
    theta = np.arctan2(V, U) * 180. / np.pi
    sigma_theta = np.sqrt((V / R2) ** 2 * U_var + (U / R2) ** 2 * V_var) * \
        180. / np.pi
    sigma_r = np.sqrt(U * U / R2 * U_var + V * V / R2 * V_var)
    patches = []

    delta_theta = 360 / artpatch

    for i in range(X.size):
        for j in range(artpatch):
            theta_l = theta[i] - 0.5 * sigma_theta[i]
            theta_u = theta_l + sigma_theta[i]

            # # we want the slice, not the rest of the annulus.
            # if theta_u < theta_l:
            #     theta_l, theta_u = theta_u, theta_l

            # it CAN be possible to be more than 360 degress unsure
            if sigma_theta[i] > 360:
                theta_u = 360
                theta_l = 0
            #TODO: should this be / scale or * scale?
            RU = (R[i] + 0.5 * sigma_r[i]) / scale
            wid2 = sigma_r[i] / scale
            if RU - wid2 < 0:
                wid2 = None
            try:
                art = Wedge([X[i], Y[i]], RU,
                            theta_l, theta_u, width=wid2,
                            color=color,
                            alpha=0.4)
            except ValueError:
                #import ipdb; ipdb.set_trace()
                print('error in wedge:', X[i], Y[i], RU, theta_l, theta_u,
                      wid2)
                # occasionally python craps out? I'm not sure what's going
                # on but ignoring the error for now has worked
                #pass
            patches.append(art)

            theta[i] += delta_theta

    collection = PatchCollection(patches, match_original=True,
                                 **patch_dict)

    return collection


def focal_plane_plot(x, y,
                     u, v,
                     u_ave=0, v_ave=0,
                     u_var=[], v_var=[],
                     focal_figure=None, focal_axis=None,
                     quiverkey_val=-1,
                     quiver_dict=None,
                     color='k',
                     scale=-1, artpatch=2,
                     whisker_width=4,
                     offset_x=0):

    """Create the figure and axis for the whisker plot

    Parameters
    ----------
    x, y : array
        Location of the whiskers on the focal plane (in mm).

    u, v : array
        x and y directions of the quiver you are making.

    u_ave, v_ave : array
        The vector average of the u and v objects (ie average of whatever your
        fundamental quantity is, say e1 and e2, then converted to u and v)

    u_var, v_var : array, optional
        Variance for each of the parameters.  You don't need to include these;
        if they are not included, then no wedges are made.

    scale : float, optional
        Scale your whisker size. The scale is directly proportional to whisker
        length, so that scale=2 will have whiskers 2x as long, and scale=0.5
        will have whiskers half as long.

        The default scaling is 0.04 arcsec per 10 mm or (250 mm / arcsec)**-1 .
        This is probably too large for the poorer nights.

    whisker_width : float, optional
        Sets the thickness of the whisker (in mm).

        Default is 2.5

    quiverkey_dict : dictionary, optional
        Common dictionary commands for the quiverkey (color, text, value)

    quiver_dict : dictionary, optional
        Common dictionary commands for the quiver (color, text, value)

    color : string, optional
        Color of the whiskers. Default is black 'k'

    artpatch : integer, optional
        Sets the spin of a given plot. IE if artpatch = 2, then we have a
        spin-2 object like a whisker and there will be wedges on both sides.
        Note: ONLY if artpatch = 1 will the arrows be on the quivers.

    focal_figure : matplotlib figure, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    focal_axis : matplotlib axis, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    offset_x : float, optional
        A constant offset to all x coordinates

    Returns
    -------
    focal_figure : matplotlib figure
        matplotlib figure instance, which you then could use to e.g. save via
        focal_figure.savefig('/path/to/output.pdf')

    focal_axis : matplotlib axis
        matplotlib axis instance, which you then can modify in the typical
        ways, e.g. adding a title would be focal_axis.title('foo')

    Notes
    -----
    The docs for plt.quiver are /extremely/ useful.

    References
    ----------
    http://des-docdb.fnal.gov:8080/cgi-bin/RetrieveFile?docid=5282

    """
    r = np.sqrt(u ** 2 + v ** 2)
    if (scale <= 0) * (artpatch % 2 != 0):
        scale = np.minimum(np.percentile(r, 90) / 15.,
                           np.percentile(r, 5) / 6.)
    elif (scale <= 0) * (artpatch % 2 == 0):
        # add an extra factor of 2 to equate to the rotation because of my
        # method for plotting whiskers goes from the tail and then adds another
        # one on the other side 180 degrees away
        scale = np.minimum(np.percentile(r, 90) / 7.5,
                           np.percentile(r, 5) / 3.)

    if not quiver_dict:
        quiver_dict = {}
    if quiverkey_val <= 0:
        #quiverkey_val = np.minimum(np.mean(r), np.median(r))
        # plot such that the length of the arrow is always the same mm
        quiverkey_val = scale * 20
    quiverkey_label = r'${0:.2e}$'.format(quiverkey_val)

    # make the figure
    if (not focal_figure) * (not focal_axis):
        focal_figure, focal_axis = focal_graph()

    # add quiver
    quiver_dict_default = dict(alpha=0.5,
                               angles='uv',
                               color=color,
                               headlength=0,
                               headwidth=1,
                               minlength=0,
                               pivot='tail',
                               scale=scale,
                               scale_units='xy',
                               units='xy',
                               width=whisker_width)

    if artpatch%2 != 0:
        #quiver_dict_default['headwidth'] = 0
        quiver_dict_default['headlength'] = 1

    for key in quiver_dict_default:
        if key not in quiver_dict:
            quiver_dict.update({key: quiver_dict_default[key]})

    for i in range(artpatch):

        Q = focal_axis.quiver(x + offset_x, y, u, v,
                              **quiver_dict)

        # update angles
        delta_theta = 2 * np.pi / artpatch
        u, v = u * np.cos(delta_theta) - v * np.sin(delta_theta), \
               u * np.sin(delta_theta) + v * np.cos(delta_theta)

        if (u_ave != 0) * (v_ave != 0):
            # add cornervalue in upper right
            focal_axis.quiver(200 + offset_x, 200,
                              u_ave, v_ave,
                              **quiver_dict)

            # update angls
            u_ave, v_ave = \
                u_ave * np.cos(delta_theta) - v_ave * np.sin(delta_theta), \
                u_ave * np.sin(delta_theta) + v_ave * np.cos(delta_theta)

    # add quiverkey in lower left
    focal_axis.quiverkey(Q, -200, -200,
                         quiverkey_val,
                         quiverkey_label,
                         coordinates='data',
                         color='k')

    if len(u_var) > 0:
        # add wedges
        wedges = wedge_collection(x + offset_x, y, u, v,
                                  u_var, v_var,
                                  scale=scale,
                                  artpatch=artpatch,
                                  color=color)
        focal_axis.add_collection(wedges)

    return focal_figure, focal_axis, scale


def collect_images(
        file_list,
        output_directory,
        graphs_list=['ellipticity', 'whisker',
                     'whisker_rotated'],
        rate=0):

    """collect the images made into big files

    Parameters
    ----------
    file_list : list
        A list of directories containing the graphs we want to merge

    output_directory : string
        Where we will output the collected images.

    graphs_list : list, optional
        The names of the files we want to merge. Default finds the comparison
        E1E2, whisker, and whisker_rotated.

    rate : integer, optional
        If given and bigger than zero, sets the number of images displayed in a
        movie per second.

    """

    # graphs_list is a list of all the file names created in the above
    # graph_routine. We will use this list to merge all the pdfs into one
    # super pdf and an mp4 as well
    if len(file_list) > 0:
        for graph_i in graphs_list:
            merged_file = output_directory + '{0}'.format(graph_i)
            if not path.exists(path.dirname(merged_file)):
                makedirs(path.dirname(merged_file))
            command = [#'bsub', '-q', 'short', '-o', path_logs,
                       'gs',
                       '-dNOPAUSE', '-dBATCH', '-dSAFER', '-q',
                       '-sDEVICE=pdfwrite',
                       '-sOutputFile={0}.pdf'.format(merged_file)]
            # append all the files
            for file_i in file_list:
                command.append(file_i + graph_i + '.png')
            # call the command
            call(command)

            if rate > 0:
                # also do movie stuff

                ''' gs -dBATCH -dNOPAUSE -sDEVICE=jpeg -r300 -dJPEGQ=100
                -sOutputFile='comparison_focus-%000d.jpg'
                comparison_focus.pdf . first r gives 2 images per second;
                ie each image must now last 0.5 seconds next r gives the
                total rate for the movie (otherwise, modifying that second
                r won't actually change the real rate of image display)
                ffmpeg -f image2 -r 10 -i comparison_focus-%d.jpg -r 30 -an
                -q:v 0 comparison_focus.mp4 find . -name '*.jpg' -delete
                '''

                # convert the pdf to jpegs
                command = [#'bsub', '-q', 'short', '-o', path_logs,
                           'gs', '-sDEVICE=jpeg',
                           '-dNOPAUSE', '-dBATCH', #'-dSAFER',
                           '-r300', '-dJPEGQ=100',
                           "-sOutputFile={0}-%000d.jpg".format(merged_file),
                           merged_file + '.pdf']
                call(command, bufsize=-1)

                # now convert the jpegs to a video
                command = [#'bsub', '-q', 'short', '-o', path_logs,
                           'ffmpeg', '-f', 'image2',
                           '-r', '{0}'.format(rate),
                           '-i', '{0}-%d.jpg'.format(merged_file),
                           '-r', '30', '-an', '-q:v', '0',
                           merged_file + '.mp4']
                call(command, bufsize=0)

                # delete the jpegs
                command = [#'bsub', '-q', 'short', '-o', path_logs,
                           'find', output_directory,
                           '-name', '*.jpg', '-delete']
                call(command, bufsize=0)

def data_focal_plot(data, color='k',
                    average=np.mean,
                    scalefactor=1,
                    scales=None,
                    figures=None,
                    axes=None,
                    defaults=True,
                    keys=[]):

    """Takes data and makes a bunch of wavefront plots

    Parameters
    ----------
    data : dictionary of arrays
        Contains the relevant parameters which we are making these graphs

    color : string
        Color of the focal plots we make

    average : function
        Function to be used for determining the average

    scales : dictionary of floats
        The values we use to make our plots

    scalefactor : float
        How much do we wish to change the size of the whiskers?
        greater than 1 = bigger
        less than 1 = smaller

    keys : list of keys
        Contains a list of the keys we want figures for.

    Returns
    -------
    figures, axes : dictionaries of matplotlib objects
        The figures and axes made from this function, or updated if already
        supplied.

    scales : dictionary of floats
        The values of scales used.

    """

    if len(keys) == 0:
        keys = ['e', 'w', 'zeta', 'delta'] # , 'wd']

    if not figures:
        figures = {}
        for key in (keys):
            figures.update({key: None})
    if not axes:
        axes = {}
        for key in (keys):
            axes.update({key: None})
    if not scales:
        scales = {}
        for key in (keys):
            if defaults:
                if key == 'e':
                    keyval = 5e-4 / scalefactor
                elif key == 'w':
                    keyval = 1.0e-2 / scalefactor
                elif key == 'zeta':
                    keyval = 7.5e-5 / scalefactor
                elif key == 'delta':
                    keyval = 1.25e-4 / scalefactor
                else:
                    keyval = -1
            else:
                keyval = -1
            scales.update({key: keyval})

    if ('x_box' in data) * ('y_box' in data):
        x = data['x_box']
        y = data['y_box']
    else:
        x = data['x']
        y = data['y']

    # plots
    if ('e1' in data) * ('e2' in data) * ('e' in keys):
        u = data['e1']
        v = data['e2']
        if ('var_e1' in data) * ('var_e2' in data):
            u_var = data['var_e1']
            v_var = data['var_e2']
        else:
            u_var = []
            v_var = []
        u_ave = average(u)
        v_ave = average(v)

        e_figure, e_axis, e_scale = focal_plane_plot(
            x, y, u, v,
            u_var=u_var, v_var=v_var,
            u_ave=u_ave, v_ave=v_ave, artpatch=1,
            color=color, scale=scales['e'],
            focal_figure=figures['e'], focal_axis=axes['e'])
        e_axis.set_title('e')
        figures.update(dict(e=e_figure))
        axes.update(dict(e=e_axis))
        scales.update(dict(e=e_scale))

    if (('e1' in data) * ('e2' in data) *
        ('w1' in data) * ('w2' in data)) * ('w' in keys):
        u = data['w1']
        v = data['w2']
        if ('var_w1' in data) * ('var_w2' in data):
            u_var = data['var_w1']
            v_var = data['var_w2']
        else:
            u_var = []
            v_var = []
        u_ave, v_ave = ellipticity_to_whisker(
            average(data['e1']), average(data['e2']))[:2]

        w_focal_figure, w_axis, w_scale = focal_plane_plot(
            x, y, u, v,
            u_var=u_var, v_var=v_var,
            u_ave=u_ave, v_ave=v_ave, artpatch=2,
            color=color, scale=scales['w'],
            focal_figure=figures['w'], focal_axis=axes['w'])
        w_axis.set_title('w')
        figures.update(dict(w=w_focal_figure))
        axes.update(dict(w=w_axis))
        scales.update(dict(w=w_scale))

    if ('zeta1' in data) * ('zeta2' in data) * ('zeta' in keys):
        u = data['zeta1']
        v = data['zeta2']
        if ('var_zeta1' in data) * ('var_zeta2' in data):
            u_var = data['var_zeta1']
            v_var = data['var_zeta2']
        else:
            u_var = []
            v_var = []
        u_ave = average(u)
        v_ave = average(v)

        zeta_focal_figure, zeta_axis, zeta_scale = focal_plane_plot(
            x, y, u, v, artpatch=1,
            u_var=u_var, v_var=v_var,
            u_ave=u_ave, v_ave=v_ave,
            color=color, scale=scales['zeta'],
            focal_figure=figures['zeta'], focal_axis=axes['zeta'])
        zeta_axis.set_title('zeta')
        figures.update(dict(zeta=zeta_focal_figure))
        axes.update(dict(zeta=zeta_axis))
        scales.update(dict(zeta=zeta_scale))

    if ('delta1' in data) * ('delta2' in data) * ('delta' in keys):
        u = data['delta1']
        v = data['delta2']
        if ('var_delta1' in data) * ('var_delta2' in data):
            u_var = data['var_delta1']
            v_var = data['var_delta2']
        else:
            u_var = []
            v_var = []
        u_ave = average(u)
        v_ave = average(v)

        delta_focal_figure, delta_axis, delta_scale = focal_plane_plot(
            x, y, u, v, artpatch=1,
            u_var=u_var, v_var=v_var,
            u_ave=u_ave, v_ave=v_ave,
            color=color, scale=scales['delta'],
            focal_figure=figures['delta'], focal_axis=axes['delta'])
        delta_axis.set_title('delta')
        figures.update(dict(delta=delta_focal_figure))
        axes.update(dict(delta=delta_axis))
        scales.update(dict(delta=delta_scale))

    if ('wd1' in data) * ('wd2' in data) * ('wd' in keys):
        u = data['wd1']
        v = data['wd2']

        delta_focal_figure, delta_axis, wdelta_scale = focal_plane_plot(
            x, y, u, v, artpatch=3,
            color=color, scale=scales['wd'],
            focal_figure=figures['wd'], focal_axis=axes['wd'])
        delta_axis.set_title('wd')
        figures.update(dict(wd=delta_focal_figure))
        axes.update(dict(wd=delta_axis))
        scales.update(dict(wd=wdelta_scale))

    return figures, axes, scales

def data_hist_plot(data, edges, scales=None,
                   figures=None, axes=None,
                   defaults=True,
                   keys=[]):
    """Takes data and makse a bunch of 2d histograms

    Parameters
    ----------
    data : dictionary of arrays
        Contains the relevant parameters which we are making these graphs

    edges : list or array
        Contains the x edges in edges[0] and y edges in edges[1]

    keys : list of keys
        Contains a list of the keys we want figures for. Can include OTHER
        objects OTHER than the ones normally made.

    Returns
    -------
    figures, axes : dictionaries of matplotlib objects
        The figures and axes made from this function.

    """

    easy_plots = ['flux', 'number']
    separable_pos_plots = ['e0', 'whisker', 'fwhm', 'a4']
    posneg_plots = ['e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']
    double_plots = ['e', 'w', 'zeta', 'delta']
    if len(keys) == 0:
        keys = easy_plots + separable_pos_plots + posneg_plots + double_plots

    if (not figures) * (not axes):
        figures = {}
        axes = {}
        for key in keys:
            if (key in data) + (key + '1' in data) + (key == 'number'):
                #(keys + easy_plots + posneg_plots + double_plots):
                fig = plt.figure(figsize=(1.5 * BASE_SIZE, BASE_SIZE),
                                 dpi=300)
                ax = fig.add_subplot(111, aspect='equal')
                ax = focal_graph_axis(ax)
                figures.update({key: fig})
                axes.update({key: ax})

    if not scales:
        scales = {}

        for key in keys:
            if (key in data) + (key + '1' in data) + (key == 'number'):
                #scales.update({key : dict(vmin=None, vmax=None)})
                if defaults:
                    if key == 'e0':
                        scales.update({key: dict(vmin=0.1, vmax=0.3)})
                    elif key == 'e1':
                        scales.update({key: dict(vmin=-0.035, vmax=0.035)})
                    elif key == 'e2':
                        scales.update({key: dict(vmin=-0.035, vmax=0.035)})
                    elif key == 'zeta1':
                        scales.update({key: dict(vmin=-0.0025, vmax=0.0025)})
                    elif key == 'zeta2':
                        scales.update({key: dict(vmin=-0.0025, vmax=0.0025)})
                    elif key == 'delta1':
                        scales.update({key: dict(vmin=-0.008, vmax=0.008)})
                    elif key == 'delta2':
                        scales.update({key: dict(vmin=-0.008, vmax=0.008)})
                    elif key == 'a4':
                        scales.update({key: dict(vmin=0.01, vmax=0.07)})
                    else:
                        scales.update({key : {}})
                else:
                    scales.update({key : {}})

    if ('x_box' in data) * ('y_box' in data):
        x = data['x_box']
        y = data['y_box']
    else:
        x = data['x']
        y = data['y']

    # plots
    # get the counts
    counts, xedges, yedges = np.histogram2d(x, y, bins=edges)

    for key in keys:
        if (key in data) * (key in (easy_plots +
                                    separable_pos_plots +
                                    posneg_plots)):

            if key == 'e0prime':
                scales[key] = scales['e0'].copy()
            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=data[key], **scales[key])
            r = data[key]
            weighted_counts, xedges, yedges = np.histogram2d(
                x, y, weights=r, bins=edges)
            ## Image = key_axis.imshow(weighted_counts.T / counts.T,
            ##     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            ##     **scales[key])
            C = np.ma.masked_invalid(weighted_counts.T / counts.T)

        elif (key + '1' in data) * (key + '2' in data) * (key in double_plots):
            u = data[key + '1']
            v = data[key + '2']

            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            r = np.sqrt(u ** 2 + v ** 2)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=r, **scales[key])
            weighted_counts, xedges, yedges = np.histogram2d(
                x, y, weights=r, bins=edges)
            ## Image = key_axis.imshow(weighted_counts.T / counts.T,
            ##     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            ##     **scales[key])
            C = np.ma.masked_invalid(weighted_counts.T / counts.T)

        # get the ones not in double, easy, or posneg
        elif (key in data):

            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=data[key], **scales[key])
            r = data[key]
            weighted_counts, xedges, yedges = np.histogram2d(
                x, y, weights=r, bins=edges)
            ## Image = key_axis.imshow(weighted_counts.T / counts.T,
            ##     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            ##     **scales[key])
            C = np.ma.masked_invalid(weighted_counts.T / counts.T)

        elif (key == 'number'):
            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=data[key], **scales[key])
            ## Image = key_axis.imshow(counts.T,
            ##     extent=[xedges_counts[0], xedges_counts[-1],
            ##             yedges_counts[0], yedges_counts[-1]],
            ##     **scales[key])
            C = counts.T

        else:
            continue

        ## # move vmin and vmax around for deviations about mean
        ## if (((key in easy_plots) + (key in double_plots)) *
        ##     ('vmin' not in scales[key]) *
        ##     ('vmax' not in scales[key])):
        ##     center_value = np.mean(C)
        ##     scales[key].update({
        ##         'vmin': np.min(C) + center_value,
        ##         'vmax': np.max(C) + center_value})

        # set the cmap
        ## if ((key in separable_pos_plots) + (key in double_plots)):
        ##     cmap = blue_red
        ## elif np.all(C >= 0):
        ##     cmap = reds
        ## elif np.all(C <= 0):
        ##     cmap = blues_r
        ## else:
        ##     cmap = blue_red
        cmap = blue_red

        Image = key_axis.pcolor(xedges, yedges, C, cmap=cmap,
                                ## origin='lower',
                                ## interpolation='none',
                                **scales[key])
        CB = key_figure.colorbar(Image, ax=key_axis)
        scales[key].update(dict(vmin = CB.vmin, vmax = CB.vmax))

        figures.update({key: key_figure})
        axes.update({key: key_axis})

    return figures, axes, scales

def data_contour_plot(data, edges, scales=None,
                   figures=None, axes=None,
                   defaults=True,
                   keys=[]):
    """Takes data and makse a bunch of 2d histograms

    Parameters
    ----------
    data : dictionary of arrays
        Contains the relevant parameters which we are making these graphs

    edges : list or array
        Contains the x edges in edges[0] and y edges in edges[1]

    keys : list of keys
        Contains a list of the keys we want figures for. Can include OTHER
        objects OTHER than the ones normally made.

    Returns
    -------
    figures, axes : dictionaries of matplotlib objects
        The figures and axes made from this function.

    """

    easy_plots = ['flux', 'number']
    separable_pos_plots = ['e0', 'whisker', 'fwhm', 'a4']
    posneg_plots = ['e1', 'e2', 'delta1', 'delta2', 'zeta1', 'zeta2']
    double_plots = ['e', 'w', 'zeta', 'delta']
    if len(keys) == 0:
        keys = easy_plots + separable_pos_plots + posneg_plots + double_plots

    if (not figures) * (not axes):
        figures = {}
        axes = {}
        for key in keys:
            if (key in data) + (key + '1' in data) + (key == 'number'):
                #(keys + easy_plots + posneg_plots + double_plots):
                fig = plt.figure(figsize=(BASE_SIZE * 1.5, BASE_SIZE),
                                 dpi=300)
                ax = fig.gca(projection='3d')
                ax = focal_graph_axis(ax)

                figures.update({key: fig})
                axes.update({key: ax})

    if not scales:
        scales = {}

        for key in keys:
            if (key in data) + (key + '1' in data) + (key == 'number'):
                #scales.update({key : dict(vmin=None, vmax=None)})
                if defaults:
                    if key == 'e0':
                        scales.update({key: dict(vmin=0.1, vmax=0.3)})
                    elif key == 'e1':
                        scales.update({key: dict(vmin=-0.035, vmax=0.035)})
                    elif key == 'e2':
                        scales.update({key: dict(vmin=-0.035, vmax=0.035)})
                    elif key == 'zeta1':
                        scales.update({key: dict(vmin=-0.0025, vmax=0.0025)})
                    elif key == 'zeta2':
                        scales.update({key: dict(vmin=-0.0025, vmax=0.0025)})
                    elif key == 'delta1':
                        scales.update({key: dict(vmin=-0.008, vmax=0.008)})
                    elif key == 'delta2':
                        scales.update({key: dict(vmin=-0.008, vmax=0.008)})
                    elif key == 'a4':
                        scales.update({key: dict(vmin=0.01, vmax=0.07)})
                    else:
                        scales.update({key : {}})
                else:
                    scales.update({key : {}})

    if ('x_box' in data) * ('y_box' in data):
        x = data['x_box']
        y = data['y_box']
    else:
        x = data['x']
        y = data['y']

    # plots
    # get the counts
    counts, xedges, yedges = np.histogram2d(x, y, bins=edges)

    for key in keys:
        if (key in data) * (key in (easy_plots +
                                    separable_pos_plots +
                                    posneg_plots)):

            if key == 'e0prime':
                scales[key] = scales['e0'].copy()
            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=data[key], **scales[key])
            r = data[key]
            weighted_counts, xedges, yedges = np.histogram2d(
                x, y, weights=r, bins=edges)
            ## Image = key_axis.imshow(weighted_counts.T / counts.T,
            ##     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            ##     **scales[key])
            C2 = np.ma.masked_invalid(weighted_counts.T / counts.T)
            C = weighted_counts.T / counts.T

        elif (key + '1' in data) * (key + '2' in data) * (key in double_plots):
            u = data[key + '1']
            v = data[key + '2']

            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            r = np.sqrt(u ** 2 + v ** 2)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=r, **scales[key])
            weighted_counts, xedges, yedges = np.histogram2d(
                x, y, weights=r, bins=edges)
            ## Image = key_axis.imshow(weighted_counts.T / counts.T,
            ##     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            ##     **scales[key])
            C2 = np.ma.masked_invalid(weighted_counts.T / counts.T)
            C = weighted_counts.T / counts.T

        # get the ones not in double, easy, or posneg
        elif (key in data):

            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=data[key], **scales[key])
            r = data[key]
            weighted_counts, xedges, yedges = np.histogram2d(
                x, y, weights=r, bins=edges)
            ## Image = key_axis.imshow(weighted_counts.T / counts.T,
            ##     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            ##     **scales[key])
            C2 = np.ma.masked_invalid(weighted_counts.T / counts.T)
            C = weighted_counts.T / counts.T

        elif (key == 'number'):
            key_figure = figures[key]
            key_axis = axes[key]
            key_axis.set_title(key)
            ## (counts, xedges, yedges, Image) = key_axis.hist2d(
            ##     x, y, bins=edges, weights=data[key], **scales[key])
            ## Image = key_axis.imshow(counts.T,
            ##     extent=[xedges_counts[0], xedges_counts[-1],
            ##             yedges_counts[0], yedges_counts[-1]],
            ##     **scales[key])
            C = counts.T
            C2 = C

        else:
            continue

        if not ('vmin' in scales[key]):
            scales[key].update({'vmin': np.nanmin(C),
                                'vmax': np.nanmax(C)})
        elif not scales[key]['vmin']:
            # 'vmin' in scales[key]
            scales[key].update({'vmin': np.nanmin(C),
                                'vmax': np.nanmax(C)})

        C = np.where(counts.T == 0, scales[key]['vmin'], C)
        #C = np.where(C < Cmin, Cmin, C)

        cmap = blue_red
        X, Y = np.meshgrid(0.5 * (xedges[:-1] + xedges[1:]),
                           0.5 * (yedges[:-1] + yedges[1:]))

        Image = key_axis.plot_surface(X, Y, C, rstride=1, cstride=1,
                                      alpha=0.5, cmap=blue_red,
                                      linewidth=0.25, antialiased=True,
                                      **scales[key])
        Image2 = key_axis.pcolor(xedges, yedges, C2,
                                 cmap=blue_red, **scales[key])

        # put in the same plane as the bottom of the mesh
        cset = key_axis.add_collection3d(Image2,
                                         zs=scales[key]['vmin'], zdir='z')

        CB = key_figure.colorbar(Image2, ax=key_axis)
        scales[key].update(dict(vmin = CB.vmin, vmax = CB.vmax))

        key_axis.set_zlim(scales[key]['vmin'], scales[key]['vmax'])
        key_axis.view_init(elev=30, azim=-10)



        figures.update({key: key_figure})
        axes.update({key: key_axis})

    return figures, axes, scales


def plot_star(recdata, figure=None, axis=None, nPixels=32,
              weighted=False, process=True):
    """Intelligently plot a star

    Parameters
    ----------
    recdata : recarray
        Contains a parameter 'STAMP' that needs to be resized, as well as
        X2WIN_IMAGE, XYWIN_IMAGE, Y2WIN_IMAGE (moment parameters) that then
        will be converted into the adaptive moment matrix.

    figure : matplotlib figure, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    axis : matplotlib axis, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    nPixels : integer
        Size of the stamp array dimensions

    weighted : bool
        If true, apply weighting exp(-rho2 / 2) as defined in Hirata et al 2004

    Returns
    -------
    figure : matplotlib figure, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    axis : matplotlib axis, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    """

    # make the figure
    if (not figure) * (not axis):
        figure = plt.figure(figsize=(BASE_SIZE, BASE_SIZE), dpi=300)
        axis = figure.add_subplot(111, aspect='equal')

    if process:
        stamp = process_image(recdata, weighted=weighted, nPixels=nPixels)

        stamp = np.where(stamp == 0, np.nan, stamp)
    else:
        stamp = recdata['STAMP']
        stamp = stamp.reshape(nPixels, nPixels)

    im = axis.imshow(stamp)
    plt.colorbar(im)

    return figure, axis

def process_image(recdata, weighted=True, nPixels=32):
    """
    Parameters
    ----------
    recdata : recarray
        Contains a parameter 'STAMP' that needs to be resized, as well as
        X2WIN_IMAGE, XYWIN_IMAGE, Y2WIN_IMAGE (moment parameters) that then
        will be converted into the adaptive moment matrix.

    nPixels : integer
        Size of the stamp array dimensions

    weighted : bool
        If true, apply weighting exp(-rho2 / 2) as defined in Hirata et al 2004

    Returns
    -------
    stamp : array
        The manipulated stamp.

    """
    d = recdata
    stamp = d['STAMP'].copy()

    # filter via background and threshold
    stamp -= d['BACKGROUND']
    stamp = np.where(stamp > d['THRESHOLD'], stamp, 0)

    stamp = stamp.reshape(nPixels, nPixels)

    # do window
    max_nsig2 = 25
    y, x = np.indices(stamp.shape)
    Mx = nPixels / 2 + d['XWIN_IMAGE'] % int(d['XWIN_IMAGE']) - 1
    My = nPixels / 2 + d['YWIN_IMAGE'] % int(d['YWIN_IMAGE']) - 1

    r2 = (x - Mx) ** 2 + (y - My) ** 2

    Mxx = 2 * d['X2WIN_IMAGE']
    Myy = 2 * d['Y2WIN_IMAGE']
    Mxy = 2 * d['XYWIN_IMAGE']
    detM = Mxx * Myy - Mxy * Mxy
    Minv_xx = Myy / detM
    TwoMinv_xy = -Mxy / detM * 2.0
    Minv_yy = Mxx / detM
    Inv2Minv_xx = 0.5 / Minv_xx
    rho2 = Minv_xx * (x - Mx) ** 2 + TwoMinv_xy * (x - Mx) * (y - My) + Minv_yy * (y - My) ** 2
    stamp = np.where(rho2 < max_nsig2, stamp, 0)

    if weighted:
        stamp *= np.exp(-0.5 * rho2)

    return stamp



def save_func(steps,
              state_history, chisquared_history,
              chi_weights,
              plane, reference_plane,
              output_directory,
              edges,
              boxdiv=1):
    """
    Parameters
    ----------

    steps : int
        Number of steps

    state_history : dictionary of lists
        List the values

    Notes
    -----
    When using this with minuit_fit, create another encapsulating function
    that only depends on the steps parameter.
    """

    # compare
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex='all', sharey='all',
                            figsize=(BASE_SIZE * 4, BASE_SIZE * 2))
    figures = {'e': fig, 'w': fig, 'zeta': fig, 'delta': fig}
    focal_keys = figures.keys()
    for ij in range(axs.shape[0]):
        for jk in range(axs.shape[1]):
            axs[ij][jk] = focal_graph_axis(axs[ij][jk])

    axes = {'e': axs[0,0], 'w': axs[0,1], 'delta': axs[1,0], 'zeta': axs[1,1]}
    figures, axes, scales = data_focal_plot(reference_plane,
                                            color='r', boxdiv=boxdiv,
                                            figures=figures, axes=axes,
                                            keys=focal_keys,
                                            )
    # plot the comparison
    figures, axes, scales = data_focal_plot(plane,
                                            color='b', boxdiv=boxdiv,
                                            figures=figures, axes=axes,
                                            scales=scales,
                                            keys=focal_keys,
                                            )

    plane.update({
                  'e0subs': (reference_plane['e0'] - plane['e0'])
                            / reference_plane['e0'],
                  'e0prime': plane['e0'],
                  'e0': reference_plane['e0']
                  })

    figures_hist = {'e0subs': fig, 'e0': fig, 'e0prime': fig}
    axes_hist = {'e0subs': axs[1,3], 'e0': axs[0,2], 'e0prime': axs[1,2]}
    figures_hist, axes_hist, scales_hist = data_hist_plot(
            plane, edges,
            figures=figures_hist,
            axes=axes_hist,
            keys=['e0subs', 'e0', 'e0prime'])

    # make tables for param and delta and chi2
    colLabels = ("Parameter", "Value", "Delta")
    cellText = [[key, '{0:.3e}'.format(state_history[-1][key]),
                 '{0:.3e}'.format(state_history[-1][key]
                                  - state_history[-2][key])]
                 for key in state_history[-1].keys()]
    # add in chi2s
    cellText += [[key,
                  '{0:.3e}'.format(np.sum(chisquared_history[-1][key])),
                  '{0:.3e}'.format(np.sum(chisquared_history[-1][key]
                                          - chisquared_history[-2][key]))]
                 for key in chisquared_history[-1].keys()]

    chi2delta = chisquared_history[-1]['chi2'] - chisquared_history[-2]['chi2']
    cellText += [['total chi2',
                  '{0:.3e}'.format(chisquared_history[-1]['chi2']),
                  '{0:.3e}'.format(chi2delta)]]
    axs[0,3].axis('off')
    table = axs[0,3].table(cellText=cellText, colLabels=colLabels,
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1, 2)


    plt.tight_layout()
    fig.savefig(output_directory + '{0:04d}.pdf'.format(steps))
    plt.close('all')
    del fig, axs, figures, axes, figures_hist, axes_hist

    # add condition to do histograms
    if steps % 10 == 0:
        # do reference to get scales
        figures, axes, scales = data_hist_plot(reference_plane, edges)
        ## figures.savefig(output_directory +
        ##    '{0:04d}_reference_histograms.pdf'.format(steps))
        # do current
        figures, axes, scales = data_hist_plot(plane, edges, scales=scales)
        for fig_key in figures.keys():
            figures[fig_key].savefig(output_directory +
                '{0:04d}_histogram_{1}.pdf'.format(steps, fig_key))

        plt.close('all')
        del figures, axes, scales

