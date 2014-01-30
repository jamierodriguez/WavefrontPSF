#!/usr/bin/env python
"""
File: plot_wavefront.py
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
from focal_plane_routines import ellipticity_to_whisker


def focal_graph():
    """Convenience for the creation of my focal plane graphics

    Returns
    -------
    focal_figure : figure object

    focal_axis : axis object

    """
    focal_figure = plt.figure(figsize=(12, 12), dpi=300)
    focal_axis = focal_figure.add_subplot(111,
                                          aspect='equal')
    focal_axis.set_xlabel('$X$ [mm] (East)')
    focal_axis.set_ylabel('$Y$ [mm] (South)')
    focal_axis.set_xlim(-250, 250)
    focal_axis.set_ylim(-250, 250)
    return focal_figure, focal_axis

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
                    boxdiv=1,
                    average=np.mean,
                    scales=None,
                    figures=None,
                    axes=None):

    """Takes data and makes a bunch of wavefront plots

    Parameters
    ----------
    data : dictionary of arrays
        Contains the relevant parameters which we are making these graphs

    color : string
        Color of the focal plots we make

    boxdiv : int
        Divisions of chips

    average : function
        Function to be used for determining the average

    scales : dictionary of floats
        The values we use to make our plots



    Returns
    -------
    figures, axes : dictionaries of matplotlib objects
        The figures and axes made from this function, or updated if already
        supplied.

    scales : dictionary of floats
        The values of scales used.

    """


    if not figures:
        figures = dict(e=None, w=None, zeta=None, delta=None, wd=None)
    if not axes:
        axes = dict(e=None, w=None, zeta=None, delta=None, wd=None)
    if not scales:
        scales = dict(e=-1, w=-1, zeta=-1, delta=-1, wd=-1)

    if ('x_box' in data) * ('y_box' in data):
        x = data['x_box']
        y = data['y_box']
    else:
        x = data['x']
        y = data['y']

    # plots
    if ('e1' in data) * ('e2' in data):
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
        ('w1' in data) * ('w2' in data)):
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

    if ('zeta1' in data) * ('zeta2' in data):
        u = data['zeta1']
        v = data['zeta2']
        if ('var_zeta1' in data) * ('var_zeta2' in data):
            u_var = data['var_zeta1']
            v_var = data['var_zeta2']
        else:
            u_var = []
            v_var = []

        zeta_focal_figure, zeta_axis, zeta_scale = focal_plane_plot(
            x, y, u, v, artpatch=1,
            u_var=u_var, v_var=v_var,
            color=color, scale=scales['zeta'],
            focal_figure=figures['zeta'], focal_axis=axes['zeta'])
        zeta_axis.set_title('zeta')
        figures.update(dict(zeta=zeta_focal_figure))
        axes.update(dict(zeta=zeta_axis))
        scales.update(dict(zeta=zeta_scale))

    if ('delta1' in data) * ('delta2' in data):
        u = data['delta1']
        v = data['delta2']
        if ('var_delta1' in data) * ('var_delta2' in data):
            u_var = data['var_delta1']
            v_var = data['var_delta2']
        else:
            u_var = []
            v_var = []

        delta_focal_figure, delta_axis, delta_scale = focal_plane_plot(
            x, y, u, v, artpatch=1,
            u_var=u_var, v_var=v_var,
            color=color, scale=scales['delta'],
            focal_figure=figures['delta'], focal_axis=axes['delta'])
        delta_axis.set_title('delta')
        figures.update(dict(delta=delta_focal_figure))
        axes.update(dict(delta=delta_axis))
        scales.update(dict(delta=delta_scale))

    if ('wd1' in data) * ('wd2' in data):
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

def data_hist_plot(data, edges, scales=None):
    """Takes data and makse a bunch of 2d histograms

    Parameters
    ----------
    data : dictionary of arrays
        Contains the relevant parameters which we are making these graphs

    edges : list or array
        Contains the x edges in edges[0] and y edges in edges[1]

    Returns
    -------
    figures, axes : dictionaries of matplotlib objects
        The figures and axes made from this function.

    """

    easy_plots = ['e0', 'e0prime', 'xi', 'a4', 'whisker', 'flux', 'fwhm']
    double_plots = ['e', 'w', 'zeta', 'delta']
    if not scales:
        scales = {}
        for key in easy_plots:
            scales.update({key : dict(cmin=1e-9, vmin=None, vmax=None)})
        for key in double_plots:
            scales.update({key : dict(cmin=1e-9, vmin=None, vmax=None)})



    if ('x_box' in data) * ('y_box' in data):
        x = data['x_box']
        y = data['y_box']
    else:
        x = data['x']
        y = data['y']

    figures = {}
    axes = {}

    # plots

    for key in easy_plots:
        if key in data:

            key_figure, key_axis = focal_graph()
            key_axis.set_title(key)
            (counts, xedges, yedges, Image) = key_axis.hist2d(
                x, y, bins=edges, weights=data[key], **scales[key])
            CB = key_figure.colorbar(Image)
            scales[key].update(dict(vmin = CB.vmin, vmax = CB.vmax))

            figures.update({key: key_figure})
            axes.update({key: key_axis})

    for key in double_plots:
        if (key + '1' in data) * (key + '2' in data):
            u = data[key + '1']
            v = data[key + '2']

            key_figure, key_axis = focal_graph()
            key_axis.set_title(key)
            r = np.sqrt(u ** 2 + v ** 2)
            (counts, xedges, yedges, Image) = key_axis.hist2d(
                x, y, bins=edges, weights=r, **scales[key])
            CB = key_figure.colorbar(Image)
            scales[key].update(dict(vmin = CB.vmin, vmax = CB.vmax))

            figures.update({key: key_figure})
            axes.update({key: key_axis})

    return figures, axes, scales

