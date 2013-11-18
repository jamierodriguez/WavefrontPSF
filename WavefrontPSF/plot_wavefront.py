#!/usr/bin/env python
# plot_wavefront.py
from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from os import makedirs, path
from subprocess import call

'''
File: plot_wavefront.py
Author: Chris Davis
Description: Methods for plotting various wavefront configurations.
'''


def wedge_collection(X, Y, U, V,
                     U_var, V_var,
                     scale=1,
                     artpatch=0,
                     patch_dict={'alpha': 0.4}):

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
            theta_l = theta[i] - sigma_theta[i]
            theta_u = theta_l + 2 * sigma_theta[i]

            # we want the slice, not the rest of the annulus.
            if theta_u < theta_l:
                theta_l, theta_u = theta_u, theta_l
            #TODO: should this be / scale or * scale?
            RU = (R[i] + sigma_r[i]) / scale
            wid2 = sigma_r[i] / scale
            if RU - wid2 < 0:
                wid2 = None
            try:
                art = Wedge([X[i], Y[i]], RU,
                            theta_l, theta_u, width=wid2)
            except ValueError:
                #import ipdb; ipdb.set_trace()
                print('error in wedge:', X[i], Y[i], RU, theta_l, theta_u,
                      wid2)
                # occasionally python craps out? I'm not sure what's going
                # on but ignoring the error for now has worked
                #pass
            patches.append(art)

            theta[i] += delta_theta

    collection = PatchCollection(patches, **patch_dict)

    return collection


def focal_plane_plot(x, y,
                     u, v,
                     u_ave, v_ave,
                     u_var=[], v_var=[],
                     focal_figure=None, focal_axis=None,
                     quiverkey_dict={},
                     color='k',
                     scale=10 / 0.04, artpatch=2,
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

        The default scaling is 10 mm per 0.04 arcsec or 250 mm / arcsec . This
        is probably too large for the poorer nights.

    whisker_width : float, optional
        Sets the thickness of the whisker (in mm).

        Default is 2.5

    quiverky_dict : dictionary, optional
        Common dictionary commands for the quiverkey (color, text, value)

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

    # the factor of 2 is because of my method for plotting whiskers goes from
    # the tail and then adds another one on the other side 180 degrees away
    quiverkey_dict_use = {'value': 2 * 10 / scale,
                          'title': r'${0:.2e}$ arcsec'.format(10 / scale),
                          'color': color}
    quiverkey_dict_use.update(quiverkey_dict)

    # make the figure
    if not focal_figure:
        focal_figure = plt.figure(figsize=(16, 12), dpi=300)
        focal_axis = focal_figure.add_subplot(111,
                                              aspect='equal')
        focal_axis.set_xlabel('$X$ [mm] (East)')
        focal_axis.set_ylabel('$Y$ [mm] (South)')
        focal_axis.set_xlim(-250, 250)
        focal_axis.set_ylim(-250, 250)

    # add quiver
    quiver_dict = dict(alpha=0.75,
                       angles='uv',
                       color=color,
                       headlength=0,
                       headwidth=1,
                       pivot='tail',
                       scale=1. / scale,
                       scale_units='xy',
                       units='xy',
                       width=whisker_width)

    if artpatch == 1:
        quiver_dict['headlength'] = 1
        quiver_dict['headwidth'] = 1

    for i in range(artpatch):

        Q = focal_axis.quiver(x + offset_x, y, u, v,
                              **quiver_dict)

        # add cornervalue in upper right
        focal_axis.quiver(200 + offset_x, 200,
                          u_ave, v_ave,
                          **quiver_dict)
        # add quiverkey in lower left
        focal_axis.quiverkey(Q, 0.1, 0.1,
                             quiverkey_dict_use['value'],
                             quiverkey_dict_use['title'],
                             color=quiverkey_dict_use['color'])

        # update angles
        delta_theta = 2 * np.pi / artpatch
        u, v = u * np.cos(delta_theta) + v * np.sin(delta_theta), \
            u * np.sin(delta_theta) + v * np.cos(delta_theta)

        u_ave, v_ave = \
            u_ave * np.cos(delta_theta) + v_ave * np.sin(delta_theta), \
            u_ave * np.sin(delta_theta) + v_ave * np.cos(delta_theta)


    if len(u_var) > 0:
        # add wedges
        wedges = wedge_collection(x + offset_x, y, u, v,
                                  u_var, v_var,
                                  scale=quiver_dict['scale'],
                                  artpatch=artpatch)
        focal_axis.add_collection(wedges)

    return focal_figure, focal_axis

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
                       'gs', '-sDEVICE=pdfwrite',
                       '-dNOPAUSE', '-dBATCH', '-dSAFER',
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
