#!/usr/bin/env python
"""
File: wavefront.py
Author: Chris Davis
Description: Module for generating PSF objects and their moments.

TODO: Add whisker plot capabilities.
TODO: Add a copy capability
"""

from __future__ import print_function, division
from decamutil import decaminfo
from os import path, makedirs
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class Wavefront(object):
    """Class with the ability to generate PSF image and statistics. Given a
    list of positions, make a list of PSF images or statistics.

    Attributes
    ----------
    Wavefront_State
        Dictionary of parameters describing the state of the wavefront.

    PSF Interpolator
        Method for generating PSF image at a given location.

    PSF Evaluator
        Object that can take output from Interpolator and give PSF statistics


    Data
        Every datapoint. Assumed to be a pandas dataframe

    Field
        Datapoints reduced to field

    Methods
    -------
    save
        Save the wavefront.

    draw_psf
        Given coordinates, draw psf image

    get_psf_stats
        Gets PSF statistics, perhaps by evaluating the PSF

    plot_field
        Makes a focal plane plot

    """

    def __init__(self, model, PSF_Interpolator, PSF_Evaluator,
                 **args):

        self.PSF_Interpolator = PSF_Interpolator
        self.PSF_Evaluator = PSF_Evaluator
        self.PSF_Evaluator_keys = self.PSF_Evaluator.keys

        # this is useful
        self.decaminfo = decaminfo()

        if model is not None:
            self.data = model
            self.field, self.bins_x, self.bins_y = self.reduce_data_to_field(self.data, np.median, 1)

    def __getitem__(self, key):
        return self.field[key]

    def convert_lists_to_dataframe(self, values, value_names):
        # ensure we have same number of columns as names
        assert len(values) == len(value_names)
        for i in xrange(len(values) - 1):
            # make sure we have the same number of entries
            assert len(values[i]) == len(values[i + 1])
        # make data frame
        df = {}
        for i in xrange(len(values)):
            df[value_names[i]] = values[i]
        df = pd.DataFrame(df)
        return df

    def save(self, out_path):
        """Take the data and save it!

        Parameters
        ----------
        out_path : string
            The location where we will dump the pickle.

        """

        if not path.exists(path.dirname(out_path)):
            makedirs(path.dirname(out_path))
        with open(out_path, 'wb') as out_file:
            pickle.dump(self, out_file)

    def edges(self, boxdiv):
        """Convenience wrapper for decaminfo.getEdges
        """

        edges = self.decaminfo.getEdges(boxdiv)
        return edges

    def reduce_data_to_field(self, data, xkey='x', ykey='y',
            reducer=np.median, num_bins=1):
        """Take data and bin by focal plane coordinates.

        Parameters
        ----------
        data : dataframe
            Contains all the datapoints.

        reducer : function
            Function that takes set of data and returns a number.

        num_bins : int, default 1
            Number of bins for the focal plane. If less than six, then the
            number of bins is a sort of proxy for the number of divisions of a
            chip. Default is 2 bins per chip.

        Returns
        -------
        field : dataframe
            Dataframe binned by x and y coordinates.

        bins_x, bins_y : arrays
            Arrays of the bins used.

        Notes
        -----
        data needs to have 'x' and 'y' keys!

        """
        x = data[xkey]
        y = data[ykey]

        if num_bins < 6:
            bins_x, bins_y = self.edges(num_bins)
        else:
            bins_x = np.linspace(np.min(x), np.max(x), num_bins)
            bins_y = np.linspace(np.min(y), np.max(y), num_bins)
        groups = data.groupby([pd.cut(x, bins_x), pd.cut(y, bins_y)])
        field = groups.aggregate(reducer)
        # also get the count
        counts = groups[xkey].aggregate('count').values

        # filter out nanmins on x and y
        field = field[field[xkey].notnull() & field[xkey].notnull()]
        # counts already filtered out notnull so let's try!
        field['N'] = counts

        return field, bins_x, bins_y

    def reduce(self, xkey='x', ykey='y', reducer=np.median, num_bins=1):
        # convenience function to do reduce_data_to_field for data attribute
        self.field, self.bins_x, self.bins_y = self.reduce_data_to_field(
                self.data, xkey=xkey, ykey=ykey, reducer=reducer,
                num_bins=num_bins)

    def draw_psf(self, data, **kwargs):
        # note: PSF_Interpolator.x_keys need to be in data
        return self.PSF_Interpolator(data, **kwargs)

    def evaluate_psf(self, data, **kwargs):
        # depending on method, you could expect something like this:
        evaluated_psfs = self.PSF_Evaluator(data, **kwargs)
        # combine the results from PSF_Evaluator with your input data
        combined_df = evaluated_psfs.combine_first(data)
        return combined_df

    def plot_colormap(self, data, xkey, ykey, zkey, num_bins=20, fig=None, ax=None, reducer=np.median):
        field, bins_x, bins_y = self.reduce_data_to_field(
                data, xkey=xkey, ykey=ykey, num_bins=num_bins,
                reducer=reducer)
        if type(fig) == type(None):
            fig, ax = plt.subplots(figsize=(10,5))
        fig, ax = self.plot_field(zkey, field=field,
                bins_x=bins_x, bins_y=bins_y, fig=fig, ax=ax)

        return fig, ax

    def plot_field(self, key, field='None', bins_x=None, bins_y=None, fig=None, ax=None):
        """Make a plot of the field.

        Parameters
        ----------
        key : string
            What value are we trying to plot?

        fig, ax : matplotlib objects, optional
            If given, use these plots. Else, make our own!

        Returns
        -------

        fig, ax : matplotlib objects
            The figure and axis of our plot!

        """
        if type(field) == type('None'):
            if field == 'None':
                field = self.field
                bins_x = self.bins_x
                bins_y = self.bins_y

        indx_x = field.index.labels[0].values()
        indx_y = field.index.labels[1].values()
        # here is something that is going to be irritating and cludgey:
        # let's get the values of the different bins (for sorting purposes)
        x_vals = np.array([np.mean(eval(ith.replace('(','['))) for
                           ith in field.index.levels[0]])
        y_vals = np.array([np.mean(eval(ith.replace('(','['))) for
                           ith in field.index.levels[1]])
        # now sort the order for the levels
        x_vals_argsorted = np.argsort(x_vals)
        y_vals_argsorted = np.argsort(y_vals)
        # now this means that the 0th entry in x_vals_argsorted comes first
        # so we want indx_x_transform to represent the sorted values
        # so instead of indx_x representing arbitrary bin i, we want it to
        # instead represent sorted bin j
        indx_x_transform = np.argsort(np.arange(len(x_vals))[np.argsort(x_vals)])[indx_x]
        indx_y_transform = np.argsort(np.arange(len(y_vals))[np.argsort(y_vals)])[indx_y]

        if ax == None:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.set_xlabel('$X$ [mm] (East)')
            ax.set_ylabel('$Y$ [mm] (South)')
            ax.set_xlim(-250, 250)
            ax.set_ylim(-250, 250)

        # figure out shifting the colormap
        b = np.max(field[key][field[key].notnull()])
        a = np.min(field[key][field[key].notnull()])
        c = 0
        midpoint = (c - a) / (b - a)
        if midpoint <= 0 or midpoint >= 1:
            cmap = plt.cm.Reds
        else:
            cmap = shiftedColorMap(plt.cm.RdBu_r, midpoint=midpoint)
        vmin = a
        vmax = b

        C = np.ma.zeros((indx_x.max() + 1, indx_y.max() + 1))
        C.mask = np.ones((indx_x.max() + 1, indx_y.max() + 1))
        np.add.at(C, [indx_x_transform, indx_y_transform],
                  field[key].values)
        np.multiply.at(C.mask, [indx_x_transform, indx_y_transform], 0)
        # bloops
        C = C.T

        IM = ax.pcolor(bins_x, bins_y, C,
                       cmap=cmap, vmin=vmin, vmax=vmax)
        CB = fig.colorbar(IM, ax=ax)

        return fig, ax

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
Taken from

https://github.com/olgabot/prettyplotlib/blob/master/prettyplotlib/colors.py

which makes beautiful plots by the way


    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


