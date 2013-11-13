#!/usr/bin/env python
# batch_plot.py
from __future__ import print_function, division
import argparse
import numpy as np
from matplotlib import pyplot as plt
from plot_wavefront import focal_plane_plot, collect_images
from os import makedirs, path, remove

# argparse
"""
Include the locations of the moments (both fitted and comparison)

This file will take the results and plot them as well as collate everything
together into csv files. Those can also be plotted?
"""

# take as give that we have list_fitted_plane list_comparison_plane,
# list_minuit_results


