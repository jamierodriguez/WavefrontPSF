#!/usr/bin/env python
"""
File: 2014_january_06_make_catalogs.py
Author: Chris Davis
Description: Routine to submit batch jobs for downloading the data.

nightsums used here.

http://decam03.fnal.gov:8080/nightsum/nightsum-2013-09-05/nightsum.html

http://decam03.fnal.gov:8080/nightsum/nightsum-2013-09-10/nightsum.html

http://decam03.fnal.gov:8080/nightsum/nightsum-2013-09-12/nightsum.html

"""

from __future__ import print_function, division
from subprocess import call
from os import path, makedirs
import numpy as np
from focal_plane_routines import print_command

input_list = [
    [20130906105326, 20130905, 231046, 231053],
    [20130906105326, 20130905, 231089, 231096],
    [20130911103044, 20130910, 232608, 232849],
    [20130913151017, 20130912, 233377, 233571],
    [20130913151017, 20130912, 233584, 233642],
    ]
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

for i in range(len(expids)):

    expid = expids[i]
    rid = rids[i]
    date = dates[i]

    command = [
        'bsub',
        '-q', 'medium',
        '-o', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript' +
              '/logs/{0:08d}_{1:08d}_cpd.log'.format(minImage, maxImage),
        '-R', 'rhel60&&linux64',
        'python', 'catalog_maker.py',
        '-e', '{0}'.format(expid),
        '-c', '/nfs/slac/g/ki/ki18/cpd/focus/image_data.csv',  # csv
        '-m', '/u/ec/roodman/Astrophysics/Donuts/Meshes/',  # path_mesh
        '-n', "Science20120915s1v3_134239",  # mesh_name
        '-o', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/' +
              '{0:08d}/'.format(expid),
        '-t', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',  # catalog
        '-s', '16',  # stamp size
        '-f', 'all',  # filter conditions
        '-rid', '{0}'.format(rid),
        '-d', '{0}'.format(date),]
    print_command(command)
    call(command)

