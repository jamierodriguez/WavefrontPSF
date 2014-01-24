#!/usr/bin/env python
"""
File: 2014_january_06_1_fits.py
Author: Chris Davis
Description: Submit these images to the catalog to do the fits. Run first
(after getting the images.)
"""

from __future__ import print_function, division
import numpy as np
from subprocess import call
from os import path, makedirs, listdir
import fnmatch
from focal_plane_routines import print_command



expids = np.arange(231046, 231053)
expids = np.append(expids, range(231089, 231096))
expids = np.append(expids, range(232608, 232849))
expids = np.append(expids, range(233377, 233571))
expids = np.append(expids, range(233584, 233642))


# pop the ones that don't exist
entries = listdir('/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/')
# pop logs
if 'logs' in entries: entries.remove('logs')
entries = [int(entry) for entry in entries]
expids = [item for item in expids if item in entries]

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/january_06/"

if not path.exists(output_directory):
    makedirs(output_directory)
    makedirs(output_directory + 'logs')

# now go to output and filter by what is already present
results = listdir(output_directory)
results = fnmatch.filter(results, '00*')
results_expids = []
for result in results:
    results_expids.append(int(result[:8]))
expids = [item for item in expids if item not in results_expids]

for image_number in expids:
    command = ['bsub',
               '-q', 'xlong',
               '-o', output_directory +
                     'logs/{0:08d}.log'.format(image_number),
               '-R', 'rhel60&&linux64',
               'python', 'batch_fit.py',
               '-c', '/nfs/slac/g/ki/ki18/cpd/focus/image_data.csv',  # csv
               '-t', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',  # catalogs
               '-m', '/u/ec/roodman/Astrophysics/Donuts/Meshes/',  # path_mesh
               '-n', "Science20120915s1v3_134239",  # mesh_name
               '-o', output_directory,  # output_directory
               '-s', str(20),  # max_samples
               '-b', str(1),  # boxdiv
               '-a', str(0),  # subav
               '-d', str(0),  # seed
               '-f', 'default',  # filtering for conds
               '-cpd', str(1),  # use cpd or sextractor
               '-e', '{0}'.format(image_number),  # expid
               ]
    print_command(command)
    call(command)

