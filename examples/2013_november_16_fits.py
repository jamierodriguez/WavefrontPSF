#!/usr/bin/env python
"""
File: 2013_november_16_fits.py
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



numbers = np.arange(231046, 231053)
numbers = np.append(numbers, range(231089, 231096))
numbers = np.append(numbers, range(232608, 232849))
numbers = np.append(numbers, range(233377, 233642))

# pop the ones that don't exist
entries = listdir('/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/')
# pop logs
if 'logs' in entries: entries.remove('logs')
entries = [int(entry) for entry in entries]
numbers = [item for item in numbers if item in entries]

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/november_16/"

if not path.exists(output_directory):
    makedirs(output_directory)
    makedirs(output_directory + 'logs')

# now go to output and filter by what is already present
results = listdir(output_directory)
results = fnmatch.filter(results, '00*')
results_numbers = []
for result in results:
    results_numbers.append(int(result[:8]))
numbers = [item for item in numbers if item not in results_numbers]

for image_number in numbers:
    command = ['bsub',
               '-q', 'long',
               '-o', output_directory +
                     'logs/{0:08d}.log'.format(image_number),
               '-R', 'rhel60&&linux64',
               'python', 'batch_fit.py',
               '-c', '/nfs/slac/g/ki/ki18/cpd/focus/september_27/image_data.csv',  # csv
               '-t', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',  # catalogs
               '-m', '/u/ec/roodman/Astrophysics/Donuts/Meshes/',  # path_mesh
               '-n', "Science20120915s1v3_134239",  # mesh_name
               '-e', '{0}'.format(image_number),  # expid
               '-o', output_directory,  # output_directory
               '-r', str(0),  # random
               '-s', str(750),  # max_samples
               '-b', str(1),  # boxdiv
               '-a', str(0),  # subav
               ]
    #print_command(command)
    call(command)

