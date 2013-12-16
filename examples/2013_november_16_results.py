#!/usr/bin/env python
"""
File: 2013_november_16_results.py
Author: Chris Davis
Description: Go into collect_results.py for this specific run. Run second.
"""

from __future__ import print_function, division
from subprocess import call
from os import path, listdir, makedirs
import fnmatch
from focal_plane_routines import print_command

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/november_16/"

if not path.exists(output_directory + 'results/'):
    makedirs(output_directory + 'results/')

# find the ones done by the fitter
results = listdir(output_directory)
results = fnmatch.filter(results, '00*')
numbers = []
for result in results:
    numbers.append(int(result[:8]))
numbers = list(set(numbers))

input_directories = [output_directory] * len(numbers)

command = ['python', 'collect_results.py',
           '-i', str(input_directories),
           '-e', str(numbers),  # expid
           '-o', output_directory + 'results/',  # output_directory
           ]
#print_command(command)
call(command)
