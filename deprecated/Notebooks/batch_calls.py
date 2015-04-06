# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# instead of running python files for the batch calls, do them here...
# 
# # TODO:
# 
# - update these for current code

# <markdowncell>

# # make the catalogs

# <codecell>

from __future__ import print_function, division
from subprocess import call
from os import path, makedirs
import numpy as np
from routines import print_command

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
        '-q', 'long',
        '-o', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript' +
              '/logs/{0:08d}_cpd.log'.format(expid),
        '-R', 'rhel60&&linux64',
        'python', 'catalog_maker.py',
        '-m', '/u/ec/roodman/Astrophysics/Donuts/Meshes/',  # path_mesh
        '-n', "Science20120915s1v3_134239",  # mesh_name
        '-o', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/' +
              '{0:08d}/'.format(expid),
        '-t', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',  # catalog
        '-s', '32',  # stamp size
        '-f', 'minimal',  # filter conditions
        '-rid', '{0}'.format(rid),
        '-d', '{0}'.format(date),
        '-e', '{0}'.format(expid),]
    print_command(command)
    call(command)

# <markdowncell>

# # do fits

# <codecell>

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
from routines import print_command



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
               '-e', '{0}'.format(image_number),  # expid
               ]
    print_command(command)
    call(command)

# <markdowncell>

# # collect results of specific run

# <codecell>

#!/usr/bin/env python
"""
File: 2014_january_06_2_results.py
Author: Chris Davis
Description: Go into collect_results.py for this specific run. Run second.
"""

from __future__ import print_function, division
from subprocess import call
from os import path, listdir, makedirs
import fnmatch
from routines import print_command

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/january_06/"

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
print_command(command)
call(command)

command = ['python', 'collect_results_fits.py',
           '-i', str(input_directories),
           '-o', output_directory + 'results/',  # output_directory
           '-b', str(1),  # boxdiv
           '-e', str(numbers),  # expid
           ]
print_command(command)
call(command)

# <markdowncell>

# # collate all results

# <codecell>

#!/usr/bin/env python
"""
File: 2014_january_06_3_plots.py
Author: Chris Davis
Description: Routine to submit batch jobs for creating plots. Runs third.
"""

from __future__ import print_function, division
from subprocess import call
from os import path, listdir, makedirs
import fnmatch
from routines import print_command

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/january_06/"

if not path.exists(output_directory + 'plots/'):
    makedirs(output_directory + 'plots/')

# find the ones done by the fitter
results = listdir(output_directory)
results = fnmatch.filter(results, '00*')
numbers = []
for result in results:
    numbers.append(int(result[:8]))
numbers = list(set(numbers))

# filter the ones already done
finished = listdir(output_directory + 'plots/')
finished = fnmatch.filter(finished, '00*')
finished_numbers = []
for finished_i in finished:
    finished_numbers.append(int(finished_i[:8]))
finished_numbers = list(set(finished_numbers))

numbers = [item for item in numbers if item not in finished_numbers]

input_directories = [output_directory] * len(numbers)

for iterator in xrange(len(numbers)):
    command = ['bsub',
               '-q', 'short',
               '-o', output_directory +
                     'logs/{0:08d}_plot.log'.format(numbers[iterator]),
               '-R', 'rhel60&&linux64',
               'python', 'batch_plot.py',
               '-i', str([input_directories[iterator]]),
               '-o', output_directory + 'plots/',  # output_directory
               '-m', output_directory + 'results/minuit_results.csv',  # minuit_results
               '-e', str([numbers[iterator]]),  # expid
               ]
    print_command(command)
    call(command)

# <markdowncell>

# # plot results

# <codecell>

#!/usr/bin/env python
"""
File: 2014_january_06_4_combine_plots.py
Author: Chris Davis
Description: Takes all the plots you've made and makes big plots and movies.
Runs fourth.
"""

from __future__ import print_function, division
from os import listdir
from subprocess import call

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/january_06/"
rate = 5

# filter the ones already done
finished = listdir(output_directory + 'plots/')
file_list = [output_directory + 'plots/' + finished_i + '/'
             for finished_i in finished]
file_list.sort()

entry_types = ['whisker', 'whisker_rotated', 'ellipticity', 'e0']
for entry in entry_types:
    command = ['pdfjoin']
    for file in file_list:
        command.append(file + entry + '.png')
    command.append('--outfile')
    outfile = output_directory + 'results/' + entry
    command.append(outfile + '.pdf')
    print(command)
    call(command)


    # make movie

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
               "-sOutputFile={0}-%000d.jpg".format(outfile),
               outfile + '.pdf']
    call(command, bufsize=-1)

    # now convert the jpegs to a video
    command = [#'bsub', '-q', 'short', '-o', path_logs,
               'ffmpeg', '-f', 'image2',
               '-r', '{0}'.format(rate),
               '-i', '{0}-%d.jpg'.format(outfile),
               '-r', '30', '-an', '-q:v', '0',
               outfile + '.mp4']
    call(command, bufsize=0)

    # delete the jpegs
    command = [#'bsub', '-q', 'short', '-o', path_logs,
               'find', output_directory + 'results/',
               '-name', '*.jpg', '-delete']
    call(command, bufsize=0)

