#!/usr/bin/env python
"""
File: 2013_november_16_combine_plots.py
Author: Chris Davis
Description: Takes all the plots you've made and makes big plots and movies.
Runs fourth.
"""

from __future__ import print_function, division
from os import listdir
from subprocess import call

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/november_16/"
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
