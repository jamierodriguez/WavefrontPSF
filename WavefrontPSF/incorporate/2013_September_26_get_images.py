# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# nightsums:
# 
# http://decam03.fnal.gov:8080/nightsum/nightsum-2013-09-05/nightsum.html
# 
# http://decam03.fnal.gov:8080/nightsum/nightsum-2013-09-10/nightsum.html
# 
# http://decam03.fnal.gov:8080/nightsum/nightsum-2013-09-12/nightsum.html

# <codecell>

from subprocess import call
from os import path, makedirs

def print_command(command):
    string = ''
    for i in command:
        string += str(i)
        string += ' '
    print(string)
    return string

input_list = [
    [20130906105326, 20130905, 231046, 231053],
    [20130906105326, 20130905, 231089, 231096],
    [20130911103044, 20130910, 232608, 232849],
    [20130913151017, 20130912, 233377, 233571],
    [20130913151017, 20130912, 233584, 233642],
    ]

for i in input_list:
    rid, date, minImage, maxImage = i

    command = ['bsub',
               '-q', 'medium',
               '-o', '/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript' +
                     '/logs/{0:08d}_{1:08d}.log'.format(minImage, maxImage),
               '-R', 'rhel60&&linux64',
               'python', '/afs/slac.stanford.edu/u/ki/cpd/makedonuts/wgetscript.py',
               '-min', '{0}'.format(minImage),
               '-max', '{0}'.format(maxImage),
               '-i', '0',
               '-rid', '{0}'.format(rid),
               '-d', '{0}'.format(date)]
    print_command(command)
    call(command)

# <codecell>


