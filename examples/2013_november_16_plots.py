from subprocess import call
from os import path, listdir, makedirs
import fnmatch

def print_command(command):
    string = ''
    for i in command:
        string += str(i)
        string += ' '
    print(string)
    return string

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/november_16/"

if not path.exists(output_directory):
    makedirs(output_directory)
    makedirs(output_directory + 'logs/')
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
               '-e', str([numbers[iterator]]),  # expid
               '-o', output_directory + 'plots/',  # output_directory
               ]
    print_command(command)
    call(command)
