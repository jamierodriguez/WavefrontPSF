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
    makedirs(output_directory + 'results/')

# now go to output and filter by what is already present
results = listdir(output_directory)
results = fnmatch.filter(results, '00*')
numbers = []
for result in results:
    numbers.append(int(result[:8]))
numbers = list(set(numbers))
input_directories = [output_directory] * len(numbers)

command = ['python', 'batch_plot.py',
           '-i', str(input_directories),
           '-e', str(numbers),  # expid
           '-o', output_directory + 'results/',  # output_directory
           ]
#print_command(command)
call(command)
