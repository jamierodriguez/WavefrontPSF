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
