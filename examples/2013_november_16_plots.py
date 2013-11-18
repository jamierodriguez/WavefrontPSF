from subprocess import call
from os import listdir


def print_command(command):
    string = ''
    for i in command:
        string += str(i)
        string += ' '
    print(string)
    return string

output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/november_16/"

# now go to output and filter by what is already present
results = listdir(output_directory)
if 'logs' in results:
    results.remove('logs')
numbers = []
for result in results:
    numbers.append(int(result[:8]))
numbers = list(set(numbers))
input_directories = [output_directory] * len(numbers)

command = ['bsub',
           '-q', 'xlong',
           '-o', output_directory +
                 'logs/plots.log',
           '-R', 'rhel60&&linux64',
           'python', 'batch_plot.py',
           '-i', str(input_directories),
           '-e', str(numbers),  # expid
           '-o', output_directory,  # output_directory
           ]
#print_command(command)
call(command)
