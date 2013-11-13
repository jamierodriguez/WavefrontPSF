from __future__ import print_function, division
import argparse
import matplotlib
matplotlib.use('Agg')
#from pylab import *
import numpy as np
from matplotlib import pyplot as plt
from minuit_fit import Minuit_Fit
from chi_class_plot import FocalPlane_Plotter
from chi_class import create_minuit_dictionary
from iminuit import describe
from os import path, makedirs

# TODO: add a command for extracting the resultant dicts
# TODO: organize this into a set of functions instead of a long script

"""
bsub commands

from subprocess import call
from os import path, makedirs

def print_command(command):
    string = ''
    for i in command:
        string += str(i)
        string += ' '
    print(string)
    return string


output_directory = "/nfs/slac/g/ki/ki18/cpd/focus/september_18/"

if not path.exists(output_directory):
    makedirs(output_directory)
    makedirs(output_directory + 'logs')

for image_number in range(190701, 190918):
    if not path.exists(output_directory + '/{0:08d}'.format(image_number)):
        command = ['bsub',
                   '-q', 'xlong',
                   '-o', output_directory +
                         '/logs/{0:08d}.log'.format(image_number),
                   '-R', 'rhel60&&linux64',
                   'python', 'september18_batch_fit.py',
                   '-i', '{0}'.format(image_number),
                   '-o', output_directory,
                   '-r', str(0),
                   '-s', str(5),
                   '-b', str(0)]
        print_command(command)
        call(command)

"""

parser = argparse. \
    ArgumentParser(description=
                   'Fit image and dump results.')
parser.add_argument("-i",
                    dest="image_number",
                    type=int,
                    help="what image number will we fit now?")
parser.add_argument("-o",
                    dest="output_directory",
                    default="/nfs/slac/g/ki/ki18/cpd/focus/september_18/",
                    help="where will the outputs go (modulo image number)")
parser.add_argument("-r",
                    dest="doRandom",
                    default=0,
                    type=int,
                    help="Do random, or the real data?")
parser.add_argument("-s",
                    dest="max_stars",
                    default=5,
                    type=int,
                    help="How many stars per chip?")
parser.add_argument("-b",
                    dest="boxdiv",
                    default=0,
                    type=int,
                    help="Division of chips. 0 is full, 1 is one division...")
options = parser.parse_args()
args_dict = vars(options)
chi_weights = {
    'e0': 1.,
    'e1': 1.,
    'e2': 1.,
    }
order_dict = {
    'x2': {'p': 2, 'q': 0},
    'y2': {'p': 0, 'q': 2},
    'xy': {'p': 1, 'q': 1},
    }

if args_dict['image_number'] < 160000:
    externalconds = "(data['MAG_AUTO'] < 14) * " + \
        "(data['MAG_AUTO'] > 10) * (data['FWHM_WORLD'] > 0) * " + \
        "(data['FWHM_WORLD'] * 60 ** 2 < {0})".format(1.3)
elif args_dict['image_number'] < 190000:
    externalconds = "(data['MAG_AUTO'] < 15) * " + \
        "(data['MAG_AUTO'] > 10) * (data['FWHM_WORLD'] > 0) * " + \
        "(data['FWHM_WORLD'] * 60 ** 2 < {0})".format(1.3)
else:
    externalconds = None

FP_dict = dict(
    image_number=args_dict['image_number'],
    verbosity=1,
    subav=False,
    average=np.mean,
    calculate_comparison=not args_dict['doRandom'],
    max_stars=args_dict['max_stars'],
    boxdiv=args_dict['boxdiv'],
    chi_weights=chi_weights,
    order_dict=order_dict,
    randomFlag=False,
    apply_image_correction=True,
    externalconds=externalconds,
    path_enter='{0:08d}'.format(args_dict['image_number']),
    path_base=args_dict['output_directory'])

FP = FocalPlane_Plotter(**FP_dict)

# set up FP_func, the thing we will fit


#def FP_func(dz, e1, e2, rzero, dx, dy, xt, yt, z05d, z06d, z09d, z10d,
#            z04x, z04y, z07x, z07y, z08x, z08y, z09x, z09y, z10x, z10y):
def FP_func(dz, e1, e2, rzero, dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y):
    in_dict = locals().copy()
    chi, update_dict = FP(in_dict)

    return chi

FP_keys = describe(FP_func)
minuit_dict = create_minuit_dictionary(FP_keys)
# random plane creation
if args_dict['doRandom']:
    path_comparison_type = 'random'
    path_comparison = {'keys': FP_keys,
                       'minuit_dict': minuit_dict}
    FP.comparison = FP.create_comparison(
        path_comparison, FP.coords,
        path_comparison_type)
    ran_dict = FP.ran_dict
    FP.comparison_average = FP.create_average_dictionary(
        FP.comparison,
        boxdiv=FP.boxdiv, subav=FP.subav)
    FP.rzero = FP.fwhm_to_rzero(FP.average(FP.comparison['fwhm']))

    # show that these make reasonable plots
    output_directory_random = FP.path_base + FP.path_enter + '/random/'
    mom_ran, mom_ran_av = FP.create_plane(
        ran_dict, FP.coords,
        order_dict=FP.order_dict)
    fig_dict_random = FP.comparison_graph_routine(
        mom_ran_av,
        FP.comparison_average,
        title='random_dict {0:08d}'.format(FP_dict['image_number']),
        output_directory=output_directory_random,
        scale=1. / (1. + FP.boxdiv))
    print('key\tminuit\t\trandom')
    for key in FP_keys:
        print(key, '\t{0:.4e}'.format(minuit_dict[key]),
              '\t{0:.4e}'.format(ran_dict[key]))
else:
    ran_dict = None
    if FP.verbosity >= 2:
        # also plot the data
        plt.figure()
        extent = (8, 20, 0, 4)
        plt.title('{0}'.format(FP_dict['image_number']))
        combined = plt.hexbin(
            FP.comparison_data['MAG_AUTO'],
            FP.comparison_data['FWHM_WORLD'] * 60 ** 2,
            extent=extent,
            bins='log',
            C=FP.comparison_data['CLASS_STAR'])
        plt.colorbar(combined)
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        output_directory = FP.path_base + FP.path_enter + '/star_galaxy/'
        if not path.exists(path.dirname(output_directory)):
            makedirs(path.dirname(output_directory))
        plt.savefig(output_directory + 'filtered.pdf')

        plt.figure()
        datall = plt.hexbin(
            FP.comparison_data_unfiltered['MAG_AUTO'],
            FP.comparison_data_unfiltered['FWHM_WORLD'] * 60 ** 2,
            extent=extent,
            bins='log',
            C=FP.comparison_data_unfiltered['CLASS_STAR'])
        plt.colorbar(datall)
        plt.xlim(extent[0], extent[1])
        plt.ylim(extent[2], extent[3])
        if not path.exists(path.dirname(output_directory)):
            makedirs(path.dirname(output_directory))
        plt.savefig(output_directory + 'unfiltered.pdf')


# set up minuit
FP.wipe_history()
par_names = describe(FP_func)
verbosity = 3
force_derivatives = 1
grad_dict = dict(h_base=1e-1)
strategy = 1
tolerance = 40
max_iterations = len(FP_keys) * 100

minuit_fit = Minuit_Fit(FP_func, minuit_dict, par_names, grad_dict=grad_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)

# do the usual fit
FP.wipe_history()
minuit_fit.setupFit(minuit_dict)

minuit_fit.doFit()

minuit_results = minuit_fit.outFit()

# graphs!
output_directory = FP.path_base + FP.path_enter + '/'

# check the output_directory exists
if not path.exists(output_directory):
    makedirs(output_directory)

# save the minuit_results
np.save(output_directory + 'minuit_results', minuit_results)
if args_dict['doRandom']:
    np.save(output_directory + 'ran_dict', ran_dict)

## # create the history summary
## FP.create_history_summary_images(
##     output_directory=output_directory, keys=FP_keys, ran_dict=ran_dict)



scale = 1. / (1. + FP.boxdiv)

comparison_graph_routine_keys = [
    'box', 'ellipticity', 'whisker', 'e0', 'fwhm',
    'errors', 'chi2', 'compare', 'rotate', 'circles'
    ]
if 'x3' in FP.order_dict:
    comparison_graph_routine_keys.append('octupole')
in_dict = minuit_results['args']
if 'rzero' not in in_dict.keys():
    in_dict.update({'rzero': FP.rzero})

moments, moments_average = FP.create_plane(in_dict, FP.coords,
                                           order_dict=FP.order_dict)
output_directory = FP.path_base + FP.path_enter + '/fit/'
fig_dict = FP.comparison_graph_routine(moments_average, FP.comparison_average,
                                       title='fit {0:08d}'.format(FP_dict['image_number']),
                                       output_directory=output_directory,
                                       scale=scale,
                                       keys=comparison_graph_routine_keys)

# make sure inputs still work for random
if args_dict['doRandom']:
    output_directory_random = FP.path_base + FP.path_enter + '/random/'
    mom_ran, mom_ran_av = FP.create_plane(ran_dict, FP.coords,
                                          order_dict=FP.order_dict)
    fig_dict_random = FP.comparison_graph_routine(
        mom_ran_av,
        FP.comparison_average,
        title='random_dict {0:08d}'.format(FP_dict['image_number']),
        output_directory=output_directory_random,
        scale=scale,
        keys=comparison_graph_routine_keys)
# if we have history, look at the history
if FP.verbosity >= 1:
    output_directory_initial = FP.path_base + FP.path_enter + '/initial/'
    if FP.verbosity >= 2:
        moments_average_initial = FP.history['moments_average'][0]
    else:
        moments_initial, moments_average_initial = FP.create_plane(
            FP.history['in_dict'][0], FP.coords, order_dict=FP.order_dict)
    fig_dict_original = FP.comparison_graph_routine(
        moments_average_initial, FP.comparison_average,
        output_directory=output_directory_initial, title='initial {0:08d}'.format(FP_dict['image_number']),
        scale=scale,
        keys=comparison_graph_routine_keys)

# what if we exclude the variables that are commensurate with zero?  take the
# ones with values less than the error and make a comparison, to see if that
# matters...

significance_dict = {}
error_dict = minuit_results['errors']
in_dict = minuit_results['args']
for key in FP_keys:
    if abs(in_dict[key]) > 3 * abs(error_dict[key]):
        significance_dict.update({key: in_dict[key]})
        print(key, '\t', '{0:.4e}'.format(in_dict[key]), '\t',
              '{0:.4e}'.format(error_dict[key]))
    else:
        print(key, '\t', '{0:.4e}'.format(in_dict[key]), '\t',
              '{0:.4e}'.format(error_dict[key]), '\t', 'NOT')
moments_sig, moments_average_sig = FP.create_plane(
    significance_dict, FP.coords, order_dict=FP.order_dict)
output_directory = FP.path_base + FP.path_enter + '/3sigfit/'
fig_dict = FP.comparison_graph_routine(
    moments_average_sig, FP.comparison_average,
    title='3sig {0:08d}'.format(FP_dict['image_number']), output_directory=output_directory,
    scale=scale,
    keys=comparison_graph_routine_keys)
output_directory = FP.path_base + FP.path_enter + '/3sigfit_vs_fit/'
fig_dict = FP.comparison_graph_routine(
    moments_average_sig, moments_average,
    title='3sig vs full {0:08d}'.format(FP_dict['image_number']), output_directory=output_directory,
    scale=scale,
    keys=comparison_graph_routine_keys)

# show results
# get current iteration
error_dict = minuit_results['errors']
in_dict = minuit_results['args']
if args_dict['doRandom']:
    print('key\tfit\t\terror\t\tminuit\t\tfirst\t\trandom')
else:
    print('key\tfit\t\terror\t\tminuit\t\tfirst')
for key in FP_keys:
    if args_dict['doRandom']:
        print(key, '\t{0: .4e}'.format(in_dict[key]),
              '\t{0: .4e}'.format(error_dict[key]),
              '\t{0: .4e}'.format(minuit_dict[key]),
              '\t{0: .4e}'.format(FP.history['in_dict'][0][key]),
              '\t{0: .4e}'.format(ran_dict[key]))
    else:
        print(key, '\t{0: .4e}'.format(in_dict[key]),
              '\t{0: .4e}'.format(error_dict[key]),
              '\t{0: .4e}'.format(minuit_dict[key]),
              '\t{0: .4e}'.format(FP.history['in_dict'][0][key]))

# print the sigma stuff here, too
for key in FP_keys:
    if abs(in_dict[key]) > 3 * abs(error_dict[key]):
        print(key, '\t', '{0:.4e}'.format(in_dict[key]), '\t',
              '{0:.4e}'.format(error_dict[key]))
    else:
        print(key, '\t', '{0:.4e}'.format(in_dict[key]), '\t',
              '{0:.4e}'.format(error_dict[key]), '\t', 'NOT')

# and, finally, save the focal plane
output_directory = FP.path_base + FP.path_enter + '/'
FP.save(output_directory + 'FocalPlane.pkl')
