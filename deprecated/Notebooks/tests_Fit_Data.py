# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# basically take the catalog python script and put it here!

# <codecell>

import numpy as np

expid = 232697
path_mesh = '/Users/cpd/Desktop/Meshes/'

# remake the catalogs...
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

ith = np.argwhere(expid == expids)[0]
rid = rids[ith]
date = dates[ith]

dataDirectory = "/Volumes/Seagate/Images/"
dataDirectory = "/Users/cpd/Desktop/Images/"

args_dict = {'expid': expid,
             'path_mesh': path_mesh,
             'mesh_name': "Science20120915s1v3_134239",
             'output_directory': dataDirectory + '{0:08d}/'.format(expid),
             'catalogs': dataDirectory,
             'rid': rid,
             'date': date,
             'conds': 'default',
             'size': 32,
             }

parser_transform = {'expid': '-e',
                    'path_mesh': '-m',
                    'mesh_name': '-n',
                    'output_directory': '-o',
                    'catalogs': '-t',
                    'rid': '-rid',
                    'date': '-d',
                    'size': '-s',
                    'conds': '-f',}

# <markdowncell>

# # Call catalog.py for specific expid

# <codecell>

# call catalog.py
code_path = '/Users/cpd/Dropbox/secret-adventure/batch/catalog.py'


from subprocess import call
from routines import print_command
command = ['python', code_path]
for args_dict_i in args_dict:
    command.append(parser_transform[args_dict_i])
    command.append(str(args_dict[args_dict_i]))

print_command(command)
call(command)

# <codecell>

!echo $PYTHONPATH

# <markdowncell>

# # Begin fit procedure

# <codecell>

# make FocalPlane object
from focal_plane import FocalPlane
from routines import minuit_dictionary, mean_trim
from routines_files import generate_hdu_lists
from minuit_fit import Minuit_Fit
from routines_plot import data_focal_plot, data_hist_plot

list_catalogs, list_fits_extension, list_chip = \
        generate_hdu_lists(expid,
                           path_base=dataDirectory)

average = mean_trim
boxdiv = 0
chi_weights = {
    'e0': 1.,
    'e1': 1.,
    'e2': 1.,
    'delta1': 1.,
    'delta2': 1.,
    'zeta1': 1.,
    'zeta2': 1.,
    }
conds = 'default'
max_samples_box = 5
nPixels = 16
output_directory = '/Users/cpd/Desktop/fits/'


FP = FocalPlane(list_catalogs=list_catalogs,
                list_fits_extension=list_fits_extension,
                list_chip=list_chip,
                boxdiv=boxdiv,
                max_samples_box=max_samples_box,
                conds=conds,
                average=average,
                path_mesh=path_mesh,
                nPixels=nPixels,
                chi_weights=chi_weights
                )

edges = FP.decaminfo.getEdges(boxdiv=boxdiv)

# <codecell>

# define the fit function
chi2hist = []
def FitFunc(dz,
            rzero,
            dx, dy, xt, yt,
            e1, e2,
            z05d, z06d,
            z07x, z07y,
            z08x, z08y,
            z09d,
            z10d,
            ):

    in_dict_FP_func = locals().copy()
    
    
    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict_FP_func.keys():
        if np.isnan(in_dict_FP_func[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FP.remakedonut()
            return 1e20

    poles_i = FP.plane_averaged(in_dict_FP_func, coords=FP.coords,
                                average=FP.average, boxdiv=FP.boxdiv,
                                subav=FP.subav)
    poles_i['e1'] += e1
    poles_i['e2'] += e2
    
    FP.temp_plane = poles_i
    
    chi2 = FP - poles_i
    chi2hist.append(chi2)
    
    return chi2['chi2']

# <codecell>

# define the save function
from routines_plot import save_func
def SaveFunc(steps):
    in_dict = {'steps': steps,
               'state_history': FP.history,
               'chisquared_history': chi2hist,
               'chi_weights': FP.chi_weights,
               'plane': FP.temp_plane,
               'reference_plane': FP.data,
               'output_directory': output_directory,
               'boxdiv': FP.boxdiv}
    save_func(**in_dict)
    
    return

# <codecell>

# let's plot stars because why not
n_iter = 10
for i in xrange(0, FP.recdata.size, n_iter):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    fig, ax = plot_star(FP.recdata[i], nPixels=nPixels)
    ax.set_title('{0}, {1}'.format(i, FP.recdata[i]['CHIP']))
    fig.savefig(output_directory + 'stars_{0:04d}.png'.format(i))
    plt.close('all')

# <markdowncell>

# # do Minuit fit

# <codecell>

p = {'dx': -234.491545,
     'dy':  293.39127,
     'dz':  29.452629,
     'xt':  0.772277,
     'yt':  9.760292,
     'z05d':  0.049454,
     'z06d': -0.246456,
     'z07x': -0.000812,
     'z07y':   7.50000000e-05,
     'z08x':   3.00000000e-06,
     'z08y': -0.000618,
     'z09d': -0.06377,
     'z10d':  0.049792,
     'rzero': 0.125,
     'e1': 0,
     'e2': 0,
     }
# fit
# TODO: set option to let minuit do the derivative itself
par_names = p.keys()
verbosity = 3
force_derivatives = 1
strategy = 1
tolerance = 1
h_base = 1e-3
save_iter = 1
max_iterations = len(par_names) * 1000
minuit_results_list = []
# set up initial guesses
minuit_dict, h_dict = minuit_dictionary(par_names, h_base=h_base)
for pkey in p:
    minuit_dict[pkey] = p[pkey]
# setup object
minuit_fit = Minuit_Fit(FitFunc, minuit_dict, par_names=par_names,
                        SaveFunc=SaveFunc,
                        save_iter=save_iter,
                        h_dict=h_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)

# <markdowncell>

# # try all the combinations

# <rawcell>

# # do e0
# # TODO: maybe cut the dz out of here? the other params matter too
# # (though a lil less than dz) so maybe no need to include it here...
# for key in par_names:
#     if (key != 'rzero') * (key != 'dz'):
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
#     else:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
# 
# for key in FP.chi_weights:
#     if (key != 'e0'):
#         FP.chi_weights[key] = 0
#     else:
#         FP.chi_weights[key] = 1
# minuit_fit.setupFit()
# minuit_fit.doFit()

# <rawcell>

# minuit_results_list.append(minuit_fit.outFit())
# nCalls = int(minuit_fit.nCalls / 1000) * 1000
# np.save(output_directory + 'minuit_results', minuit_results_list)
# # do delta1
# for key in par_names:
#     # unfix the fixed
#     if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
#     else:
#         # use the errors from our fixed ones.
#         minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]
# 
#     minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
#         
#     if (key != 'z10d'):
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
#     else:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
# 
# for key in FP.chi_weights:
#     if (key != 'delta1'):
#         FP.chi_weights[key] = 0
#     else:
#         FP.chi_weights[key] = 1
# minuit_fit.setupFit()
# minuit_fit.nCalls = nCalls + 1000
# minuit_fit.doFit()

# <rawcell>

# minuit_results_list.append(minuit_fit.outFit())
# nCalls = int(minuit_fit.nCalls / 1000) * 1000
# np.save(output_directory + 'minuit_results', minuit_results_list)
# # do delta2
# for key in par_names:
#     # unfix the fixed
#     if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
#     else:
#         # use the errors from our fixed ones.
#         minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]
# 
#     minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
# 
#     if (key != 'z09d'):
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
#     else:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
# 
# for key in FP.chi_weights:
#     if (key != 'delta2'):
#         FP.chi_weights[key] = 0
#     else:
#         FP.chi_weights[key] = 1
# minuit_fit.setupFit()
# minuit_fit.nCalls = nCalls + 1000
# minuit_fit.doFit()

# <rawcell>

# minuit_results_list.append(minuit_fit.outFit())
# nCalls = int(minuit_fit.nCalls / 1000) * 1000
# np.save(output_directory + 'minuit_results', minuit_results_list)
# # do zeta1
# for key in par_names:
#     # unfix the fixed
#     if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
#     else:
#         # use the errors from our fixed ones.
#         minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]
# 
#     minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
# 
#     if (key != 'z08d'):
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
#     else:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
# 
# for key in FP.chi_weights:
#     if (key != 'zeta1'):
#         FP.chi_weights[key] = 0
#     else:
#         FP.chi_weights[key] = 1
# minuit_fit.setupFit()
# minuit_fit.nCalls = nCalls + 1000
# minuit_fit.doFit()

# <rawcell>

# minuit_results_list.append(minuit_fit.outFit())
# nCalls = int(minuit_fit.nCalls / 1000) * 1000
# np.save(output_directory + 'minuit_results', minuit_results_list)
# # do zeta2
# for key in par_names:
#     # unfix the fixed
#     if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
#     else:
#         # use the errors from our fixed ones.
#         minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]
# 
#     minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
# 
#     if (key != 'z07d'):
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
#     else:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
# 
# for key in FP.chi_weights:
#     if (key != 'zeta2'):
#         FP.chi_weights[key] = 0
#     else:
#         FP.chi_weights[key] = 1
# minuit_fit.setupFit()
# minuit_fit.nCalls = nCalls + 1000
# minuit_fit.doFit()

# <rawcell>

# minuit_results_list.append(minuit_fit.outFit())
# nCalls = int(minuit_fit.nCalls / 1000) * 1000
# np.save(output_directory + 'minuit_results', minuit_results_list)
# # do e's
# for key in par_names:
#     # unfix the fixed
#     if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
#     else:
#         # use the errors from our fixed ones.
#         minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]
# 
#     minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
# 
#     if (key != 'dz') * (key != 'dx') * (key != 'dy') * (key != 'xt') * (key != 'yt') * (key != 'e1') * (key != 'e2'):
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
#     else:
#         minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
# 
# for key in FP.chi_weights:
#     if (key != 'e1') * (key != 'e2'):
#         FP.chi_weights[key] = 0
#     else:
#         FP.chi_weights[key] = 1
# minuit_fit.setupFit()
# minuit_fit.nCalls = nCalls + 1000
# minuit_fit.doFit()

# <markdowncell>

# # unfix all parameters

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000
np.save(output_directory + 'minuit_results', minuit_results_list)
# do the generic fit:
# defix and update the unfixed
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    else:
        # use the errors from our fixed ones.
        minuit_fit.minuit_dict['error_{0}'.format(key)] = minuit_results_list[-1]['errors'][key]

    minuit_fit.minuit_dict[key] = minuit_results_list[-1]['args'][key]
for key in FP.chi_weights:
    FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls + 1000
minuit_fit.doFit()

# <markdowncell>

# # out the results

# <codecell>

minuit_results_list.append(minuit_fit.outFit())
nCalls = int(minuit_fit.nCalls / 1000) * 1000 + 1000
np.save(output_directory + 'minuit_results', minuit_results_list)

# <markdowncell>

# # assess how well we did

# <codecell>

minuit_results = minuit_fit.outFit()
for key in sorted(minuit_results['args']):
    print(key, p[key], minuit_results['args'][key], minuit_results['errors'][key])
for key in sorted(minuit_results):
    if (key == 'correlation') + (key == 'covariance'):
        continue
    print(key, minuit_results[key])
for key in sorted(minuit_dict):
    print(key, minuit_dict[key])

# <codecell>

# compare
chi2out = FitFunc(**minuit_results['args'])
SaveFunc(9990)

# <codecell>

# plot chi2s
plt.figure(figsize=(12,12), dpi=300)
for key in sorted(chi2hist[-1].keys()):
    x = range(len(chi2hist))
    y = []
    for i in x:
        y.append(np.sum(chi2hist[i][key]))
    plt.plot(x, np.log10(y), label=key)
plt.legend()

# <codecell>

# plot values
for key in sorted(FP.history[-1].keys()):
    plt.figure(figsize=(12,12), dpi=300)
    x = range(len(FP.history))
    y = []
    for i in x:
        y.append(np.sum(FP.history[i][key]))
    plt.plot(x, y, label=key)
    plt.title('{0}'.format(key))

