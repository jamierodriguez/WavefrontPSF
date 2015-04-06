# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# - Take a contrived example and fit to it

# <codecell>

# make FocalPlaneShell object
from focal_plane_shell import FocalPlaneShell
from focal_plane_routines import minuit_dictionary, mean_trim
from minuit_fit import Minuit_Fit
from routines_plot import data_focal_plot, data_hist_plot
path_mesh = '/Users/cpd/Desktop/Meshes/'
FP = FocalPlaneShell(path_mesh, nPixels=32)

# make coords
boxdiv = 1
max_samples_box = 10
average = mean_trim
edges = FP.decaminfo.getEdges(boxdiv=boxdiv)

coords_in = FP.random_coordinates(max_samples_box=max_samples_box, boxdiv=boxdiv)


p = { 'dz': 59.552194668975915,
      'e1': -0.0012389864322315189,
      'e2': 0.0010453971867986772,
      'rzero': 0.22176393163202593,
      'dx': -492.87699941737856,
      'dy': 1262.9265099464865,
      'xt': -15.218565800964825,
      'yt': 19.620119335545496,
      'z05d': 0.039744417978012203,
      'z06d': -0.17712627657852364,
      'z07x': -0.00026600961210990811,
      'z07y': 1.9811018580357526e-05,
      'z08x': 0.0003670440013102249,
      'z08y': 0.00048305786765613125,
      'z09d': -0.05,
      'z10d': 0.01,
      }

poles = FP.plane_averaged(p, coords=coords_in, average=average, boxdiv=boxdiv,)
poles['e1'] += p['e1']
poles['e2'] += p['e2']

# coords = FP.random_coordinates(max_samples_box=max_samples_box, boxdiv=boxdiv)
# coords_in_chip = np.unique(coords_in[:,2]).tolist()
# conds = np.array([coord[2] in coords_in_chip for coord in coords])
# coords = coords[conds]
coords = coords_in

# <codecell>

from routines_plot import focal_graph_axis
# make the fit function
FP.chi_weights = {
    'e0': 1.,
    'e1': 1.,
    'e2': 1.,
    'delta1': 1.,
    'delta2': 1.,
    'zeta1': 1.,
    'zeta2': 1.,
    }

chi2hist = {key:[] for key in FP.chi_weights}
p_hist = {key:[] for key in p}
f_list = []
FP.temp = {}
FP.temp_old = {}
output_directory = '/Users/cpd/Desktop/fits2/'
def FP_func(dz, e1, e2, rzero,
            dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y, z09d, z10d,
            ):

    in_dict_FP_func = locals().copy()

    for key in in_dict_FP_func:
        p_hist[key].append(in_dict_FP_func[key])
    
    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict_FP_func.keys():
        if np.isnan(in_dict_FP_func[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FPS.remakedonut()
            return 1e20

    poles_i = FP.plane_averaged(in_dict_FP_func, coords=coords, average=average, boxdiv=boxdiv)
    poles_i['e1'] += e1
    poles_i['e2'] += e2
    
    FP.temp, FP.temp_old = poles_i, FP.temp
    chi2 = 0
    for key in FP.chi_weights:
        val_a = poles[key]
        val_b = poles_i[key]
        var = poles['var_{0}'.format(key)]
        weight = FP.chi_weights[key]
        
        chi2_i = np.square(val_a - val_b) / var
        chi2hist[key].append(chi2_i)
        chi2 += np.sum(weight * chi2_i)
    
    if (chi2 < 0) + (np.isnan(chi2)):
        chi2 = 1e20
        
    # update the chi2 by *= 1. / (Nobs - Nparam)
    chi2 *= 1. / (len(poles_i['e1']) * sum([FP.chi_weights[i] for i in FP.chi_weights]) -
                  len(in_dict_FP_func.keys()))
    return chi2

# <codecell>

def SaveFunc(steps,
             dz, e1, e2, rzero,
            dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y, z09d, z10d,
            ):
    
    in_dict_FP_func = locals().copy()
    
    poles_i = FP.temp
    # compare
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex='all', sharey='all', figsize=(12*4, 12*2))
    figures = {'e': fig, 'w': fig, 'zeta': fig, 'delta': fig}
    focal_keys = figures.keys()
    for ij in range(axs.shape[0]):
        for jk in range(axs.shape[1]):
            axs[ij][jk] = focal_graph_axis(axs[ij][jk])
    axes = {'e': axs[0,0], 'w': axs[0,1], 'delta': axs[1,0], 'zeta': axs[1,1]}
    figures_hist = {'e0subs': fig, 'e0': fig, 'e0prime': fig}
    axes_hist = {'e0subs': axs[1,3], 'e0': axs[0,2], 'e0prime': axs[1,2]}
    figures, axes, scales = data_focal_plot(poles,
                                            color='r', boxdiv=boxdiv,
                                            figures=figures, axes=axes,
                                            keys=focal_keys,
                                            )
    # plot the comparison
    figures, axes, scales = data_focal_plot(poles_i,
                                            color='b', boxdiv=boxdiv,
                                            figures=figures, axes=axes, scales=scales,
                                            keys=focal_keys,
                                            )

    poles_i.update({
                  'e0subs': poles['e0'] - poles_i['e0'],
                  'e0prime': poles_i['e0'],
                  'e0': poles['e0']
                  })
    figures_hist, axes_hist, scales_hist = data_hist_plot(poles_i, edges,
                                                          figures=figures_hist,
                                                          axes=axes_hist,
                                                          keys=['e0subs', 'e0', 'e0prime'])
    
    # make tables for param and delta and chi2
    colLabels = ("Parameter", "Value", "Delta")
    cellText = [[key, '{0:.3e}'.format(p_hist[key][-1]),
                 '{0:.3e}'.format(p_hist[key][-1] - p_hist[key][-2])] for key in p.keys()]
    # add in chi2s
    cellText += [[key, 
                  '{0:.3e}'.format(np.sum(chi2hist[key][-1])), 
                  '{0:.3e}'.format(np.sum(chi2hist[key][-1] - chi2hist[key][-2]))] 
                 for key in chi2hist.keys()]
    chi2 = 0
    chi2old = 0
    for key in FP.chi_weights:
        weight = FP.chi_weights[key]
        chi2 += np.sum(weight * chi2hist[key][-1])
        chi2old += np.sum(weight * chi2hist[key][-2])
    chi2delta = chi2 - chi2old
    cellText += [['total chi2', '{0:.3e}'.format(chi2), '{0:.3e}'.format(chi2delta)]]
    axs[0,3].axis('off')
    table = axs[0,3].table(cellText=cellText, colLabels=colLabels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1, 2)
    

    plt.tight_layout()
    fig.savefig(output_directory + '{0:04d}.png'.format(steps))
    plt.close('all')

# <codecell>

ptest = {'dx': 0,
 'dy': 0,
 'dz': 0,
 'e1': 0,
 'e2': 0,
 'rzero': 0.125,
 'xt': 0,
 'yt': 0,
 'z05d': 0,
 'z06d': 0,
 'z07x': 0,
 'z07y': 0,
 'z08x': 0,
 'z08y': 0,
 'z09d': 0,
 'z10d': 0}

# <codecell>

%timeit FP_func(**ptest)

# <codecell>

%timeit SaveFunc(0, **ptest)

# <codecell>

# fit
par_names = p.keys()
verbosity = 3
force_derivatives = 0
strategy = 1
tolerance = 40
h_base = 1e-3
max_iterations = len(par_names) * 1000

# set up initial guesses
minuit_dict, h_dict = minuit_dictionary(par_names, h_base=h_base)
# for pkey in p:
#     minuit_dict[pkey] = p[pkey]
# do fit
minuit_fit = Minuit_Fit(FP_func, minuit_dict, par_names=par_names,
                        SaveFunc=SaveFunc,
                        save_iter=1,
                        h_dict=h_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)
# fix all but rzero
for key in par_names:
    if (key != 'rzero') * (key != 'dz'):
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = True
# for key in FP.chi_weights:
#     if key != 'e0':
#         FP.chi_weights[key] = 0
minuit_fit.setupFit()
minuit_fit.doFit()
minuit_results_rzero = minuit_fit.outFit()
nCalls = minuit_fit.nCalls + 1000
# defix and update the unfixed
for key in par_names:
    # unfix the fixed
    if minuit_fit.minuit_dict['fix_{0}'.format(key)]:
        minuit_fit.minuit_dict['fix_{0}'.format(key)] = False
    # update the fixed
    else:
        minuit_fit.minuit_dict[key] = minuit_results_rzero['args'][key]
# for key in FP.chi_weights:
#     FP.chi_weights[key] = 1
minuit_fit.setupFit()
minuit_fit.nCalls = nCalls
minuit_fit.doFit()
minuit_results = minuit_fit.outFit()

# <codecell>

for key in sorted(minuit_results['args']):
    print(key, p[key], minuit_results['args'][key], minuit_results['errors'][key])

# <codecell>

for key in sorted(minuit_results):
    print(key, minuit_results[key])

# <codecell>

for key in sorted(minuit_dict):
    print(key, minuit_dict[key])

# <codecell>

# compare
figures, axes, scales = data_focal_plot(poles,
                                        color='r', boxdiv=boxdiv,
                                        )

# plot the comparison
poles_i = FP.plane_averaged(minuit_results['args'], coords=coords, average=average, boxdiv=boxdiv)
poles_i['e1'] += minuit_results['args']['e1']
poles_i['e2'] += minuit_results['args']['e2']
figures, axes, scales = data_focal_plot(poles_i,
                                        color='b', boxdiv=boxdiv,
                                        figures=figures, axes=axes, scales=scales,
                                        )
poles_comp = {'x_box': poles['x_box'], 'y_box': poles['y_box'],
              'e0_diff': poles['e0'] - poles_i['e0'],
              }
keys = ['e0_diff']
figures_hist, axes_hist, scales_hist = data_hist_plot(poles_comp, edges, keys=keys,
                                        )

# <codecell>

# plot chi2s
plt.figure(figsize=(12,12), dpi=300)
for key in sorted(chi2hist.keys()):
    chis = chi2hist[key]
    x = range(len(chis))
    y = []
    for i in x:
        y.append(np.sum(chis[i]) * FP.chi_weights[key])
    plt.plot(x, np.log10(y), label=key)
plt.legend()

# <codecell>

# plot values
for key in sorted(p_hist.keys()):
    plt.figure(figsize=(12,12), dpi=300)
    x = range(len(p_hist[key]))
    y = []
    for i in x:
        y.append(np.sum(p_hist[key][i]))
    plt.plot(x, np.log10(y), label=key)
    plt.title('{0}'.format(key))

# <codecell>


