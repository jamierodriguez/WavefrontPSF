# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# The intention with this notebook is to take a focal plane and vary the parameters and also find the derivative with these variations (as a function of h). The 'chi2' will be in comparison with 0 inputs.

# <codecell>

# make FocalPlaneShell object
from focal_plane_shell import FocalPlaneShell
from focal_plane_routines import minuit_dictionary, mean_trim
path_mesh = '/Users/cpd/Desktop/Meshes/'
nPixels = 32
FPS = FocalPlaneShell(path_mesh, nPixels)

# make coords
boxdiv=1
max_samples_box = 20
coords = FPS.random_coordinates(max_samples_box=max_samples_box, boxdiv=boxdiv)
average = mean_trim
edges = FPS.decaminfo.getEdges(boxdiv=boxdiv)

keys = ['dz', 'rzero',
        'dx', 'dy', 'xt', 'yt',
        'z04x', 'z04y',
        'z05d', 'z05x', 'z05y',
        'z06d', 'z06x', 'z06y',
        'z07d', 'z07x', 'z07y',
        'z08d', 'z08x', 'z08y',
        'z09d', 'z09x', 'z09y',
        'z10d', 'z10x', 'z10y',
        'z11d', 'z11x', 'z11y',
        ]
p_base = {key: 0 for key in keys}
p_base.update({'rzero': 0.14})
comparison = FPS.plane_averaged(p_base, coords=coords, average=average, boxdiv=boxdiv)

# <codecell>

# make the fit function
chi_weights = {
    'e0': 1.,
    'e1': 1.,
    'e2': 1.,
    'delta1': 1.,
    'delta2': 1.,
    'zeta1': 1.,
    'zeta2': 1.,
    }

def FP_func(dz, rzero,
            dx, dy, xt, yt,
            z04x, z04y,
            z05d, z05x, z05y,
            z06d, z06x, z06y,
            z07d, z07x, z07y,
            z08d, z08x, z08y,
            z09d, z09x, z09y,
            z10d, z10x, z10y,
            z11d, z11x, z11y,
            ):

    in_dict_FP_func = locals().copy()

    poles = FPS.plane_averaged(in_dict_FP_func, coords=coords, average=average, boxdiv=boxdiv)
    
    inv_dof = 1. / (len(comparison['e1']) - len(in_dict_FP_func.keys()))
    
    chi2dict = {}
    for key in chi_weights:
        val_a = comparison[key]
        val_b = poles[key]
        var = comparison['var_{0}'.format(key)]
        
        chi2 = np.square(val_a - val_b) / var
        
        # update the chi2 by *= 1. / (Nobs - Nparam)
        chi2 *= inv_dof
            
        chi2dict.update({key: chi2})
    
    return chi2dict

# <codecell>

output_directory = '/Users/cpd/Desktop/derivatives2/'
minuit_dict, h_dict = minuit_dictionary(keys, h_base=1e-3)

chi2dict = {key: {chikey: [] for chikey in chi_weights} for key in keys}
dchi2dict = {key: {chikey: [] for chikey in chi_weights} for key in keys}
valuesdict = {key: np.linspace(minuit_dict['limit_{0}'.format(key)][0],
                         minuit_dict['limit_{0}'.format(key)][1],
                         25) for key in keys}

for key in sorted(keys):
    values = valuesdict[key]
    h = h_dict[key]
    print(key, len(values))
    for value_i in range(len(values)):
        value = values[value_i]
        p = p_base.copy()
        
        # calculate the derivative using centered 2 point stencil
        p.update({key: value - h})
        chi2dict_m = FP_func(**p)
        p.update({key: value + h})
        chi2dict_p = FP_func(**p)
        
        for chi2dict_key in sorted(chi2dict[key]):
            dchi2dict[key][chi2dict_key].append(0.5 / h * (chi2dict_p[chi2dict_key] - chi2dict_m[chi2dict_key]))
            chi2dict[key][chi2dict_key].append(chi2dict_p[chi2dict_key])
            chi2dict[key][chi2dict_key].append(chi2dict_m[chi2dict_key])

            
    fig = plt.figure(figsize=(12, 24))
    ax = fig.add_subplot(211)
    dax = fig.add_subplot(212)
    ax.set_title(key)
    
    values = valuesdict[key]
    dchi2 = dchi2dict[key]
    chi2 = chi2dict[key]
    
    for chi2dict_key in sorted(chi2):
        y = []
        dy = []
        x = []
        dx = []
        for i in xrange(len(dchi2[chi2dict_key])):
            x.append(2 * [values[i]])
            y.append([chi2[chi2dict_key][2 * i].sum(), chi2[chi2dict_key][2 * i + 1].sum()])
            
            dx.append(values[i])
            dy.append(dchi2[chi2dict_key][i].sum())
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        dx = np.array(dx).flatten()
        dy = np.array(dy).flatten()
        
        ax.semilogy(x, y, label=chi2dict_key, linewidth=3)
        ax.set_ylabel('$\chi^{2}$')
        dax.plot(dx, dy, label=chi2dict_key, linewidth=3)
        dax.set_yscale('symlog')
        dax.set_xlabel(key)
        dax.set_ylabel('$d \chi^{2} / d p$')
        
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output_directory + key + '.pdf')
    plt.close('all')

# <markdowncell>

# vary the step size for the derivative

# <codecell>

keys = ['dz', 'rzero',
        'dx', 'dy', 'xt', 'yt',
        'z04x', 'z04y',
        'z05d', 'z05x', 'z05y',
        'z06d', 'z06x', 'z06y',
        'z07d', 'z07x', 'z07y',
        'z08d', 'z08x', 'z08y',
        'z09d', 'z09x', 'z09y',
        'z10d', 'z10x', 'z10y',
        'z11d', 'z11x', 'z11y',
        ]
output_directory = '/Users/cpd/Desktop/derivatives/'
minuit_dict, h_dict = minuit_dictionary(keys, h_base=1)

h_mod = {'1':1e-1, '2':1e-2, '6':1e-6, '4':1e-4, '8':1e-8}
dchi2dict = {key: {chi2dict_key: {h_key: [] for h_key in h_mod} for chi2dict_key in chi_weights}for key in keys}
valuesdict = {key: np.linspace(minuit_dict['limit_{0}'.format(key)][0],
                         minuit_dict['limit_{0}'.format(key)][1],
                         30) for key in keys}

for key in sorted(keys):
    values = valuesdict[key]
    print(key, len(values))
    for value_i in range(len(values)):
        value = values[value_i]
        p = p_base.copy()
        
        for h_key in h_mod:
            h = h_dict[key] * h_mod[h_key]
            # calculate the derivative using centered 2 point stencil
            p.update({key: value - h})
            chi2dict_m = FP_func(**p)
            p.update({key: value + h})
            chi2dict_p = FP_func(**p)
        
            for chi2dict_key in sorted(chi2dict_p):
                fprime = 0
                fprime += 0.5 / h * np.sum(chi2dict_p[chi2dict_key] - chi2dict_m[chi2dict_key])
                dchi2dict[key][chi2dict_key][h_key].append(fprime)
    
    values = valuesdict[key]
    dchi2 = dchi2dict[key]
    
    dx = values
    for chi2dict_key in sorted(dchi2):
        fig = plt.figure(figsize=(12, 12))
        dax = fig.add_subplot(111)
        dax.set_title(key)
        for h_key in h_mod:
            dy = dchi2[chi2dict_key][h_key]
            dax.plot(dx, dy, label=chi2dict_key + '_' + h_key, linewidth=3)
            dax.set_yscale('symlog')
            dax.set_xlabel(key)
            dax.set_ylabel('$d \chi^{2} / d p$')
        
        dax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(output_directory + key + '_' + chi2dict_key + '_stepsize.pdf')
        plt.close('all')

# <codecell>

# scan for average_types
average_types = ['scalar_whisker', 'vector_whisker', 'fwhm', 'e1', 'e2', 'e0']

