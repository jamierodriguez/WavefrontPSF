# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# - Take a contrived example and fit to it

# <codecell>

# make FocalPlaneShell object
from focal_plane_shell import FocalPlaneShell
from focal_plane_routines import minuit_dictionary
from minuit_fit import Minuit_Fit
from routines_plot import data_focal_plot
path_mesh = '/Users/cpd/Desktop/Meshes/'
FPS = FocalPlaneShell(path_mesh)

# make coords
boxdiv = 1
max_samples_box = 5
average = np.mean
edges = FPS.decaminfo.getEdges(boxdiv=boxdiv)
order_dict = {'x2': {'p': 2, 'q': 0},
              'y2': {'p': 0, 'q': 2},
              'xy': {'p': 1, 'q': 1},
              'x3': {'p': 3, 'q': 0},
              'x2y': {'p': 2, 'q': 1},
              'xy2': {'p': 1, 'q': 2},
              'y3': {'p': 0, 'q': 3},
              'x4': {'p': 4, 'q': 0},
              'x2y2': {'p': 2, 'q': 2},
              'y4': {'p': 0, 'q': 4},
              }

coords = FPS.random_coordinates(max_samples_box=max_samples_box, boxdiv=boxdiv)


p = dict(dz=-50,
         e1=0.00,
         e2=0.005, 
         rzero=0.09, 
         dx=251, 
         dy=-25, 
         xt=31, 
         yt=102, 
         z05d=0.014, 
         z06d=-0.05,
         z07x=0.001, 
         z07y=-0.002, 
         z08x=0.0005, 
         z08y=0.0005,
         z09d=0.01,
         z10d=-0.01,
         )


poles = FPS.plane_averaged(p, coords=coords, average=average, boxdiv=boxdiv, order_dict=order_dict)
poles['e1'] += p['e1']
poles['e2'] += p['e2']

# <codecell>

figures, axes, scales = data_focal_plot(poles,
                                        color='r', boxdiv=boxdiv,
                                        )

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

chi2hist = {key:[] for key in chi_weights}
p_hist = []

def FP_func(dz, e1, e2, rzero,
            dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y, z09d, z10d):

    in_dict_FP_func = locals().copy()

    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict_FP_func.keys():
        if np.isnan(in_dict_FP_func[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FPS.remakedonut()
            return 1e20

    poles_i = FPS.plane_averaged(in_dict_FP_func, coords=coords, average=average, boxdiv=boxdiv, order_dict=order_dict)
    poles_i['e1'] += e1
    poles_i['e2'] += e2
    
    p_hist.append(in_dict_FP_func)
    
    chi2 = 0
    for key in chi_weights:
        val_a = poles[key]
        val_b = poles_i[key]
        var = poles['var_{0}'.format(key)]
        weight = chi_weights[key]
        
        chi2_i = np.square(val_a - val_b) / var
        chi2hist[key].append(chi2_i)
        chi2 += np.sum(weight * chi2_i)
    
    if (chi2 < 0) + (np.isnan(chi2)):
        chi2 = 1e20
        
    # update the chi2 by *= 1. / (Nobs - Nparam)
    chi2 *= 1. / (len(poles_i['e1']) -
                  len(in_dict_FP_func.keys()))
    # divide another bit by the sum of the chi_weights
    # not so sure about this one...
    chi2 *= 1. / sum([chi_weights[i] for i in chi_weights])

    
    return chi2
        

# <codecell>

# fit
par_names = p.keys()
verbosity = 3
force_derivatives = 1
strategy = 1
tolerance = 40
h_base = 1e-3
max_iterations = len(par_names) * 1000

# set up initial guesses
minuit_dict, h_dict = minuit_dictionary(par_names, h_base=h_base)
# do fit
minuit_fit = Minuit_Fit(FP_func, minuit_dict, par_names=par_names,
                        h_dict=h_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)
minuit_fit.setupFit()
minuit_fit.doFit()
minuit_results = minuit_fit.outFit()

# <codecell>

for key in sorted(minuit_results['args']):
    print(key, p[key], minuit_results['args'][key], minuit_results['errors'][key])

# <rawcell>

# dx 251 251.180198112 130.144932835
# dy -25 -28.7852813186 120.165382544
# dz -50 -50.1209845486 6.44399278328
# e1 0.0 -9.67387334835e-05 0.00356404500965
# e2 0.005 0.0050258991599 0.00256144274997
# rzero 0.09 0.0900058072802 0.000272781667193
# xt 31 31.1184490162 13.4375804568
# yt 102 101.994507946 13.498173859
# z05d 0.014 0.0145670455889 0.0420458486134
# z06d -0.05 -0.0522203551805 0.0530359530695
# z07x 0.001 0.00100339449531 0.000240946139559
# z07y -0.002 -0.0019972701452 0.000235031142192
# z08x 0.0005 0.000502901795842 0.000225894375275
# z08y 0.0005 0.000498588161359 0.000218755001062
# z09d 0.01 0.00966822114477 0.0205495423132
# z10d -0.01 -0.00934201831138 0.0228103604002

# <codecell>

for key in sorted(minuit_results):
    print(key, minuit_results[key])

# <rawcell>

# args {'z07x': 0.0010033944953142474, 'z06d': -0.05222035518050161, 'z10d': -0.0093420183113810396, 'z09d': 0.0096682211447722821, 'z05d': 0.014567045588919414, 'z07y': -0.0019972701452012718, 'dz': -50.120984548584858, 'dx': 251.18019811233353, 'dy': -28.785281318608213, 'rzero': 0.090005807280232109, 'z08x': 0.00050290179584222723, 'z08y': 0.00049858816135926742, 'e1': -9.6738733483543726e-05, 'yt': 101.99450794583754, 'xt': 31.11844901618133, 'e2': 0.0050258991599011829}
# correlation [[  1.00000000e+00   2.57186144e-01  -1.14592838e-01   1.67664896e-01
#    -5.80094981e-02  -7.11989582e-02   2.26072246e-02   7.52555144e-02
#     1.49837361e-01   1.58598187e-01  -2.60640714e-01   3.10602890e-01
#    -2.17321170e-01  -1.07861083e-01  -7.40384311e-02   7.75789739e-04]
#  [  2.57186144e-01   1.00000000e+00   2.67934440e-01  -9.89475072e-02
#    -4.29545789e-01  -3.24953894e-02   4.30746664e-01   5.56376012e-02
#     2.18668945e-01   3.06191713e-01  -1.37160885e-01   6.71763965e-02
#    -5.09109551e-01   4.26237969e-02  -5.97871777e-02   2.20282114e-01]
#  [ -1.14592838e-01   2.67934440e-01   1.00000000e+00   3.34196763e-01
#    -4.16171176e-03  -3.43435309e-01   2.93775769e-01  -9.16067824e-02
#    -1.56451394e-01  -1.35852842e-01  -3.26262220e-02  -1.05568681e-01
#    -2.25049939e-01   4.90704197e-01   2.04056947e-01   4.81343923e-01]
#  [  1.67664896e-01  -9.89475072e-02   3.34196763e-01   1.00000000e+00
#    -2.05240475e-01  -4.28911357e-01   7.84114522e-02  -4.13175339e-02
#    -1.01154900e-01  -1.23443929e-01   1.14380223e-02   4.11441944e-03
#    -2.76345124e-01  -7.22858346e-02  -2.62668976e-02   3.36697047e-01]
#  [ -5.80094981e-02  -4.29545789e-01  -4.16171176e-03  -2.05240475e-01
#     1.00000000e+00  -4.93981069e-02  -1.53538370e-01   5.18291342e-02
#     1.29158952e-01  -4.53058629e-02  -9.08834662e-03   6.81723199e-02
#     2.16103587e-01   5.74554320e-02  -1.54888278e-01  -8.07733752e-02]
#  [ -7.11989582e-02  -3.24953894e-02  -3.43435309e-01  -4.28911357e-01
#    -4.93981069e-02   1.00000000e+00   1.14858683e-01   8.92699052e-02
#    -1.11108279e-01  -4.20290659e-01   8.08680559e-02   1.42734899e-02
#     9.27198346e-02   1.23281496e-01   1.87515593e-02  -2.44488511e-01]
#  [  2.26072246e-02   4.30746664e-01   2.93775769e-01   7.84114522e-02
#    -1.53538370e-01   1.14858683e-01   1.00000000e+00   8.97444732e-02
#    -8.70516792e-02  -1.28844596e-01   6.11867402e-02   1.70804951e-02
#    -3.33547322e-01   1.90431510e-01  -2.24892987e-01   1.69336049e-01]
#  [  7.52555144e-02   5.56376012e-02  -9.16067824e-02  -4.13175339e-02
#     5.18291342e-02   8.92699052e-02   8.97444732e-02   1.00000000e+00
#     2.91615082e-02  -3.39587048e-02  -3.26248849e-02   7.15682400e-01
#    -3.16050450e-01   4.30616934e-02   8.34986747e-02  -1.35423687e-01]
#  [  1.49837361e-01   2.18668945e-01  -1.56451394e-01  -1.01154900e-01
#     1.29158952e-01  -1.11108279e-01  -8.70516792e-02   2.91615082e-02
#     1.00000000e+00   2.45709151e-01  -2.34520304e-02   8.47871546e-02
#    -7.11126699e-02  -2.64479521e-01  -4.92534379e-01   7.34821595e-02]
#  [  1.58598187e-01   3.06191713e-01  -1.35852842e-01  -1.23443929e-01
#    -4.53058629e-02  -4.20290659e-01  -1.28844596e-01  -3.39587048e-02
#     2.45709151e-01   1.00000000e+00  -2.80276226e-02   8.84612527e-02
#    -6.57896671e-02  -3.36584007e-01  -8.08682469e-02  -2.03136997e-01]
#  [ -2.60640714e-01  -1.37160885e-01  -3.26262220e-02   1.14380223e-02
#    -9.08834662e-03   8.08680559e-02   6.11867402e-02  -3.26248849e-02
#    -2.34520304e-02  -2.80276226e-02   1.00000000e+00  -1.12979190e-01
#     6.95650620e-02   4.10105344e-01   3.54649907e-02   1.10704176e-01]
#  [  3.10602890e-01   6.71763965e-02  -1.05568681e-01   4.11441944e-03
#     6.81723199e-02   1.42734899e-02   1.70804951e-02   7.15682400e-01
#     8.47871546e-02   8.84612527e-02  -1.12979190e-01   1.00000000e+00
#    -2.51117205e-01  -4.90111396e-02   1.06113686e-01  -2.06659596e-01]
#  [ -2.17321170e-01  -5.09109551e-01  -2.25049939e-01  -2.76345124e-01
#     2.16103587e-01   9.27198346e-02  -3.33547322e-01  -3.16050450e-01
#    -7.11126699e-02  -6.57896671e-02   6.95650620e-02  -2.51117205e-01
#     1.00000000e+00  -2.14913747e-01  -1.12505846e-01  -2.32782249e-02]
#  [ -1.07861083e-01   4.26237969e-02   4.90704197e-01  -7.22858346e-02
#     5.74554320e-02   1.23281496e-01   1.90431510e-01   4.30616934e-02
#    -2.64479521e-01  -3.36584007e-01   4.10105344e-01  -4.90111396e-02
#    -2.14913747e-01   1.00000000e+00   3.52302389e-01   2.63534979e-01]
#  [ -7.40384311e-02  -5.97871777e-02   2.04056947e-01  -2.62668976e-02
#    -1.54888278e-01   1.87515593e-02  -2.24892987e-01   8.34986747e-02
#    -4.92534379e-01  -8.08682469e-02   3.54649907e-02   1.06113686e-01
#    -1.12505846e-01   3.52302389e-01   1.00000000e+00   3.40006711e-02]
#  [  7.75789739e-04   2.20282114e-01   4.81343923e-01   3.36697047e-01
#    -8.07733752e-02  -2.44488511e-01   1.69336049e-01  -1.35423687e-01
#     7.34821595e-02  -2.03136997e-01   1.10704176e-01  -2.06659596e-01
#    -2.32782249e-02   2.63534979e-01   3.40006711e-02   1.00000000e+00]]
# covariance [[  4.22389422e-04   3.40638312e-02  -3.06550483e-01   4.14123297e-01
#    -3.25215827e-07  -3.30599015e-07   1.01653802e-07   5.51704450e-06
#     4.15673464e-02   4.38002367e-02  -1.37269962e-05   3.38841639e-04
#    -1.01896129e-04  -9.32550265e-05  -3.57697309e-07   3.84234729e-09]
#  [  3.40638312e-02   4.15316144e+01   2.24753240e+02  -7.66346657e+01
#    -7.55118170e-04  -4.73132249e-05   6.07339029e-04   1.27899676e-03
#     1.90218252e+01   2.65157895e+01  -2.26514718e-03   2.29795159e-02
#    -7.48513755e-02   1.15556000e-02  -9.05731366e-05   3.42109166e-04]
#  [ -3.06550483e-01   2.24753240e+02   1.69424428e+04   5.22782758e+03
#    -1.47766525e-04  -1.00996043e-02   8.36611533e-03  -4.25331242e-02
#    -2.74879987e+02  -2.37617523e+02  -1.08825718e-02  -7.29386761e-01
#    -6.68291669e-01   2.68694480e+00   6.24369022e-03   1.50987007e-02]
#  [  4.14123297e-01  -7.66346657e+01   5.22782758e+03   1.44431528e+04
#    -6.72837305e-03  -1.16458166e-02   2.06172322e-03  -1.77123847e-02
#    -1.64094338e+02  -1.99352856e+02   3.52256230e-03   2.62466773e-02
#    -7.57673230e-01  -3.65456054e-01  -7.42064631e-04   9.75138668e-03]
#  [ -3.25215827e-07  -7.55118170e-04  -1.47766525e-04  -6.72837305e-03
#     7.44101356e-08  -3.04437224e-09  -9.16331961e-09   5.04314719e-08
#     4.75571946e-04  -1.66070379e-04  -6.35298265e-09   9.87094850e-07
#     1.34486025e-06   6.59322809e-07  -9.93199627e-09  -5.30982670e-09]
#  [ -3.30599015e-07  -4.73132249e-05  -1.00996043e-02  -1.16458166e-02
#    -3.04437224e-09   5.10437764e-08   5.67747318e-09   7.19429288e-08
#    -3.38839076e-04  -1.27597704e-03   4.68193242e-08   1.71173424e-07
#     4.77906768e-07   1.17171032e-06   9.95888539e-10  -1.33114671e-08]
#  [  1.01653802e-07   6.07339029e-04   8.36611533e-03   2.06172322e-03
#    -9.16331961e-09   5.67747318e-09   4.78673872e-08   7.00388836e-08
#    -2.57082600e-04  -3.78798099e-04   3.43047230e-08   1.98360449e-07
#    -1.66485515e-06   1.75270825e-06  -1.15663871e-08   8.92822964e-09]
#  [  5.51704450e-06   1.27899676e-03  -4.25331242e-02  -1.77123847e-02
#     5.04314719e-08   7.19429288e-08   7.00388836e-08   1.27239889e-05
#     1.40409759e-03  -1.62773792e-03  -2.98220414e-07   1.35508564e-04
#    -2.57197764e-05   6.46180285e-06   7.00153352e-08  -1.16413231e-07]
#  [  4.15673464e-02   1.90218252e+01  -2.74879987e+02  -1.64094338e+02
#     4.75571946e-04  -3.38839076e-04  -2.57082600e-04   1.40409759e-03
#     1.82201244e+02   4.45675463e+01  -8.11209180e-04   6.07491905e-02
#    -2.18988742e-02  -1.50182274e-01  -1.56283874e-03   2.39030681e-04]
#  [  4.38002367e-02   2.65157895e+01  -2.37617523e+02  -1.99352856e+02
#    -1.66070379e-04  -1.27597704e-03  -3.78798099e-04  -1.62773792e-03
#     4.45675463e+01   1.80569105e+02  -9.65127575e-04   6.30971253e-02
#    -2.01687297e-02  -1.90268177e-01  -2.55447526e-04  -6.57819616e-04]
#  [ -1.37269962e-05  -2.26514718e-03  -1.08825718e-02   3.52256230e-03
#    -6.35298265e-09   4.68193242e-08   3.43047230e-08  -2.98220414e-07
#    -8.11209180e-04  -9.65127575e-04   6.56679530e-06  -1.53677385e-05
#     4.06693330e-06   4.42102658e-05   2.13638017e-08   6.83654712e-08]
#  [  3.38841639e-04   2.29795159e-02  -7.29386761e-01   2.62466773e-02
#     9.87094850e-07   1.71173424e-07   1.98360449e-07   1.35508564e-04
#     6.07491905e-02   6.30971253e-02  -1.53677385e-05   2.81753637e-03
#    -3.04095762e-04  -1.09441119e-04   1.32406137e-06  -2.64354318e-06]
#  [ -1.01896129e-04  -7.48513755e-02  -6.68291669e-01  -7.57673230e-01
#     1.34486025e-06   4.77906768e-07  -1.66485515e-06  -2.57197764e-05
#    -2.18988742e-02  -2.01687297e-02   4.06693330e-06  -3.04095762e-04
#     5.20473075e-04  -2.06259655e-04  -6.03359526e-07  -1.27980871e-07]
#  [ -9.32550265e-05   1.15556000e-02   2.68694480e+00  -3.65456054e-01
#     6.59322809e-07   1.17171032e-06   1.75270825e-06   6.46180285e-06
#    -1.50182274e-01  -1.90268177e-01   4.42102658e-05  -1.09441119e-04
#    -2.06259655e-04   1.76970923e-03   3.48392257e-06   2.67168526e-06]
#  [ -3.57697309e-07  -9.05731366e-05   6.24369022e-03  -7.42064631e-04
#    -9.93199627e-09   9.95888539e-10  -1.15663871e-08   7.00153352e-08
#    -1.56283874e-03  -2.55447526e-04   2.13638017e-08   1.32406137e-06
#    -6.03359526e-07   3.48392257e-06   5.52591115e-08   1.92612961e-09]
#  [  3.84234729e-09   3.42109166e-04   1.50987007e-02   9.75138668e-03
#    -5.30982670e-09  -1.33114671e-08   8.92822964e-09  -1.16413231e-07
#     2.39030681e-04  -6.57819616e-04   6.83654712e-08  -2.64354318e-06
#    -1.27980871e-07   2.67168526e-06   1.92612961e-09   5.80753902e-08]]
# errors {'z07x': 0.000240946139559493, 'z06d': 0.053035953069457353, 'z10d': 0.022810360400174146, 'z09d': 0.02054954231322581, 'z05d': 0.042045848613382952, 'z07y': 0.00023503114219150653, 'dz': 6.4439927832763999, 'dx': 130.14493283503452, 'dy': 120.16538254435181, 'rzero': 0.00027278166719300889, 'z08x': 0.00022589437527486406, 'z08y': 0.0002187550010623227, 'e1': 0.0035640450096547545, 'yt': 13.49817385904953, 'xt': 13.437580456828528, 'e2': 0.0025614427499668253}
# minuit {'error_e2': 0.0025614427499668253, 'error_e1': 0.0035640450096547545, 'error_z07y': 0.00023503114219150653, 'z09d': 0.0096682211447722821, 'error_z05d': 0.042045848613382952, 'error_dy': 120.16538254435181, 'error_dx': 130.14493283503452, 'error_dz': 6.4439927832763999, 'error_z06d': 0.053035953069457353, 'dz': -50.120984548584858, 'error_xt': 13.437580456828528, 'dx': 251.18019811233353, 'dy': -28.785281318608213, 'error_z07x': 0.000240946139559493, 'rzero': 0.090005807280232109, 'error_yt': 13.49817385904953, 'z08y': 0.00049858816135926742, 'e1': -9.6738733483543726e-05, 'z08x': 0.00050290179584222723, 'xt': 31.11844901618133, 'e2': 0.0050258991599011829, 'z05d': 0.014567045588919414, 'error_rzero': 0.00027278166719300889, 'z10d': -0.0093420183113810396, 'z06d': -0.05222035518050161, 'error_z09d': 0.02054954231322581, 'yt': 101.99450794583754, 'error_z08y': 0.0002187550010623227, 'error_z08x': 0.00022589437527486406, 'z07y': -0.0019972701452012718, 'z07x': 0.0010033944953142474, 'error_z10d': 0.022810360400174146}
# mnstat {'nvpar': 16.0, 'nparx': 16.0, 'amin': 0.005321598523976011, 'icstat': 3.0, 'nCallsDerivative': 31.0, 'nCalls': 322.0, 'errdef': 1.0, 'edm': 0.01064835291545894}
# status {'nCalls': 322.0, 'migrad_ierflg': 0, 'GetStatus': 0, 'deltatime': 10849.329426, 'force_derivatives': 1, 'max_iterations': 16000, 'npar': 16, 'verbosity': 3, 'strategy': 1, 'startingtime': 2090.723007, 'h_dict': {'z07x': 5e-07, 'z06d': 0.0001, 'z10d': 0.0001, 'z09d': 0.0001, 'z05d': 0.0001, 'z07y': 5e-07, 'dz': 0.04, 'dx': 0.1, 'dy': 0.1, 'rzero': 5e-06, 'z08x': 5e-07, 'z08y': 5e-07, 'e1': 5e-06, 'yt': 0.1, 'xt': 0.1, 'e2': 5e-06}, 'tolerance': 40, 'nCallsDerivative': 31.0, 'par_names': ['z09d', 'dz', 'dx', 'dy', 'rzero', 'z08x', 'z08y', 'e1', 'yt', 'xt', 'e2', 'z06d', 'z10d', 'z05d', 'z07y', 'z07x']}

# <codecell>

for key in sorted(minuit_dict):
    print(key, minuit_dict[key])

# <rawcell>

# dx 0
# dy 0
# dz 0
# e1 0
# e2 0
# error_dx 100
# error_dy 100
# error_dz 40
# error_e1 0.005
# error_e2 0.005
# error_rzero 0.005
# error_xt 100
# error_yt 100
# error_z05d 0.1
# error_z06d 0.1
# error_z07x 0.0005
# error_z07y 0.0005
# error_z08x 0.0005
# error_z08y 0.0005
# error_z09d 0.1
# error_z10d 0.1
# fix_dx False
# fix_dy False
# fix_dz False
# fix_e1 False
# fix_e2 False
# fix_rzero False
# fix_xt False
# fix_yt False
# fix_z05d False
# fix_z06d False
# fix_z07x False
# fix_z07y False
# fix_z08x False
# fix_z08y False
# fix_z09d False
# fix_z10d False
# limit_dx (-4500, 4500)
# limit_dy (-4500, 4500)
# limit_dz (-300, 300)
# limit_e1 (-0.05, 0.05)
# limit_e2 (-0.05, 0.05)
# limit_rzero (0.07, 0.4)
# limit_xt (-4500, 4500)
# limit_yt (-4500, 4500)
# limit_z05d (-0.75, 0.75)
# limit_z06d (-0.75, 0.75)
# limit_z07x (-0.0075, 0.0075)
# limit_z07y (-0.0075, 0.0075)
# limit_z08x (-0.0075, 0.0075)
# limit_z08y (-0.0075, 0.0075)
# limit_z09d (-0.75, 0.75)
# limit_z10d (-0.75, 0.75)
# rzero 0.125
# xt 0
# yt 0
# z05d 0
# z06d 0
# z07x 0
# z07y 0
# z08x 0
# z08y 0
# z09d 0
# z10d 0

# <codecell>

# compare
figures, axes, scales = data_focal_plot(poles,
                                        color='r', boxdiv=boxdiv,
                                        )
# plot the comparison
poles_i = FPS.plane_averaged(minuit_results['args'], coords=coords, average=average, boxdiv=boxdiv, order_dict=order_dict)
poles_i['e1'] += minuit_results['args']['e1']
poles_i['e2'] += minuit_results['args']['e2']
figures, axes, scales = data_focal_plot(poles_i,
                                        color='b', boxdiv=boxdiv,
                                        figures=figures, axes=axes, scales=scales,
                                        )

# <codecell>

# plot chi2s
plt.figure(figsize=(12,12), dpi=300)
for key in sorted(chi2hist.keys()):
    chis = chi2hist[key]
    x = range(len(chis))
    y = []
    for i in x:
        y.append(np.sum(chis[i]) * chi_weights[key])
    plt.plot(x, np.log10(y), label=key)
plt.legend()

# <markdowncell>

# - do the fit with a different set of coordinates

# <codecell>

coords2 = FPS.random_coordinates(max_samples_box=max_samples_box, boxdiv=boxdiv)
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

chi2hist2 = {key:[] for key in chi_weights}
p_hist2 = []

def FP_func2(dz, e1, e2, rzero,
            dx, dy, xt, yt, z05d, z06d,
            z07x, z07y, z08x, z08y, z09d, z10d):

    in_dict_FP_func = locals().copy()

    # go through the key_FP_funcs and make sure there are no nans
    for key_FP_func in in_dict_FP_func.keys():
        if np.isnan(in_dict_FP_func[key_FP_func]).any():
            # if there is a nan, don't even bother calling, just return a
            # big chi2
            FPS.remakedonut()
            return 1e20

    poles_i = FPS.plane_averaged(in_dict_FP_func, coords=coords2, average=average, boxdiv=boxdiv, order_dict=order_dict)
    poles_i['e1'] += e1
    poles_i['e2'] += e2
    
    p_hist2.append(in_dict_FP_func)
    
    chi2 = 0
    for key in chi_weights:
        val_a = poles[key]
        val_b = poles_i[key]
        var = poles['var_{0}'.format(key)]
        weight = chi_weights[key]
        
        chi2_i = np.square(val_a - val_b) / var
        chi2hist2[key].append(chi2_i)
        chi2 += np.sum(weight * chi2_i)
    
    if (chi2 < 0) + (np.isnan(chi2)):
        chi2 = 1e20
        
    # update the chi2 by *= 1. / (Nobs - Nparam)
    chi2 *= 1. / (len(poles_i['e1']) -
                  len(in_dict_FP_func.keys()))
    # divide another bit by the sum of the chi_weights
    # not so sure about this one...
    chi2 *= 1. / sum([chi_weights[i] for i in chi_weights])

    
    return chi2

# <codecell>

# fit
par_names = p.keys()
verbosity = 3
force_derivatives = 1
strategy = 1
tolerance = 40
h_base = 1e-3
max_iterations = len(par_names) * 1000

# set up initial guesses
minuit_dict, h_dict = minuit_dictionary(par_names, h_base=h_base)
# do fit
minuit_fit = Minuit_Fit(FP_func2, minuit_dict, par_names=par_names,
                        h_dict=h_dict,
                        verbosity=verbosity,
                        force_derivatives=force_derivatives,
                        strategy=strategy, tolerance=tolerance,
                        max_iterations=max_iterations)
minuit_fit.setupFit()
minuit_fit.doFit()
minuit_results = minuit_fit.outFit()

# <codecell>

for key in sorted(minuit_results['args']):
    print(key, p[key], minuit_results['args'][key], minuit_results['errors'][key])
for key in sorted(minuit_dict):
    print(key, minuit_dict[key])
# compare
figures, axes, scales = data_focal_plot(poles,
                                        color='r', boxdiv=boxdiv,
                                        )
# plot the comparison
poles_i = FPS.plane_averaged(minuit_results['args'], coords=coords2, average=average, boxdiv=boxdiv, order_dict=order_dict)
poles_i['e1'] += minuit_results['args']['e1']
poles_i['e2'] += minuit_results['args']['e2']
figures, axes, scales = data_focal_plot(poles_i,
                                        color='b', boxdiv=boxdiv,
                                        figures=figures, axes=axes, scales=scales,
                                        )
# plot chi2s
plt.figure(figsize=(12,12), dpi=300)
for key in sorted(chi2hist2.keys()):
    chis = chi2hist2[key]
    x = range(len(chis))
    y = []
    for i in x:
        y.append(np.sum(chis[i]) * chi_weights[key])
    plt.plot(x, np.log10(y), label=key)
plt.legend()

# <markdowncell>

# - Find the minimal *parameter*

# <markdowncell>

# - Look at effects of including the higher order moments vs excluding

