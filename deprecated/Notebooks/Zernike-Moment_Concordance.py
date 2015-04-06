# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# repeat the below random sampling from the focal plane instead of a uniform distribution.
# 
# try modelling the non-thetas as a line with z dependent dispersion about the line (so higher abs values have more dispersion or somesuch)? or else coming up with some sort of empirical distributions...

# <codecell>

from wavefront import Wavefront
from focal_plane_routines import convert_moments

wf = Wavefront(nPixels=32)
rzero = 0.14
coord = [0, 0]
def zernike_moment(zernikes):

    stamp = wf.stamp(zernikes, rzero=rzero, coord=coord)
    moments = wf.moments(stamp, background=wf.background, threshold=1e-9)
    poles = convert_moments(moments)

    return poles

# <codecell>

# get some nominal values
from focal_plane_shell import FocalPlaneShell

path_mesh = '/Users/cpd/Desktop/Meshes/'

FPS = FocalPlaneShell(
                path_mesh=path_mesh,
                nPixels=16,
                )
FPS.zernikes([[0, 0, 28]], {})

# <codecell>

titles = ['e0', 'e_mag', 'e_theta', 'e_both', 'delta_mag', 'delta_theta', 'delta_both', 'zeta_mag', 'zeta_theta', 'zeta_both',
          'a4', 'fwhm', 'w', 'x2', 'x2y', 'x3', 'xy', 'xy2', 'y2', 'y3']

N = 201
z4vals = [-0.5, -0.1, 0, 0.1, 0.5]
for z4val in z4vals:
    for i in range(3, 11):
        zrange = linspace(-2., 2., N)
        poles = []
        for zi in zrange:
            zernikes = [0] * 11
            zernikes[3] = z4val
            zernikes[i] = zi
            poles_i = zernike_moment(zernikes)
            poles.append(poles_i)
        
        for title in titles:
            fig = plt.figure(figsize=(12, 12), dpi=300)
            ax = fig.add_subplot(111)
            ax.set_xlim(zrange[0], zrange[-1])
            
            if '_both' in title:
                name = title.split('_')[0]
                _ = ax.scatter(zrange, [poles_i['{0}1'.format(name)] for poles_i in poles], c='b')
                _ = ax.scatter(zrange, [poles_i['{0}2'.format(name)] for poles_i in poles], c='r')
            elif '_mag' in title:
                name = title.split('_')[0]
                _ = ax.scatter(zrange, [np.sqrt(poles_i['{0}1'.format(name)] ** 2 +
                                                poles_i['{0}2'.format(name)] ** 2)
                                        for poles_i in poles])
            elif '_theta' in title:
                name = title.split('_')[0]
                _ = ax.scatter(zrange, [np.arctan2(poles_i['{0}2'.format(name)],
                                                   poles_i['{0}1'.format(name)])
                                        for poles_i in poles])            
            else:
                _ = ax.scatter(zrange, [poles_i[title] for poles_i in poles])
        
            fig.savefig('/Users/cpd/Desktop/coefficients/zernikes_{1}_z{0:02d}_z04_eq_{2}.pdf'.format(i + 1, title, z4val))
            plt.clf()

# <codecell>

# show that even if you vary the other zs you get these same relations...

titles = ['e0', 'e_mag', 'e_theta', 'e_both', 'delta_mag', 'delta_theta', 'delta_both', 'zeta_mag', 'zeta_theta', 'zeta_both',
          'a4', 'fwhm', 'w', 'x2', 'x2y', 'x3', 'xy', 'xy2', 'y2', 'y3']

zernikes = np.random.random(size=(3000, 11)) * (1 - -1) + -1
poles = []
for zernike in zernikes:
    poles_i = zernike_moment(zernike)
    poles.append(poles_i)
    
for i in range(3, 11):        
    for title in titles:
        fig = plt.figure(figsize=(12, 12), dpi=300)
        ax = fig.add_subplot(111)
        ax.set_xlim(zernikes[:, i].min(), zernikes[:, i].max())
        
        if '_both' in title:
            name = title.split('_')[0]
            _ = ax.scatter(zernikes[:, i], [poles_i['{0}1'.format(name)] for poles_i in poles], c='b')
            _ = ax.scatter(zernikes[:, i], [poles_i['{0}2'.format(name)] for poles_i in poles], c='r')
        elif '_mag' in title:
            name = title.split('_')[0]
            _ = ax.scatter(zernikes[:, i], [np.sqrt(poles_i['{0}1'.format(name)] ** 2 +
                                            poles_i['{0}2'.format(name)] ** 2)
                                    for poles_i in poles])
        elif '_theta' in title:
            name = title.split('_')[0]
            _ = ax.scatter(zernikes[:, i], [np.arctan2(poles_i['{0}2'.format(name)],
                                               poles_i['{0}1'.format(name)])
                                    for poles_i in poles])            
        else:
            _ = ax.scatter(zernikes[:, i], [poles_i[title] for poles_i in poles])
    
        fig.savefig('/Users/cpd/Desktop/coefficients_random/zernikes_{1}_z{0:02d}_random.pdf'.format(i + 1, title))
        plt.clf()

# <codecell>

# show that even if you vary the other zs you get these same relations...

titles = ['e0', 
          'e_mag', 'e_theta', 'e1', 'e2', 
          'delta_mag', 'delta_theta', 'delta1', 'delta2', 
          'zeta_mag', 'zeta_theta', 'zeta1', 'zeta2',
          'a4', 'fwhm', 'w', 'x2', 'x2y', 'x3', 'xy', 'xy2', 'y2', 'y3']

zernikes = np.random.random(size=(15000, 11)) * (1 - -1) + -1
zernikes[:,3] = 0
zernikes[:,-1] = 0
poles = []
N = 100
for zernike in zernikes:
    poles_i = zernike_moment(zernike)
    poles.append(poles_i)
    
for i in range(3, 11):        
    for title in titles:
        fig = plt.figure(figsize=(12, 12), dpi=300)
        ax = fig.add_subplot(111)
        ax.set_xlim(-1, 1)
        
        if '_mag' in title:
            name = title.split('_')[0]
            _ = ax.hist2d(zernikes[:, i], [np.sqrt(poles_i['{0}1'.format(name)] ** 2 +
                                            poles_i['{0}2'.format(name)] ** 2)
                                    for poles_i in poles],
                          bins=N)            
        elif '_theta' in title:
            name = title.split('_')[0]
            _ = ax.hist2d(zernikes[:, i], [np.arctan2(poles_i['{0}2'.format(name)],
                                               poles_i['{0}1'.format(name)])
                                    for poles_i in poles],
                          bins=N)            
        else:
            _ = ax.hist2d(zernikes[:, i], [poles_i[title] for poles_i in poles],
                          bins=N)            
    
        fig.savefig('/Users/cpd/Desktop/coefficients_random_hist_noz4/zernikes_{1}_z{0:02d}_random.pdf'.format(i + 1, title))
        plt.clf()
        
np.save('/Users/cpd/Desktop/coefficients_random_hist_noz4/zernikes', zernikes)
np.save('/Users/cpd/Desktop/coefficients_random_hist_noz4/poles', poles)

# <codecell>


