# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# This is an attempt to show some distinctive-looking plots by construction

# <codecell>

%connect_info

# <codecell>

from focal_plane_shell import FocalPlaneShell
from routines_plot import focal_plane_plot, focal_graph, data_focal_plot, data_hist_plot
from focal_plane_routines import second_moment_to_ellipticity, ellipticity_to_whisker, third_moments_to_octupoles, MAD, mean_trim, average_dictionary, convert_moments

path_mesh = '/Users/cpd/Desktop/Meshes/'
FPS = FocalPlaneShell(path_mesh,)
boxdiv = 2
max_samples_box = 2
FPcoords = FPS.random_coordinates(max_samples_box=max_samples_box, boxdiv=boxdiv)
coords = FPcoords

# <codecell>

def graphs(p, max_samples_box=5, boxdiv=1, coords=coords, scales=None, average=mean_trim, subav=False):

    poles = FPS.plane_averaged(p, coords=coords, average=average, boxdiv=boxdiv, subav=subav)
        
    figures, axes, scales = data_focal_plot(poles, boxdiv=boxdiv, scales=scales)
    #edges = FPS.decaminfo.getEdges(boxdiv=boxdiv)
    #figs, axs, scales2 = data_hist_plot(poles, edges)
    return scales

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'xt': 200}
scales = graphs(p, boxdiv=boxdiv, coords=coords, max_samples_box=max_samples_box, average=np.mean)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'xt': 200}
scales = graphs(p, boxdiv=boxdiv, coords=coords, max_samples_box=max_samples_box, average=np.mean, subav=True)

# <codecell>

p = {'rzero':0.27}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box, scales=scales)

# <codecell>

p = {'rzero':0.07}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box, scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100}
scales = graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'xt': 500}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'xt': -500}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': -100, 'xt': 500}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 0, 'xt': -500}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'yt': -500}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'yt': 500}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z10d': 1.2}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'dy': -2000}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'xt': 2000}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z05d': 1.0}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z06d': 1.0}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z07d': 1.0}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z08d': 1.0}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z09d': 1.0}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z10d': 1.0}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z11d': 1.0}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z05x': 0.01}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z06x': 0.01}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z07x': 0.005}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z08x': 0.005}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z09x': 0.005}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z10x': 0.005}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>

p = {'rzero':0.14, 'dz': 100, 'z11x': 0.005}
graphs(p, boxdiv=boxdiv, max_samples_box=max_samples_box,) # scales=scales)

# <codecell>


# <codecell>


