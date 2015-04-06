# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Last looked at January 28

# <markdowncell>

# Generate random coordinates

# <codecell>

from routines_plot import focal_graph, focal_plane_plot
focal_figure, focal_axis = focal_graph()
# focal_figure = plt.figure(figsize=(12, 12), dpi=300)
# focal_axis = focal_figure.add_subplot(111,
#                                       aspect='equal')
# focal_axis.set_xlabel('$X$ [mm] (East)')
# focal_axis.set_ylabel('$Y$ [mm] (South)')
# focal_axis.set_xlim(-250, 250)
# focal_axis.set_ylim(-250, 250)
x = np.array([0., 10])
y = np.array([0., 10])
u = np.array([0, 2.0e-2])
v = np.array([2.0e-2, 0])
scale = v[0] / 20
Q = focal_axis.quiver(x, y, u, v, angles='uv', color='r', headlength=0, headwidth=0, pivot='tail',
              scale=scale, scale_units='xy', units='xy', width=4)
qk = focal_axis.quiverkey(Q, 0.1, 0.1, scale, label='{0}'.format(v[0]))

# <codecell>

from routines_plot import focal_plane_plot
x = np.array([0., 10])
y = np.array([0., 10])
u = np.array([0, 2.0e-2])
v = np.array([2.0e-2, 0])
scale = v[0] / 10
plt.clf()
focal_figure = plt.figure(figsize=(12, 12), dpi=300)
focal_axis = focal_figure.add_subplot(111,
                                      aspect='equal')
focal_axis.set_xlabel('$X$ [mm] (East)')
focal_axis.set_ylabel('$Y$ [mm] (South)')
focal_axis.set_xlim(-250, 250)
focal_axis.set_ylim(-250, 250)
focal_figure, focal_axis = focal_plane_plot(x, y, u, v, u[0], v[0], scale=scale, artpatch=2,
                                            focal_figure=focal_figure, focal_axis=focal_axis)

# <codecell>

from focal_plane_shell import FocalPlaneShell
path_mesh = '/Users/cpd/Desktop/Meshes/'
nPixels = 32
nbin = 256
pixelOverSample = 8
scaleFactor = 1
FPS = FocalPlaneShell(path_mesh,
                      #nPixels=nPixels, nbin=nbin, pixelOverSample=pixelOverSample, scaleFactor=scaleFactor,
                      )
coords = FPS.random_coordinates(max_samples_box=5, boxdiv=1)

# <markdowncell>

# try plotting everything!

# <codecell>

from routines_plot import focal_plane_plot
from focal_plane_routines import second_moment_to_ellipticity, ellipticity_to_whisker

moments = FPS.plane({'rzero':0.14, 'dz': 100, 'xt': 250}, coords=coords)
e0, e0prime, e1, e2 = \
    second_moment_to_ellipticity(
        moments['x2'],
        moments['y2'],
        moments['xy'])
x = moments['x']
y = moments['y']

u, v, w, phi = ellipticity_to_whisker(e1, e2)

# <codecell>

focal_plane_plot(x, y, u, v, artpatch=2)

# <markdowncell>

# plot the averages

# <codecell>

from focal_plane_routines import average_dictionary, variance_dictionary, mean_trim
def mean_trim_s(data):
    return mean_trim(data, sigma=3)

av = average_dictionary(dict(x=x, y=y, e1=e1, e2=e2, e0=e0), mean_trim_s, boxdiv=1, subav=False)
u, v, w, phi = ellipticity_to_whisker(av['e1'], av['e2'])

focal_plane_plot(av['x_box'], av['y_box'], av['e1'], av['e2'], artpatch=1, scale=4e-3 / 5)
focal_plane_plot(av['x_box'], av['y_box'], u, v, artpatch=2)

# <markdowncell>

# now look at how to best plot your e0 stuffs

# <codecell>


