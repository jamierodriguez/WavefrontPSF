# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# See if what the major differences are from trying different random sample sizes etc

# <codecell>

from focal_plane_shell import FocalPlaneShell
from routines_plot import focal_plane_plot, focal_graph, data_focal_plot, data_hist_plot
from focal_plane_routines import MAD, mean_trim, convert_moments, ellipticity_to_whisker, average_dictionary, variance_dictionary

path_mesh = '/Users/cpd/Desktop/Meshes/'
FPS = FocalPlaneShell(path_mesh,)

# <codecell>

# generate the focal planes near focus
boxdiv = 1
edges = FPS.decaminfo.getEdges(boxdiv=boxdiv)
average = np.mean

N = 5
poles_dict = {}
poles_list = []
for i in range(20):
    FPcoords = FPS.random_coordinates(max_samples_box=N, boxdiv=boxdiv)
    coords = FPcoords
    p = {'rzero':0.14}
    poles = FPS.plane_averaged(p, coords=coords, average=average, boxdiv=boxdiv)
    poles_list.append(poles)
    if i == 0:
        # make the entries
        for key in poles:
            poles_dict.update({key: poles[key]})
    else:
        for key in poles:
            poles_dict[key] = np.append(poles_dict[key], poles[key])

# <codecell>

# use the multiple iterations to produce a plot of the distribution at each box
poles_dict_to_average = {}
for key in poles_dict:
    if 'var' not in key:
        poles_dict_to_average.update({key: poles_dict[key]})
poles_averaged = average_dictionary(poles_dict_to_average, average, boxdiv=boxdiv)
figures, axes, scales = data_focal_plot(poles_averaged, color='r')

# <codecell>

# subtract first and second iteration
ii = 0
jj = 1
pole_i = poles_list[ii]
# for key in pole_i.keys():
#     pole_i.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_i, color='r')
pole_j = poles_list[jj]
# for key in pole_i.keys():
#     pole_j.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_j, color='b',
                                        figures=figures, axes=axes, scales=scales)

subs = {}
for key in pole_i:
    if 'var' not in key:
        subs.update({key: pole_i[key] - pole_j[key]})
    else:
        subs.update({key: pole_i[key] + pole_j[key]})
for key in ['x_box', 'y_box']:
    subs.update({key: pole_i[key]})
figures, axes, scales = data_focal_plot(subs, color='k',
                                        figures=figures, axes=axes, scales=scales)
figures, axes, scales = data_focal_plot(subs, color='k', scales=scales)

# <markdowncell>

# ### try changing the number

# <codecell>

# generate the focal planes near focus
boxdiv = 1
edges = FPS.decaminfo.getEdges(boxdiv=boxdiv)
average = np.mean

N = 10
poles_dict = {}
poles_list = []
i=0
for N in range(2, 20):
    FPcoords = FPS.random_coordinates(max_samples_box=N, boxdiv=boxdiv)
    coords = FPcoords
    p = {'rzero':0.14}
    poles = FPS.plane_averaged(p, coords=coords, average=average, boxdiv=boxdiv)
    poles_list.append(poles)
    if i == 0:
        # make the entries
        for key in poles:
            poles_dict.update({key: poles[key]})
        i += 1
    else:
        for key in poles:
            poles_dict[key] = np.append(poles_dict[key], poles[key])

# use the multiple iterations to produce a plot of the distribution at each box
poles_dict_to_average = {}
for key in poles_dict:
    if 'var' not in key:
        poles_dict_to_average.update({key: poles_dict[key]})
poles_averaged = average_dictionary(poles_dict_to_average, average, boxdiv=boxdiv)
figures, axes, scales = data_focal_plot(poles_averaged, color='r')

# subtract first and second iteration
ii = 0
jj = 1
pole_i = poles_list[ii]
# for key in pole_i.keys():
#     pole_i.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_i, color='r')
pole_j = poles_list[jj]
# for key in pole_i.keys():
#     pole_j.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_j, color='b',
                                        figures=figures, axes=axes, scales=scales)

subs = {}
for key in pole_i:
    if 'var' not in key:
        subs.update({key: pole_i[key] - pole_j[key]})
    else:
        subs.update({key: pole_i[key] + pole_j[key]})
for key in ['x_box', 'y_box']:
    subs.update({key: pole_i[key]})
figures, axes, scales = data_focal_plot(subs, color='k',
                                        figures=figures, axes=axes, scales=scales)
figures, axes, scales = data_focal_plot(subs, color='k', scales=scales)

# <markdowncell>

# repeat the above but severely out of focus and other such things

# <codecell>

# try changing the number
# generate the focal planes near focus
boxdiv = 1
edges = FPS.decaminfo.getEdges(boxdiv=boxdiv)
average = np.mean

N = 10
poles_dict = {}
poles_list = []
i = 0
for N in range(2, 20):
    FPcoords = FPS.random_coordinates(max_samples_box=N, boxdiv=boxdiv)
    coords = FPcoords
    p = {'rzero':0.14}
    poles = FPS.plane_averaged(p, coords=coords, average=average, boxdiv=boxdiv)
    poles_list.append(poles)
    if i == 0:
        # make the entries
        for key in poles:
            poles_dict.update({key: poles[key]})
        i += 1
    else:
        for key in poles:
            poles_dict[key] = np.append(poles_dict[key], poles[key])

# use the multiple iterations to produce a plot of the distribution at each box
poles_dict_to_average = {}
for key in poles_dict:
    if 'var' not in key:
        poles_dict_to_average.update({key: poles_dict[key]})
poles_averaged = average_dictionary(poles_dict_to_average, average, boxdiv=boxdiv)
figures, axes, scales = data_focal_plot(poles_averaged, color='r')

# subtract first and second iteration
ii = 0
jj = 1
pole_i = poles_list[ii]
# for key in pole_i.keys():
#     pole_i.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_i, color='r')
pole_j = poles_list[jj]
# for key in pole_i.keys():
#     pole_j.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_j, color='b',
                                        figures=figures, axes=axes, scales=scales)

subs = {}
for key in pole_i:
    if 'var' not in key:
        subs.update({key: pole_i[key] - pole_j[key]})
    else:
        subs.update({key: pole_i[key] + pole_j[key]})
for key in ['x_box', 'y_box']:
    subs.update({key: pole_i[key]})
figures, axes, scales = data_focal_plot(subs, color='k',
                                        figures=figures, axes=axes, scales=scales)
figures, axes, scales = data_focal_plot(subs, color='k', scales=scales)

# <markdowncell>

# also see if there are differences from using 32 vs 16 pixels

# <codecell>

nPixels = 32
nbin = 256
pixelOverSample = 8
scaleFactor = 1
FPS2 = FocalPlaneShell(path_mesh,
                      nPixels=nPixels, nbin=nbin, pixelOverSample=pixelOverSample, scaleFactor=scaleFactor,
                      )
nPixels = 16
nbin = 64
pixelOverSample = 4
scaleFactor = 2.
FPS1 = FocalPlaneShell(path_mesh,
                      nPixels=nPixels, nbin=nbin, pixelOverSample=pixelOverSample, scaleFactor=scaleFactor,
                      )

# <codecell>

# try changing the number
# generate the focal planes near focus
boxdiv = 1
edges = FPS.decaminfo.getEdges(boxdiv=boxdiv)
average = np.mean

N = 20
poles_dict = {}
poles_list = []
i = 0
FPcoords = FPS.random_coordinates(max_samples_box=N, boxdiv=boxdiv)
for FPS in [FPS1, FPS2]:

    FPcoords = FPS.random_coordinates(max_samples_box=N, boxdiv=boxdiv)
    coords = FPcoords
    p = {'rzero':0.14}
    poles = FPS.plane_averaged(p, coords=coords, average=average, boxdiv=boxdiv)
    poles_list.append(poles)
    if i == 0:
        # make the entries
        for key in poles:
            poles_dict.update({key: poles[key]})
        i += 1
    else:
        for key in poles:
            poles_dict[key] = np.append(poles_dict[key], poles[key])

# use the multiple iterations to produce a plot of the distribution at each box
poles_dict_to_average = {}
for key in poles_dict:
    if 'var' not in key:
        poles_dict_to_average.update({key: poles_dict[key]})
poles_averaged = average_dictionary(poles_dict_to_average, average, boxdiv=boxdiv)
figures, axes, scales = data_focal_plot(poles_averaged, color='r')

# subtract first and second iteration
ii = 0
jj = 1
pole_i = poles_list[ii]
# for key in pole_i.keys():
#     pole_i.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_i, color='r')
pole_j = poles_list[jj]
# for key in pole_i.keys():
#     pole_j.update({'var_{0}'.format(key): []})
figures, axes, scales = data_focal_plot(pole_j, color='b',
                                        figures=figures, axes=axes, scales=scales)

subs = {}
for key in pole_i:
    if 'var' not in key:
        subs.update({key: pole_i[key] - pole_j[key]})
    else:
        subs.update({key: pole_i[key] + pole_j[key]})
for key in ['x_box', 'y_box']:
    subs.update({key: pole_i[key]})
figures, axes, scales = data_focal_plot(subs, color='k',
                                        figures=figures, axes=axes, scales=scales)
figures, axes, scales = data_focal_plot(subs, color='k', scales=scales)

# <codecell>


