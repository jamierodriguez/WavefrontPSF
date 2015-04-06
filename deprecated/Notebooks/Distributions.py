# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# the goal of this notebook is to generate steps through the various parameters and then make a movie out of them
# 
# See the ki-ls version for a more recent edition

# <codecell>

# make FocalPlaneShell object
from focal_plane_shell import FocalPlaneShell
from focal_plane_routines import minuit_dictionary
from minuit_fit import Minuit_Fit
from routines_plot import data_focal_plot, data_hist_plot, collect_images
from subprocess import call
path_mesh = '/Users/cpd/Desktop/Meshes/'
FPS = FocalPlaneShell(path_mesh)

# make coords
boxdiv=1
max_samples_box = 5
average = np.mean
color = 'k'
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
p_base = dict(dz=0, rzero=0.14)
keys = [
        'dx', 'dy', 'xt', 'yt',
#         'z04x', 'z04y',
#         'z05d', 'z05x', 'z05y',
#         'z06d', 'z06x', 'z06y',
#         'z07d', 'z07x', 'z07y',
#         'z08d', 'z08x', 'z08y',
#         'z09d', 'z09x', 'z09y',
#         'z10d', 'z10x', 'z10y',
#         'z11d', 'z11x', 'z11y',
        ]
minuit_dict, h_dict = minuit_dictionary(keys)

output_directory = '/Users/cpd/Desktop/pdf/lowdz_'

# <codecell>

poles = FPS.plane_averaged(p_base, coords=coords, average=average, boxdiv=boxdiv, order_dict=order_dict)
figures, _, scales = data_focal_plot(poles, boxdiv=boxdiv, average=average, color=color)
figures_hist, _, scales_hist = data_hist_plot(poles, edges)
for key in scales_hist:
    scales_hist[key]['vmax'] *= 2
    scales_hist[key]['vmin'] *= 0.5

# <codecell>

file_list = []
for key in keys:
    values = np.linspace(minuit_dict['limit_{0}'.format(key)][0],
                         minuit_dict['limit_{0}'.format(key)][1],
                         100)
    print(key, len(values))
    for value_i in range(len(values)):
        value = values[value_i]
        p = p_base.copy()
        p.update({key: value})
        
        poles = FPS.plane_averaged(p, coords=coords, average=average, boxdiv=boxdiv, order_dict=order_dict)
        figures, axes, scales = data_focal_plot(poles, boxdiv=boxdiv, average=average, color=color, scales=scales)
        figures_hist, axes_hist, scales_hist = data_hist_plot(poles, edges, scales=scales_hist)

        # save the figures
        for key_fig in figures:
            axes[key_fig].set_title('{0}: ({1} = {2:.2e})'.format(key_fig, key, value))
            file_i = output_directory + '{0}_{1}_{2:03d}.pdf'.format(key, key_fig, value_i)
            file_list.append(file_i)
            figures[key_fig].savefig(file_i)

        for key_fig in figures_hist:
            axes_hist[key_fig].set_title('{0}: ({1} = {2:.2e})'.format(key_fig, key, value))
            file_i = output_directory + '{0}_{1}-hist_{2:03d}.pdf'.format(key, key_fig, value_i)
            file_list.append(file_i)
            figures_hist[key_fig].savefig(file_i)
        plt.close('all')

# <codecell>

file_list = []
for key in keys:
    values = np.linspace(minuit_dict['limit_{0}'.format(key)][0],
                         minuit_dict['limit_{0}'.format(key)][1],
                         100)
    print(key, len(values))
    for value_i in range(len(values)):
        value = values[value_i]

        # save the figures
        for key_fig in figures:
            file_i = output_directory + '{0}_{1}_{2:03d}.pdf'.format(key, key_fig, value_i)
            file_list.append(file_i)

        for key_fig in figures_hist:
            file_i = output_directory + '{0}_{1}-hist_{2:03d}.pdf'.format(key, key_fig, value_i)
            file_list.append(file_i)

# <codecell>

# collect images
for i in xrange(len(file_list)):
    file_i = file_list[i]
    file_i = file_i.split('/', 100)[-1]
    file_list[i] = file_i.split('_')
file_list = np.array(file_list)
type_iter = list(np.unique(file_list[:, 0]))
type_param = list(np.unique(file_list[:, 1]))

# <codecell>

for key_iter in type_iter:
    for key_param in type_param:
        # construct the file_list of graphics we merge
        file_merge = [output_directory + '{0}_{1}_{2}'.format(*i)
                     for i in file_list[(file_list[:, 0] == key_iter) * (file_list[:, 1] == key_param)]]
        
        # make the command
        merged_file = output_directory + '{0}_{1}'.format(key_iter, key_param)
        command = ['gs',
                   '-dNOPAUSE', '-dBATCH', '-dSAFER', '-q',
                   '-sDEVICE=pdfwrite',
                   '-sOutputFile={0}.pdf'.format(merged_file)]
        for file_i in file_merge:
            command.append(file_i)
        call(command)
        
        # delete all the files
        command = ['find', output_directory,
                   '-name', '{0}_{1}_*.pdf'.format(key_iter, key_param),
                   '-delete']
        call(command)

# <codecell>

rate = 5
for key_iter in type_iter:
    for key_param in type_param:
        # make the command
        merged_file = output_directory + '{0}_{1}'.format(key_iter, key_param)
        # movies!
        # convert the pdf to jpegs
        command = [#'bsub', '-q', 'short', '-o', path_logs,
                   'gs', '-sDEVICE=jpeg',
                   '-dNOPAUSE', '-dBATCH', #'-dSAFER',
                   '-r300', '-dJPEGQ=100',
                   "-sOutputFile={0}-%000d.jpg".format(merged_file),
                   merged_file + '.pdf']
        call(command)

        # now convert the jpegs to a video
        command = [#'bsub', '-q', 'short', '-o', path_logs,
                   'ffmpeg', '-f', 'image2',
                   '-r', '{0}'.format(rate),
                   '-i', '{0}-%d.jpg'.format(merged_file),
                   '-r', '30', '-an', '-q:v', '0',
                   merged_file + '.mp4']
        call(command)

        # delete the jpegs
        command = [#'bsub', '-q', 'short', '-o', path_logs,
                   'find', output_directory,
                   '-name', '*.jpg', '-delete']
        call(command)

# <codecell>


