# the idea is to remake this for each run
"""
Other new thing: I changed the expid from all of sva1 to some 500 images!

Some words of wisdom and thoughts:

    The goal is to be able to run this code in essentially two swipes:
    1. Run a longform battery of plots and fits via bsub commands. I think a good division right now is one job per expid, but this of course may change.
    2. Run a code that merges all the results.

    It would be great if I could get (2) to fit inside (1) (say by checking that all the jobs submitted to the batch queue have finished).

    also todo is bumpbing up the number of stars used. I think I honestly could go all the way.

    it would be /really/ cool if the plots I made were interactive html files....
"""
###############################################################################
# Imports
###############################################################################

from __future__ import print_function, division
import matplotlib
# the agg is so I can submit for batch jobs.
matplotlib.use('Agg')
import numpy as np
from glob import glob
from subprocess import call
from os import environ, path, makedirs
from matplotlib.mlab import csv2rec

from routines import print_command, convert_dictionary
from do_PsfCat_and_validation import mkSelPsfCat
from do_fit import do_fit

###############################################################################
# Argparsing
###############################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--driver', dest='driver', default='run_master',
                    help='What run are we doing?')
parser.add_argument('-e', '--execute', dest='execute', default=False,
                    action='store_true',
                    help='Do we _really_ want to do the run?')
parser.add_argument('-v', '--verbose', dest='verbose', default=False,
                    action='store_true',
                    help='Do we want a lot of words while doing the run?')
parser.add_argument('--expid', dest='expid', default=0, type=int,
                    help='Expid we care about.')
parser.add_argument('--mesh_name', dest='mesh_name', default='', help='mesh name to use?!')
args = vars(parser.parse_known_args()[0])

execute = args['execute']
verbose = args['verbose']
expid = args['expid']
driver = args['driver']
mesh_name = args['mesh_name']

shell = environ['SHELL']
kils_shell = '/afs/slac.stanford.edu/u/ki/cpd/.local/bin/zsh'
cpd_shell = '/bin/zsh'

###############################################################################
# Specify Params
###############################################################################

daycode = '_08_08_14_multiple_meshes'
code_name = "driver" + daycode
mesh_name_dict = {'Science-20130425s1-v1i2_All': 'intra_r',
                  'Science-20130325s1-v1i2_All': 'extra_r',
                  'Science-20140212s2-v1i2_All': 'extra_i',
                  '': 'empty',}
mesh_name_type = mesh_name_dict[mesh_name]
fitname = 'fits' + daycode + '_' + mesh_name_type

if shell == cpd_shell:
    # for my computer
    dataDirectory = "/Users/cpd/Desktop/Images/"
    master_code_path = "/Users/cpd/Dropbox/secret-adventure/batch/" + code_name + ".py"
elif shell == kils_shell:
    # for ki-ls
    master_code_path = "/u/ki/cpd/secret-adventure/batch/" + code_name + ".py"

    base_catalog_directory = '/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/'
    expids = np.load('/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/SVA1_FINALCUT/sva1-list_best.npy')

    #expids = np.load('/nfs/slac/g/ki/ki18/cpd/catalogs/sva1-list.npy')['expid']

tag = 'SVA1_FINALCUT'
db_csv = base_catalog_directory + tag + '/db.csv'
explist = base_catalog_directory + tag + '_filelist.out'
catalog_directory = path.join(base_catalog_directory, tag)
fraction = 0.2
n_samples_box = 1000
boxdiv = -1

def catalog_directory_func(expid,
                           catalog_directory=catalog_directory,
                           use_rungroup=True):
    rungroup = int(expid/1000) * 1000
    if use_rungroup:
        directory = path.join(catalog_directory,"psfcat/%08d/%08d/" % (rungroup,expid))
    else:
        directory = path.join(catalog_directory,"psfcat/%08d/" % (expid))
    return directory

run_directory = path.join(catalog_directory,fitname + '/')
catalog_directory = catalog_directory_func(expid)
fits_directory = catalog_directory.replace('psfcat', fitname)
fits_afterburn_directory = catalog_directory.replace('psfcat', fitname + '_afterburn')

mesh_names = ['Science-20130425s1-v1i2_All', 'Science-20130325s1-v1i2_All', 'Science-20140212s2-v1i2_All']

par_names = ['rzero',
             'z04d',
             'z05d', 'z06d',
             'xt', 'yt',
             'dx', 'dy',
             'e1', 'e2',
             #'delta1', 'delta2',
             #'zeta1', 'zeta2',
             ]

chi_weights = {'e0': 0.1,
               'e1': 1.,
               'e2': 1.,
               #'delta1': 0.1,
               #'delta2': 0.1,
               #'zeta1': 0.01,
               #'zeta2': 0.01,
               }

p_init = {
'delta1' : 0,
'delta2' : 0,
'dx' :     0,
'dy' :     0,
'dz' :     0,
'e1' :     0,
'e2' :     0,
'rzero' :  0.14,
'xt' :     0,
'yt' :     0,
'z04d' :   0,
'z04x' :   0,
'z04y' :   0,
'z05d' :   0,
'z05x' :   0,
'z05y' :   0,
'z06d' :   0,
'z06x' :   0,
'z06y' :   0,
'z07d' :   0,
'z07x' :   0,
'z07y' :   0,
'z08d' :   0,
'z08x' :   0,
'z08y' :   0,
'z09d' :   0,
'z09x' :   0,
'z09y' :   0,
'z10d' :   0,
'z10x' :   0,
'z10y' :   0,
'z11d' :   0,
'z11x' :   0,
'z11y' :   0,
'zeta1' :  0,
'zeta2' :  0,
}



###############################################################################
# Check status of run
###############################################################################

todo_list = []
# conditions for things to append to the todo_list

# catalog filelists
filelists = glob(catalog_directory + '/filelist*')
if len(filelists) < 60:
    todo_list.append('catalog_filelist')

# catalog_make
ccd_filelists = glob(catalog_directory + '/filelist*')
ccds = [int(ccd_filelists_i.split('_')[-1].split('.')[0])
        for ccd_filelists_i in ccd_filelists]
ccds = [ccd for ccd in ccds if ccd < 63]
ccds.sort()

ccds = [ccd for ccd in ccds if not
        path.exists(catalog_directory +
                    "DECam_%08d_%02d_selpsfcat.fits" % (expid, ccd))]
if len(ccds) > 60:
    todo_list.append('catalog_make')

# analytic_fit
if not path.exists(fits_directory + 'minuit_results.npy'):
    todo_list.append('fit_analytic')

if verbose:
    print(todo_list)
    print("expid", expid)
    print("driver", driver)
    print("verbose", verbose)
    print("execute", execute)
if not execute:
    raise Exception("All done!")

###############################################################################
# master driver
###############################################################################
if driver == 'run_master':
    jobnames = []
    run_directory_base = run_directory
    for expid in expids:
        for mesh_name_i, mesh_name in enumerate(mesh_names):
            run_directory = run_directory_base.replace('empty', mesh_name_dict[mesh_name])

            # check if results are there, else skip
            catalog_directory = catalog_directory_func(expid, use_rungroup=True)
            fits_directory = catalog_directory.replace('psfcat', fitname)
            if path.exists(fits_directory + 'minuit_results.npy'):
                continue



            if not path.exists(run_directory):
                makedirs(run_directory)
                makedirs(run_directory + 'logs/')
            jobname = '{0:08d}_{1}'.format(expid, mesh_name_i)
            jobnames.append(jobname)
            args_input = {'driver': 'run_expid',
                          'expid': expid,
                          'mesh_name': mesh_name}
            if shell == cpd_shell:
                # for my computer
                command = ['python', master_code_path]
            elif shell == kils_shell:
                # for ki-ls
                command = [
                    'bsub',
                    #'-W', '120',
                    '-q', 'long',
                    '-o', run_directory +
                          'logs/{0}.log'.format(jobname),
                    '-R', 'rhel60&&linux64',
                    '-J', jobname,
                    'python', master_code_path]
            for arg_i in args_input:
                command.append('--' + arg_i)
                command.append(str(args_input[arg_i]))
            if verbose:
                command.append("--verbose")
            command.append("--execute")

            if verbose:
                print_command(command)
            if execute:
                call(command)

if (driver == 'make_master_filelist') + (driver == 'make_all'):

    username = "cpd"
    password = "cpd70chips"

    outname = catalog_directory + '_filelist.out'

    cmd = 'trivialAccess -u %s -p %s -d dessci -c "select f.path from filepath f, image i, exposure e, runtag t where f.id = i.id and i.run = t.run and t.tag = \'%s\' and i.imagetype = \'red\' and i.exposureid = e.id' % (username, password, tag)
    cmd += ' and (e.expnum = %d' % (expids[0])
    for expid in expids[1:]:
        cmd += ' or e.expnum = %d' % (expid)
    cmd += ')'
    cmd = cmd + '" > %s' % (outname)

    if not path.exists(outname):
        call(cmd, shell=True)

if (driver == 'run_catalog') + (driver == 'make_all'):

    ###############################################################################
    # make catalog filelists for each ccd
    ###############################################################################
    if 'catalog_filelist' in todo_list:
        lines = [line.rstrip('\r\n') for line in open(explist)]
        junk = lines.pop(0)

        lines_filtered = filter(lambda x: '{0:08d}'.format(expid) in x, lines)
        if len(lines_filtered) < 60:
            # missing some chips?!
            print('Expid', expid, 'has only', len(lines_filtered), 'entries!')
            raise Exception('Too few entries!')

        ccd_range = range(1, 63)
        for ccd in ccd_range:
            filelist = catalog_directory + 'filelist_%d_%d.out' % (expid, ccd)
            if path.exists(filelist):
                continue
            lines_filtered_ccd = filter(lambda x:
                    '{0:08d}_{1:02d}'.format(expid, ccd) in x, lines_filtered)

            if len(lines_filtered_ccd) > 0:
                f = open(filelist, 'w')
                f.write('junk\n')
                for line in lines_filtered_ccd:
                    f.write(line + '\r\n')
                f.close()

    ###############################################################################
    # make catalog for each ccd
    ###############################################################################

    if 'catalog_make' in todo_list:

        ccd_filelists = glob(catalog_directory + '/filelist*')
        ccds = [int(ccd_filelists_i.split('_')[-1].split('.')[0])
                for ccd_filelists_i in ccd_filelists]
        ccds = [ccd for ccd in ccds if ccd < 63]
        ccds.sort()

        ccds = [ccd for ccd in ccds if not
                path.exists(catalog_directory +
                            "/DECam_%08d_%02d_selpsfcat.fits" % (expid, ccd))]

        for ccd in ccds:
            args_input = {'expid': expid,
                          'ccd': ccd,
                          'fraction': fraction,
                          'base_catalog_directory': base_catalog_directory,
                          'getIn': True,
                          'deleteIn': True,
                          'tag': tag,
                          'download_psfcat': True,
                          'download_catalog': True,
                          'download_background': False,
                          'download_image': False,
                          'filelist': None,
                          }
            mkSelPsfCat(**args_input)


if (driver == 'run_expid') + (driver == 'make_all'):
    ###############################################################################
    # fit catalog analytic
    ###############################################################################
    if 'fit_analytic' in todo_list:
        if not path.exists(fits_directory):
            makedirs(fits_directory)
        # db = csv2rec(db_csv)
        # db = db[db['expid'] == expid]
        # p_init['dx'] = 1.162 * (-db['dordodx'][0]) - 516.954
        # p_init['dy'] = 1.476 * (-db['dordody'][0]) + 663.945

        conds = (#'"' +
            "((recdata['X' + self.coord_name] > pixel_border) *" +
            " (recdata['X' + self.coord_name] < 2048 - pixel_border) *" +
            " (recdata['Y' + self.coord_name] > pixel_border) *" +
            " (recdata['Y' + self.coord_name] < 4096 - pixel_border)" +
            ") * " +
            "(recdata['SNR_WIN'] > 40)"
            #+ '"'
            )

        args_input = {'expid': expid,
                      'catalogs': catalog_directory,
                      'name': 'selpsfcat',
                      'extension': 2,
                      'fits_directory': fits_directory,
                      'analytic': 1,
                      'boxdiv': boxdiv,
                      'methodVal': 50,  # number of nearest neighbors
                      'verbose': 1,
                      'n_samples_box': n_samples_box,
                      'save_iter': 50,
                      'p_init': p_init,
                      'conds': conds,
                      'chi_weights': chi_weights,
                      'par_names': par_names,
                      'mesh_name': mesh_name}
        if verbose:
            print(args_input)
        if execute:
            do_fit(args_input)

###############################################################################
# merge results
###############################################################################

if driver == 'merge_results':

    results_names = ['coords_sample.npy', 'minuit_results.npy', 'plane_compare.npy', 'plane_fit.npy']

    db = np.recfromcsv('/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/SVA1_FINALCUT/db.csv')

    fitname_base = fitname
    run_directory_base = run_directory


    ## fitname_list = ['fits', 'fits_19.06.14', 'fits_19.06.14_float_hexapod',
    ##                 'fits_19.06.14_float_hexapod_noe0']
    ## use_rungroup = False


    ## fitname_list = ['all_moments_07_08_14_marchmesh', 'all_moments_28_07_14',
    ##                 'all_moments_06_08_14']
    ## use_rungroup = True


    fitname_list = ['fits' + daycode + '_' + mesh_name_dict[mesh_name_i]
                    for mesh_name_i in mesh_name_dict.keys()
                    if len(mesh_name_i) > 0]
    use_rungroup = True


    for fitname in fitname_list:
        run_directory = run_directory_base.replace(fitname_base, fitname)

        fits_dict = {'expid': []}
        for expid in expids:
            catalog_directory = catalog_directory_func(expid, use_rungroup=use_rungroup)
            fits_directory = catalog_directory.replace('psfcat', fitname)
            if path.exists(fits_directory + 'minuit_results.npy'):
                fits_dict['expid'].append(expid)
                minuit_results_i = np.load(fits_directory + 'minuit_results.npy').item()

                # deal with minuit_results_i
                minuit_results_keys = ['args_dict', 'mnstat', 'status']
                for key_min in minuit_results_keys:
                    for key_i in minuit_results_i[key_min]:
                        key = key_min + '_' + key_i
                        if fitname == 'fits_19.06.14':
                            if 'par_names' in key:
                                continue
                        if key not in fits_dict.keys():
                            fits_dict.update({key: []})
                        fits_dict[key].append(str(
                            minuit_results_i[key_min][key_i]))

                # now put in the fits using 'minuit'
                for key_i in minuit_results_i['minuit']:
                    key = 'fit_' + key_i
                    if key not in fits_dict.keys():
                        fits_dict.update({key: []})
                    fits_dict[key].append(minuit_results_i['minuit'][key_i])


                # now do covariance matrix
                key = 'args_covariance'
                if key not in fits_dict.keys():
                    fits_dict.update({key: []})
                fits_dict[key].append(minuit_results_i['covariance'])


                db_i = db[db['expid'] == expid]
                for key_i in db_i.dtype.names:
                    key = 'db_' + key_i
                    if key not in fits_dict.keys():
                        fits_dict.update({key: []})
                    # extra [0] is to convert to number
                    entry = db_i[key_i][0]
                    if entry <= -9998:
                        entry = np.nan
                    fits_dict[key].append(entry)


                key = 'path_plane_fit'
                entry = fits_directory + 'plane_fit.npy'
                if key not in fits_dict.keys():
                    fits_dict.update({key: []})
                fits_dict[key].append(entry)

                key = 'path_plane_compare'
                entry = fits_directory + 'plane_compare.npy'
                if key not in fits_dict.keys():
                    fits_dict.update({key: []})
                fits_dict[key].append(entry)

                # the things that take forever:
                ## if path.exists(fits_directory + 'coords_sample.npy'):
                ##     coords_sample_i = np.load(fits_directory + 'coords_sample.npy')

                ##     fits_dict['expid'].append(expid)
                ##     key = 'coords'
                ##     if key not in fits_dict.keys():
                ##         fits_dict.update({key: []})
                ##     fits_dict[key].append(coords_sample_i)

                ## plane_fit_i = np.load(fits_directory + 'plane_fit.npy').item()

                ## for key_i in plane_fit_i.keys():
                ##     key = 'fit_' + key_i
                ##     if key not in fits_dict.keys():
                ##         fits_dict.update({key: []})
                ##     fits_dict[key].append(plane_fit_i[key_i])

                ## plane_compare_i = np.load(fits_directory + 'plane_compare.npy').item()
                ## for key_i in plane_compare_i.keys():
                ##     key = 'data_' + key_i
                ##     if key not in fits_dict.keys():
                ##         fits_dict.update({key: []})
                ##     fits_dict[key].append(plane_compare_i[key_i])

                # for plane_compare, add in mean, median, std, and MAD of
                # parameters
                var_names = ['delta1', 'delta2',
                             'zeta1', 'zeta2',
                             'e0', 'e1', 'e2',
                             'flux_radius', 'a4',
                             'fwhm_adaptive']
                avgs = [np.mean, np.median]
                avg_names = ['mean', 'median']
                std_names = ['std', 'mad']
                plane_compare_i = np.load(fits_directory + 'plane_compare.npy').item()
                for key_i in range(len(var_names)):
                    var_name = var_names[key_i]
                    data = plane_compare_i[var_name]
                    for avg_i in range(len(avgs)):
                        avg = avgs[avg_i]
                        avg_name = avg_names[avg_i]
                        std_name = std_names[avg_i]

                        data_mod = avg(data)
                        key = 'data_' + var_name + '_' + avg_name
                        if key not in fits_dict.keys():
                            fits_dict.update({key: []})
                        fits_dict[key].append(data_mod)

                        data_mod = np.sqrt(avg(np.square(data - avg(data))))
                        key = 'data_' + var_name + '_' + std_name
                        if key not in fits_dict.keys():
                            fits_dict.update({key: []})
                        fits_dict[key].append(data_mod)

        if verbose:
            print(fitname)
            keys = fits_dict.keys()
            if len(keys) == 0:
                raise Exception("No keys = failed to run!")
            key = keys[-1]
            print(len(fits_dict[key]))
        # convert dictionary
        fits_rec = convert_dictionary(fits_dict)
        # save results
        np.save(run_directory + 'results', fits_rec)


