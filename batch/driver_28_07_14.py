# the idea is to remake this for each run
###############################################################################
# Imports
###############################################################################

from __future__ import print_function, division
import matplotlib
# the agg is so I can submit for batch jobs.
matplotlib.use('Agg')
from matplotlib.pyplot import close
import numpy as np
from glob import glob
from routines import print_command, convert_dictionary
from routines_files import make_directory, download_desdm_filelist
from subprocess import call
from os import environ, path, makedirs, remove, rmdir, chdir
from matplotlib.mlab import csv2rec

from do_PsfCat_and_validation import mkSelPsfCat
from do_fit import do_fit

###############################################################################
# Argparsing
###############################################################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--driver', dest='driver', default='null',
                    help='What run are we doing?')
parser.add_argument('-e', '--execute', dest='execute', default=False,
                    action='store_true',
                    help='Do we _really_ want to do the run?')
parser.add_argument('-v', '--verbose', dest='verbose', default=False,
                    action='store_true',
                    help='Do we want a lot of words while doing the run?')
parser.add_argument('--expid', dest='expid', default=0, type=int,
                    help='Expid we care about.')
args = vars(parser.parse_known_args()[0])

execute = args['execute']
verbose = args['verbose']
expid = args['expid']
driver = args['driver']


shell = environ['SHELL']
kils_shell = '/afs/slac.stanford.edu/u/ki/cpd/.local/bin/zsh'
cpd_shell = '/bin/zsh'

###############################################################################
# Specify Params
###############################################################################
code_name = "driver_28_07_14"
fitname = "noe0_28_07_14"

if shell == cpd_shell:
    # for my computer
    dataDirectory = "/Users/cpd/Desktop/Images/"
    master_code_path = "/Users/cpd/Dropbox/secret-adventure/batch/" + code_name + ".py"
elif shell == kils_shell:
    # for ki-ls
    master_code_path = "/u/ki/cpd/secret-adventure/batch/" + code_name + ".py"

    base_catalog_directory = '/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/'

tag = 'SVA1_FINALCUT'
db_csv = base_catalog_directory + tag + '/db.csv'
explist = base_catalog_directory + tag + '_filelist.out'
catalog_directory = path.join(base_catalog_directory, tag)
fraction = 0.2
n_samples_box = 100
boxdiv = 4

def catalog_directory_func(expid,
                           catalog_directory=catalog_directory):
    rungroup = int(expid/1000) * 1000
    directory = path.join(catalog_directory,"psfcat/%08d/%08d/" % (rungroup,expid))
    return directory

run_directory = catalog_directory.replace('psfcat', fitname)
if not path.exists(run_directory):
    makedirs(run_directory)
    makedirs(run_directory + 'logs/')
catalog_directory = catalog_directory_func(expid)
fits_directory = catalog_directory.replace('psfcat', fitname)
fits_afterburn_directory = catalog_directory.replace('psfcat', fitname + '_afterburn')


par_names = ['rzero', 'z04d', 'z05d', 'z06d', 'xt', 'yt', 'e1', 'e2']

chi_weights = {'e0': 0,
               'e1': 1.,
               'e2': 1.,
               'delta1': 0,
               'delta2': 0,
               'zeta1': 0,
               'zeta2': 0,}

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
'z11d' :   0.2,  # constant offset
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
if len(ccds) < 60:
    todo_list.append('catalog_make')

# analytic_fit
if not path.exists(fits_directory + 'minuit_results.npy'):
    todo_list.append('analytic_fit')

if verbose:
    print(todo_list)
if not execute:
    raise Exception("All done!")

###############################################################################
# master driver
###############################################################################
if driver == 'run_master':
    expids = np.load('/nfs/slac/g/ki/ki18/cpd/catalogs/sva1-list.npy')['expid']
    for expid in expids:
        if shell == cpd_shell:
            # for my computer
            command = ['python', code_name]
        elif shell == kils_shell:
            # for ki-ls
            command = [
                'bsub',
                '-W', '300',
                '-o', run_directory +
                      '/logs/run_{0:08d}_cpd.log'.format(expid),
                '-R', 'rhel60&&linux64',
                'python', code_name]
        args_input = {'driver': 'run_expid',
                      'expid': expid,
                      'execute': True,
                      'verbose': True}
        for arg_i in args_input:
            command.append('--' + arg_i)
            command.append(str(args_input[arg_i]))
        if verbose:
            print_command(command)
        if execute:
            call(command)

elif driver == 'make_master_filelist':

    expids = [0]  # to set!

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

elif driver == 'run_expid':

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

        parser_transform = {'expid': '--expid',
                            'filelist': '--filelist',
                            'basedir': '--basedir',
                            'fraction': '--fraction',
                            'getIn': '--getIn',
                            'deleteIn': '--deleteIn',
                            'tag': '--tag',
                            'ccd': '--ccd',
                            'download_image': '--downImg',
                            'download_background': '--downBak',
                            'download_catalog': '--downCat',
                            'download_psfcat': '--downPsf',
                            }


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


    ###############################################################################
    # fit catalog analytic
    ###############################################################################
    if 'fit_analytic' in todo_list:
        if not path.exists(fits_directory):
            makedirs(fits_directory)
        db = csv2rec(db_csv)
        db = db[db['expid'] == expid]
        p_init['dx'] = 1.162 * (-db['dordodx']) - 516.954
        p_init['dy'] = 1.476 * (-db['dordody']) + 663.945

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
                      'extension': str(2),
                      'fits_directory': fits_directory,
                      'analytic': str(1),
                      'boxdiv': boxdiv,
                      'methodVal': 50,  # number of nearest neighbors
                      'verbose': str(1),
                      'n_samples_box': n_samples_box,
                      'save_iter': str(50),
                      'p_init': str(p_init),
                      'conds': str(conds),
                      'par_names': str(par_names)}
        do_fit(args_input)


###############################################################################
#
###############################################################################
