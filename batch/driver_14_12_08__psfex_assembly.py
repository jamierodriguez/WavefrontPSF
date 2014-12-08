from glob import glob
import numpy as np
from subprocess import call

"""
afterwards:
http://stackoverflow.com/questions/16890582/unixmerge-multiple-csv-files-with-same-header-by-keeping-the-header-of-the-firs
awk 'FNR==1 && NR!=1{next;}{print}' *.csv
"""

out_dir = '/nfs/slac/g/ki/ki18/des/cpd/big_psfex_rerun'
base_psfex = '/nfs/slac/g/ki/ki21/cosmo/beckermr/DES/meds_psf_sva1/EXTRA/red'
psfs = sorted(glob(base_psfex + '/**/psfex-rerun/v2/**/*psfcat.psf'))
print('{0} (~{1} images) psfs found!'.format(len(psfs), len(psfs) / 60))

psf_expids = []
directories = []
for psf in psfs:
    _, expid_str, ext_str, _ = psf.split('DECam')[-1].split('_')
    expid = int(expid_str)
    directory = psf.split('DECam_{0:08d}_'.format(expid))[0]  # extra _ is vital!
    psf_expids.append(expid)
    directories.append(directory)

psf_expids = np.unique(psf_expids)
directories = np.unique(directories)
print('{0} images, {1} directories'.format(len(psf_expids), len(directories)))

## print(psf)
##
## psf_expids = {}
## for psf in psfs:
##     base_path, expid_str, ext_str, _ = psf.split('DECam')[-1].split('_')
##     expid = int(expid_str)
##     ext = int(ext_str)
##     key = base_path + 'DECam_' + expid_str + '_'
##     if key not in psf_expids:
##         psf_expids[key] = [expid, [ext]]
##      else:
##         psf_expids[key][1].append(ext)
##
## print('{0} images'.format(len(psf_expids.keys())))

# once you have the expids and directories, call them!
for directory in directories:
    expid = int(directory[-9:-1])
    compute_time_str = '5'
    logfile = out_dir + '/logs/{0:08d}.log'.format(expid)
    jobname = str(expid)
    mem_use = '1000'
    requirements = 'rhel60&&linux64 rusage[mem={0}] span[hosts=1]'.format(mem_use)
    code_path = '/nfs/slac/g/ki/ki18/cpd/code/WavefrontPSF/batch/assemble_data_psfex.py'
    command = ['bsub',
               '-W', compute_time_str,
               '-o', logfile,
               '-R', requirements,
               '-J', jobname,
               '-M', mem_use,
               'python', code_path,
               '--expid', str(expid),
               '--directory', directory]
    call(command)
