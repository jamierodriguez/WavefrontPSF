# NOTE: UNFINISHED
"""
In the end the for loop below should create the data for one expid
and save it in a csv file
then I need an argparse for dealing with the expid creation (for batch farming)
and an argparse for doing the batch farming
and an argparse for combining all the csvs together at the end of the day
"""

from glob import glob
from astropy.io import fits
import pandas as pd
import numpy as np
from os import path, makedirs

from WavefrontPSF.focal_plane_psfex import FocalPlanePSFEx
from WavefrontPSF.decamutil_cpd import decaminfo
from WavefrontPSF.routines import convert_dictionary

out_dir = '/nfs/slac/g/ki/ki18/des/cpd/big_psfex_rerun'

used_keys = ['DELTAX_IMAGE', 'DELTAY_IMAGE',
             'NORM_PSF', 'CHI2_PSF', 'RESI_PSF']
moments_keys = ['a4',
                'e0', 'e0prime',
                'e1', 'e2',
                'x2', 'y2', 'xy',
                'delta1', 'delta2',
                'zeta1', 'zeta2',
                'flux', 'fwhm']

decaminf = decaminfo()

def assemble_data_psfex(expid, directory):

    data = {}

    # create focal plane object
    ext_list = range(1, 61) + [62]



    list_catalogs = ['{0}/DECam_{1:08d}_{2:02d}_psfcat.psf'.format(
        directory, expid, ext) for ext in ext_list]
    list_chip = [decaminf.ccddict[ext] for ext in ext_list]

    FP = FocalPlanePSFEx(list_catalogs, list_chip)

    list_fits_catalogs = ['{0}/DECam_{1:08d}_{2:02d}_psfcat.used.fits'.format(
        directory, expid, ext) for ext in ext_list]
    coords = np.empty([0, 3])
    for used_i, used in enumerate(list_fits_catalogs):
        # collect other info information
        data_i = fits.getdata(used, ext=2)

        coords_i = np.vstack((data_i['X_IMAGE'], data_i['Y_IMAGE'],
                            [ext_list[used_i]] * len(data_i))).T
        coords = np.vstack((coords, coords_i))

        for key in used_keys:
            if key not in data:
                data[key] = []
            data[key].append(data_i[key])

    moments = FP.plane(coords)

    expids = np.array([expid] * coords.shape[0])
    coords_focal = FP.pixel_to_position(coords)



    # data is empty; so let's make it!

    # do moments_keys
    for key in moments_keys:
        data[key] = moments[key]
    # do used_keys
    for key in used_keys:
        data[key] = np.concatenate(data[key])
    # do coordinates with proper transformation also included
    data['X_IMAGE'] = moments['x']
    data['Y_IMAGE'] = moments['y']
    data['x'] = coords_focal[:, 0]
    data['y'] = coords_focal[:, 1]
    data['ext'] = coords_focal[:, 2]
    data['expid'] = expids


    print('saving {0} with {1} entries'.format(expid, len(data['expid'])))
    recarr = convert_dictionary(data)
    ## np.save(out_dir + 'collate_psf', recarr)
    # https://github.com/pydata/pandas/issues/3778
    df = pd.DataFrame(recarr)
    # note that __repr__ will fail for df some reason,
    # but columns won't?! (e.g. df['x'] is fine)
    # should happen first; gets us the header
    out_dir_expid = out_dir + '/{0:08d}'.format(expid)
    if not path.exists(out_dir_expid):
        makedirs(out_dir_expid)
    df.to_csv(out_dir_expid + '/collate_psf.csv')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='run the jobs!')
    parser.add_argument('--expid',
                        action='store',
                        dest='expid',
                        type=int,
                        help='expid')
    parser.add_argument('--directory',
                        action='store',
                        dest='directory',
                        help='what directory do we look at?')
    options = parser.parse_args()
    args = vars(options)

    assemble_data_psfex(args['expid'], args['directory'])
