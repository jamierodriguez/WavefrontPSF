# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%load_ext line_profiler

# <codecell>

from focal_plane import FocalPlane
from decamutil_cpd import decaminfo
from decam_csv_routines import generate_hdu_lists
from decam_csv_routines import combine_decam_catalogs
# create an FP object
path_base = '/Users/cpd/Desktop/Images/'
i = 1
expid = 232698
list_catalogs_base = \
    path_base + '{0:08d}/DECam_{0:08d}_'.format(expid)
list_catalogs = [list_catalogs_base + '{0:02d}_cat.fits'.format(i)]
list_chip = [[decaminfo().ccddict[i]]]
list_fits_extension = [[2]]

list_catalogs, list_fits_extension, list_chip = generate_hdu_lists(232698, path_base)

cpd_args = dict(
    path_mesh='/Users/cpd/Desktop/Meshes/',
    mesh_name='Science20120915s1v3_134239',
    boxdiv=0,
    max_samples_box=20000,
    conds='default', #'minimal',
    list_catalogs=list_catalogs,
    list_fits_extension=list_fits_extension,
    list_chip=list_chip,

    nbin=256,
    nPixels=32,
    pixelOverSample=8,
    scaleFactor=1.,
    background=0,
    )

f = FocalPlane(**cpd_args)

cpd_args2 = cpd_args.copy()
cpd_args2['max_samples_box'] = 20
cpd_args2['boxdiv'] = 1
f2 = FocalPlane(**cpd_args2)

# <codecell>

recdata, extension, recheader_all = combine_decam_catalogs(list_catalogs, list_fits_extension, list_chip)
lp = %lprun -r -f f.filter f.filter(recdata=recdata, conds='default')
lp.print_stats()

# <codecell>

recdata, extension, recheader_all = combine_decam_catalogs(list_catalogs, list_fits_extension, list_chip)
lp1 = %lprun -r -f f.filter_number_in_box f.filter_number_in_box(recdata=recdata, extension=extension, max_samples_box=f.max_samples_box, boxdiv=f.boxdiv)
lp1.print_stats()
# old time 0.907664 s

# <codecell>

recdata, extension, recheader_all = combine_decam_catalogs(list_catalogs, list_fits_extension, list_chip)
lp3 = %lprun -r -f f.filter_number_in_box_old f.filter_number_in_box_old(recdata=recdata, extension=extension, max_samples_box=f.max_samples_box, boxdiv=f.boxdiv)
lp3.print_stats()

# <codecell>

lp = %lprun -r -f f.create_data_unaveraged f.create_data_unaveraged(recdata=f.recdata, extension=f.extension)
lp.print_stats()

# <codecell>

lp = %lprun -r -f f2.plane f2.plane({'rzero':0.14}, coords=f2.coords)
lp.print_stats()

# <codecell>

from adaptive_moments import adaptive_moments

rzero = 0.14
zernikes = f2.zernikes(f2.coords, {})
background = f2.background
threshold = 0
data = f2.stamp(zernike=zernikes[0], rzero=rzero, coord=f2.coords[0][:2])
stamp = np.where(data - background > threshold, data - background, 0)

lp = %lprun -r -f f2.moments f2.moments(data)
lp.print_stats()

# <codecell>

lp2 = %lprun -r -f combine_decam_catalogs combine_decam_catalogs(list_catalogs, list_fits_extension, list_chip)
lp2.print_stats()

# <codecell>

import pyfits
list_catalogs, list_fits_extension, list_chip = generate_hdu_lists(232698, path_base)
def combine_decam_catalogs2(list_catalogs, list_fits_extension, list_chip):
    """assemble an array from all the focal plane chips

    Parameters
    ----------
    list_catalogs : list
        a list pointing to all the catalogs we wish to combine.

    list_fits_extension : list of integers
        a list pointing which extension on a given fits file we open
        format: [[2], [3,4]] says for the first in list_catalog, combine
        the 2nd extension with the 2nd list_catalog's 3rd and 4th
        extensions.

    list_chip : list of strings
        a list containing the extension name of the chip. ie [['N1'],
        ['S29', 'S5']]

    Returns
    -------
    recdata_all : recarray
        The entire contents of all the fits extensions combined

    ext_all : array
        Array of all the extension names

    """

    recdata_all = []
    recheader_all = []
    ext_all = []
    for catalog_i in xrange(len(list_catalogs)):
        hdu_path = list_catalogs[catalog_i]

        try:
            hdu = pyfits.open(hdu_path)
        except IOError:
            print('Cannot open ', hdu_path)
            continue

        fits_extension_i = list_fits_extension[catalog_i]
        chip_i = list_chip[catalog_i]

        for fits_extension_ij in xrange(len(fits_extension_i)):
            ext_name = chip_i[fits_extension_ij]
            recdata = hdu[fits_extension_i[fits_extension_ij]].data
            recheader = hdu[fits_extension_i[fits_extension_ij]].header

            recdata_all += recdata.tolist()
            ext_all += [ext_name] * recdata.size
            recheader_all += recheader

        hdu.close()

    recdata_all = np.array(recdata_all, dtype=recdata.dtype)
    ext_all = np.array(ext_all)
    return recdata_all, ext_all, recheader_all
list_catalogs, list_fits_extension, list_chip = generate_hdu_lists(232698, path_base)
r, e, h = combine_decam_catalogs2(list_catalogs, list_fits_extension, list_chip)
lp2 = %lprun -r -f combine_decam_catalogs2 combine_decam_catalogs2(list_catalogs, list_fits_extension, list_chip)
lp2.print_stats()

# <codecell>

%%timeit
list_catalogs, list_fits_extension, list_chip = generate_hdu_lists(232698, path_base)
_ = combine_decam_catalogs(list_catalogs, list_fits_extension, list_chip)

# <codecell>

%%timeit
list_catalogs, list_fits_extension, list_chip = generate_hdu_lists(232698, path_base)
_ = combine_decam_catalogs2(list_catalogs, list_fits_extension, list_chip)

# <codecell>

%%timeit
f2 = FocalPlane(**cpd_args2)

# <codecell>

lp2 = %lprun -r -f FocalPlane FocalPlane(**cpd_args2)
lp2.print_stats()

# <codecell>


