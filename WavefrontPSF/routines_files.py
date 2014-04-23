#!/usr/bin/env python
"""
File: routines_files.py
Author: Chris Davis
Description: A set of common routines for csv generation and creation.
"""

from __future__ import print_function, division
import numpy as np
from os import path, makedirs, system, remove, chdir
from subprocess import call
from decamutil_cpd import decaminfo
#import pyfits
from astropy.io import fits as pyfits
from routines import print_command

def extract_image_data(expids, path_image_data, path_out):
    """The big csv is huge; let's make a smaller one

    Parameters
    ----------
    expids : list
        A list of the expid's we are interested in extracting.

    path_image_data : string
        Path to the csv file that we load.

    path_out : string
        The place we will write to.

    Returns
    -------
    data_pruned : array
        The extracted data for the expids listed.

    Notes
    -----
    The slowest part of this function is in loading the big csv file.

    """

    data = np.recfromcsv(path_image_data, usemask=True)

    extract_list = []
    for expid in expids:
        # there has to be a better way
        extract_list.append(np.where(data['expid'] == expid)[0][0])

    data_pruned = data[extract_list]

    # make directory if it doesn't exist
    if not path.exists(path.dirname(path_out)):
        makedirs(path.dirname(path_out))

    # if the file does not exist, add the names_string
    if not path.exists(path_out):

        names = np.array(data_pruned.dtype.names)
        names_string = ''
        for name in names:
            names_string += name + ','

        names_string = names_string[:-1]

        f = open(path_out, 'w')
        f.write(names_string)
        f.write('\n')
    else:
        f = open(path_out, 'a')
    # this is annoying that I have to do this manually
    for i in range(data_pruned.size):
        size = len(data_pruned.mask[i])
        mask_i = data_pruned.mask[i]
        data_i = data_pruned.data[i]
        for j in range(size):
            if not mask_i[j]:
                f.write(np.str(data_i[j]))
            if j == size - 1:
                f.write('\n')
            else:
                f.write(',')
    f.close()

    return data_pruned


def collect_dictionary_results(path_out, item_dict={}):

    """Collect the results into one big csv file.

    Parameters
    ----------
    path_out : string
        The path where we want to output.

    item_dict : dictionary, optional
        Each key is a key that goes into the csv file. Each item for each key
        is a list of all the values for that key to be put in.

    Returns
    -------
    Nothing

    """

    if not path.exists(path_out):
        # create header
        names_string = ''
        for item_name in item_dict:
            names_string += item_name + ','
        names_string = names_string[:-1] + '\n'
        f = open(path_out, 'w')
        f.write(names_string)
    else:
        # I hope you have the same header!
        f = open(path_out, 'a')

    result_string = ''
    # item features
    length = len(item_dict[item_dict.keys()[0]])
    for item_iter in xrange(length):
        for item_name in item_dict:
            result_string += str(item_dict[item_name][item_iter])
            result_string += ','

        result_string = result_string[:-1] + '\n'
    f.write(result_string)

    f.close()

    return


def collect_fit_results(path_results, path_out,
                        user_dict={},
                        sub_dictionary=dict(
                            args=['rzero', 'dz', 'dx', 'dy', 'xt', 'yt',
                                  'e1', 'e2', 'z05d', 'z06d',
                                  'z07x', 'z07y', 'z08x', 'z08y'],
                            errors=['rzero', 'dz', 'dx', 'dy', 'xt', 'yt',
                                    'e1', 'e2', 'z05d', 'z06d',
                                    'z07x', 'z07y', 'z08x', 'z08y'],
                            mnstat=['amin', 'edm', 'errdef', 'nvpar',
                                    'nparx', 'icstat'],
                            status=['migrad_ierflg',
                                    'deltatime', 'force_derivatives',
                                    'strategy', 'max_iterations',
                                    'tolerance', 'verbosity',
                                    'nCalls', 'nCallsDerivative'])):

    """Collect the results into one big csv file.

    Parameters
    ----------
    path_results : list of strings
        A list of the paths to the results we want to collect.

    path_out : string
        The path where we want to output.

    user_dict : dictionary, optional
        Any extra tags you want to put into the csv file, under the heading
        'user_'. Each entry of the dictionary must be a list of the same length
        as the results list.
        Example dictionary: {'expid'

    sub_dictionary : dictionary, optional
        A dictionary whose entries are the names of the sub dictionaries in the
        results that we look at. The entries each have a list which correspond
        to the parameters we extract from that sub dictionary. So for example,
        sub_dictionary = {'mnstat':['amin', 'edm']} will extract from the
        fit_dictionary['mnstat']['amin'] and list it as 'mnstat_amin' in the
        csv file.

    Returns
    -------
    Nothing

    Notes
    -----
    status_migrad_erflg = 0 indicates migrad converged. 4 means it failed.

    """

    if not path.exists(path_out):
        # create header
        names_string = ''
        for sub_name in sub_dictionary:
            for sub_item in sub_dictionary[sub_name]:
                names_string += sub_name + '_' + sub_item + ','
        for user_name in user_dict:
            names_string += 'user_' + user_name + ','
        names_string = names_string[:-1] + '\n'

        f = open(path_out, 'w')
        f.write(names_string)
    else:
        # I hope you have the same header!
        f = open(path_out, 'a')

    for path_result_i in xrange(len(path_results)):
        path_result = path_results[path_result_i]
        fit_dict = np.load(path_result).item()
        fit_string = ''
        # sub_dictionary
        for sub_name in sub_dictionary:
            for sub_item in sub_dictionary[sub_name]:
                if sub_item in fit_dict[sub_name].keys():
                    fit_string += np.str(fit_dict[sub_name][sub_item])
                fit_string += ','
        # user features
        for user_name in user_dict:
            fit_string += np.str(user_dict[user_name][path_result_i])
            fit_string += ','

        fit_string = fit_string[:-1] + '\n'
        f.write(fit_string)
    f.close()

    return


def generate_path_results(expids, path_base):
    """convenience function to get my path_results

    Parameters
    ----------
    expids : list
        The list of expids I would like to filter from. The ones that don't
        exist get ignored.

    path_base : string
        The directory in which I look to make these results

    Returns
    -------
    path_results : list
        A list of strings pointing to the output *.npy files.

    expids_used : list
        A list of the expids used.

    """

    path_results = []
    expids_used = []
    for expid in expids:
        result = path_base + '{0:08d}_minuit_results.npy'.format(expid)
        if path.exists(path.dirname(result)):
            path_results.append(result)
            expids_used.append('{0:08d}'.format(expid))

    return path_results, expids_used


def generate_hdu_lists(
        expid,
        path_base='/nfs/slac/g/ki/ki18/cpd/catalogs/wgetscript/',
        name='cat_cpd',
        extension=2):
    """quick and dirty way of getting the hdu list format I am now using

    Parameters
    ----------
    expid : integer
        The image number I want to look at

    path_base : string
        The directory in which the catalogs are located

    name : string
        Tag for files.

    extension : int
        What extension of each hdu file are we looking at?

    Returns
    -------
    list_catalogs : list
        a list pointing to all the catalogs we wish to combine.
        Format is:

        path_base +
        '{0:08d}/DECam_{0:08d}_'.format(expid) +
        '{0:02d}_{1}.fits'.format(chip, name)

    list_fits_extension : list of integers
        a list pointing which extension on a given fits file we open
        format: [[2], [3,4]] says for the first in list_catalog, combine
        the 2nd extension with the 2nd list_catalog's 3rd and 4th
        extensions.

    list_chip : list of strings
        a list containing the extension name of the chip. ie [['N1'],
        ['S29', 'S5']]

    """

    list_catalogs_base = \
        path_base + '{0:08d}/DECam_{0:08d}_'.format(expid)
    list_catalogs = [list_catalogs_base + '{0:02d}_{1}.fits'.format(i, name)
                     for i in xrange(1, 63)]
    list_catalogs.pop(60)
    # ccd 2 went bad too.
    if expid > 258804:
        list_catalogs.pop(1)

    list_chip = [[decaminfo().ccddict[i]] for i in xrange(1, 63)]
    list_chip.pop(60)
    # ccd 2 went bad too.
    if expid > 258804:
        list_chip.pop(1)

    # ccd 2 went bad too.
    if expid > 258804:
        list_fits_extension = [[extension]] * (63-3)
    else:
        list_fits_extension = [[extension]] * (63-2)

    return list_catalogs, list_fits_extension, list_chip

def combine_decam_catalogs(list_catalogs, list_fits_extension, list_chip):
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


def download_desdm_filelist(expid, dataDirectory,
                   tag='Y1N_FIRSTCUT',
                   ccd=None, verbose=True):
    username = "cpd"
    password = "cpd70chips"

    # make directory if it doesn't exist
    if not path.exists(dataDirectory):
        makedirs(dataDirectory)

    # move there!
    chdir(dataDirectory)

    outname = "tempfilelist_%d.out" % (expid)

    # exposure table has expnum, and id, and this id is called exposureid in
    # the image table
    # image table has exposureid, id, run and imagetype
    # runtag table has the run and tag
    # filepath table has the id and filepath
    #
    cmd = 'trivialAccess -u %s -p %s -d dessci -c "select f.path from filepath f, image i, exposure e, runtag t where f.id = i.id and i.run = t.run and t.tag = \'%s\' and e.expnum = %d and i.imagetype = \'red\' and i.exposureid = e.id' % (username, password, tag, expid)

    if (ccd is not None):
        cmd = cmd + ' and i.ccd = %d' % (ccd)

    outname = "tempfilelist_%d.out" % (expid)
    cmd = cmd + '" > %s' % (outname)

    if verbose:
        print(cmd)

    call(cmd,shell=True)

def download_desdm(expid, dataDirectory,
                   tag='Y1N_FIRSTCUT',
                   download_catalog=True,
                   download_image=True,
                   download_psfcat=False,
                   download_background=False,
                   ccd=None, verbose=True):
    username = "cpd"
    password = "cpd70chips"

    # make directory if it doesn't exist
    if not path.exists(dataDirectory):
        makedirs(dataDirectory)

    # move there!
    chdir(dataDirectory)

    outname = "tempfilelist_%d.out" % (expid)
    if not path.exists(outname):
        download_desdm_filelist(expid, dataDirectory,
                   tag,
                   ccd, verbose)

    lines = [line.rstrip('\r\n') for line in open(outname)]
    junk = lines.pop(0)


    for line in sorted(lines):
        if ccd is not None:
            # only use the line that contains that ccd
            if '{0:08d}_{1:02d}'.format(expid, ccd) not in line:
                continue
        imfiles=[]
        catfiles=[]
        psfcatfiles=[]
        bkgfiles=[]

        parts=line.split('.fits')
        imfiles.append('https://desar2.cosmology.illinois.edu/DESFiles/desardata/' + parts[0]+'.fits.fz')
        catfiles.append('https://desar2.cosmology.illinois.edu/DESFiles/desardata/' + parts[0]+'_cat.fits')
        psfcatfiles.append('https://desar2.cosmology.illinois.edu/DESFiles/desardata/' + parts[0]+'_psfcat.fits')
        bkgfiles.append('https://desar2.cosmology.illinois.edu/DESFiles/desardata/' + parts[0]+'_bkg.fits.fz')

        for i in range(0,len(imfiles)):
            if not (download_catalog + download_image +
                    download_psfcat + download_background):
                print('You are not downloading anything!!!')
            else:
                cmd = 'wget --user='+username+' --password='+password+' --no-check-certificate'
                if download_image:
                    if not path.exists(imfiles[i].split('/')[-1]):
                        cmd += ' ' + imfiles[i]
                if download_catalog:
                    if not path.exists(catfiles[i].split('/')[-1]):
                        cmd += ' ' + catfiles[i]
                if download_psfcat:
                    if not path.exists(psfcatfiles[i].split('/')[-1]):
                        cmd += ' ' + psfcatfiles[i]
                if download_background:
                    if not path.exists(bkgfiles[i].split('/')[-1]):
                        cmd += ' ' + bkgfiles[i]

                if verbose:
                    print(cmd)

                call(cmd, shell=True)

def make_directory(directory):
    # I got tired of these two lines
    if not path.exists(directory):
        makedirs(directory)
