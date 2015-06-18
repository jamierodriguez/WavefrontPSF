#!/usr/bin/env python
"""
File: digestor.py
Author: Chris Davis
Description: Module for taking in data and making sense of it.


TODO: Incorporate digesting multiple objects
"""
from __future__ import print_function
try:
    from pandas import DataFrame
except Exception:
    raise ImportError('Pandas is missing!')

from WavefrontPSF.decamutil import decaminfo

class Digestor(object):
    """Class shell that takes in objects and makes sense of them.

    Attributes
    ----------

    Methods
    -------

    Notes
    -----

    """

    def __init__(self, **kwargs):
        pass

    def digest_fits(self, file_name, columns=None, exclude=['VIGNET', 'FLUX_APER', 'FLUXERR_APER', 'MAG_APER', 'MAGERR_APER'], ext=2, **kwargs):
        try:
            #from astropy.table import Table
            from astropy.io import fits
        except Exception:
            print('No Astropy installation. Trying pyfits!')
            try:
                import pyfits as fits
            except Exception:
                raise ImportError('Astropy and Pyfits both missing!')

        df = DataFrame.from_records(fits.getdata(file_name, ext=ext), exclude=exclude, columns=columns)
        if 'x' not in df.keys() and 'XWIN_IMAGE' in df.keys() and 'ext' in df.keys():
            decaminf = decaminfo()
            # get focal plane coordinates
            xPos = df['XWIN_IMAGE']
            yPos = df['YWIN_IMAGE']
            extnums = df['ext'].values
            x, y = decaminf.getPosition_extnum(extnums, xPos, yPos)
            df['x'] = x
            df['y'] = y

        # TODO: This is a really annoying hackaround the endianness problem:
        # turn everything into floats. This should be fixed in the next major
        # astropy release 1.1
        df = df.astype('<f8')

        return df

        #return Table(file_name, hdu=ext).to_pandas()

    def digest_csv(self, file_name, **kwargs):
        try:
            from pandas import read_csv
        except Exception:
            raise ImportError('Pandas is missing!')

        return read_csv(file_name, index_col=0)

    def digest_npbinary(self, file_name, **kwargs):
        try:
            from numpy import load
        except Exception:
            raise ImportError('Numpy is missing!')

        return DataFrame.from_records(load(file_name))

    def digest_directory(self, file_directory, file_type='.fits'):
        # another example file_type = _selpsfcat.fits
        from glob import glob
        files = glob(file_directory + '/*{0}'.format(file_type))
        data = self(files[0])
        for file in files[1:]:
            data = data.append(self(file), ignore_index=True)
        return data


    def digest(self, file_name, **kwargs):
        """Convert file to a record array containing data we want. This
        function determines which digestor we want to use.

        Parameters
        ----------

        @param file_name        string, list of strings. Location of files to
                                digest.
        @returns digested_data  Digested data.
        """

        # check file exists
        from os import path
        if not path.isfile(file_name):
            raise IOError("The file '"+str(file_name)+"' does not exist.")

        # get file type
        file_type = file_name.split('.')[-1]
        if file_type == 'npy':
            data = self.digest_npbinary(file_name, **kwargs)
        elif file_type == 'fits':
            data = self.digest_fits(file_name, **kwargs)
        elif file_type == 'csv':
            data = self.digest_csv(file_name, **kwargs)
        else:
            # try just using numpy load
            try:
                data = self.digest_npbinary(file_name, **kwargs)
            except Exception:
                # I think this is the right kind of error
                raise ValueError("{0} does not seem to be a type of file ({1}) that can be loaded currently.".format(file_name, file_type))

        # digest accordingly! See if I need to do anything...
        digested_data = data

        return digested_data

    def __call__(self, file_name, **kwargs):
        """
        Parameters
        ----------

        @param file_name    string, list of strings. Location of files to digest.
        @returns            Digested data.
        """

        return self.digest(file_name, **kwargs)



