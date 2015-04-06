#!/usr/bin/env python
"""
File: digestor.py
Author: Chris Davis
Description: Module for taking in data and making sense of it.


TODO: Make digested data all the same KIND of object. Let's say Pandas Dataframe!
TODO: Incorporate digesting multiple objects
"""
from __future__ import print_function
try:
    from pandas import DataFrame
except Exception:
    raise ImportError('Pandas is missing!')

class Digestor(object):
    """Class shell that takes in objects and makes sense of them.

    Attributes
    ----------

    Methods
    -------

    Notes
    -----

    """

    def __init__(self):
        pass

    def digest_fits(self, file_name):
        try:
            from astropy.io import fits
        except Exception:
            print('No Astropy installation. Trying pyfits!')
            try:
                import pyfits as fits
            except Exception:
                raise ImportError('Astropy and Pyfits both missing!')

        return DataFrame.from_records(fits.getdata(file_name))

    def digest_csv(self, file_name):
        try:
            from pandas import read_csv
        except Exception:
            raise ImportError('Pandas is missing!')

        return read_csv(file_name)

    def digest_npbinary(self, file_name):
        try:
            from numpy import load
        except Exception:
            raise ImportError('Numpy is missing!')

        return DataFrame.from_records(load(file_name))

    def digest(self, file_name):
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
            data = self.digest_npbinary(file_name)
        elif file_type == 'fits':
            data = self.digest_fits(file_name)
        elif file_type == 'csv':
            data = self.digest_csv(file_name)
        else:
            # try just using numpy load
            try:
                data = self.digest_npbinary(file_name)
            except Exception:
                # I think this is the right kind of error
                raise ValueError("{0} does not seem to be a type of file ({1}) that can be loaded currently.".format(file_name, file_type))

        # digest accordingly! See if I need to do anything...
        digested_data = data

        return digested_data

    def __call__(self, file_name):
        """
        Parameters
        ----------

        @param file_name    string, list of strings. Location of files to digest.
        @returns            Digested data.
        """

        return self.digest(file_name)


