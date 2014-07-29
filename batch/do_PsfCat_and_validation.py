#!/usr/bin/env python

#
# $Rev:: 177                                                          $:
# $Author:: cpd                                                       $:
# $LastChangedDate:: 2014-03-04 15:45:28 -0800 (Tue, 04 Mar 2014)     $:
#
# code to analyze psfcat stars
#
import os
import argparse
import subprocess
import numpy
import astropy.io.fits as pyfits
from wavefront import Wavefront
from routines_moments import convert_moments
from routines_files import download_desdm, download_desdm_filelist
import pdb

parser = argparse.ArgumentParser(prog='selectPsfCat_and_validation')
parser.add_argument("-e", "--expid",
              dest="expid",
              type=int,
              help="sispi exposure number")
parser.add_argument("-l", "--filelist",
              dest="filelist",
              default="/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/SVA1_FINALCUT_filelist.out",
              help="file with lists of file locations")
parser.add_argument("-o", "--base_catalog_directory",
              dest="base_catalog_directory",
              default="/nfs/slac/g/ki/ki18/cpd/psfextest",
              help="Location of base output directory.")
parser.add_argument("-f", "--fraction",
              dest="fraction",
              default=0.2,
              type=float,
              help="Fraction of data saved for validation.")
parser.add_argument("-g", "--getIn",
              dest="getIn",
              default=1,
              type=int,
              help="Do we download the files?")
parser.add_argument("-d", "--deleteIn",
              dest="deleteIn",
              default=1,
              type=int,
              help="Do we delete input files once done?")
parser.add_argument("-t", "--tag",
              dest="tag",
              default="SVA1_FINALCUT",
              help="Tag for the input file. (SVA1_FINALCUT, Y1N_FIRSTCUT, etc)")
parser.add_argument("-c", "--ccd",
              dest="ccd",
              default=0,
              type=int,
              help="CCD Number to run on. 0 does all science chips.")
parser.add_argument("--download_catalog",
              dest="download_catalog",
              default=0,
              type=int,
              help="Download associated finalcut catalog?")
parser.add_argument("--download_background",
              dest="download_background",
              default=0,
              type=int,
              help="Download associated finalcut background?")
parser.add_argument("--download_image",
              dest="download_image",
              default=0,
              type=int,
              help="Download associated finalcut image?")
parser.add_argument("--download_psfcat",
              dest="download_psfcat",
              default=1,
              type=int,
              help="Download psfcat and make assocciated validation cat?")

def find_nearest(a, a0):
    """ return index of element in nd array a equal to the scalar value a0
    Assumes match is unique.
    """
    idx = numpy.abs(a - a0).argmin()
    if a[idx]!=a0:
        idx = None
    return idx

def calc_threshold(values):
    """ calculate median and sigma level of all pixel values
    """

    # remove pixel values for masked pixels
    values = values[(values>-1e8)*(values>-80)]

    # calculate MAD,sigma
    med = numpy.median(values)
    mad = numpy.median(numpy.abs(values-med))
    sigma = 1.4826 * mad   #for a Gaussian distribution
    print "calc_threshold: 1st iteration Median,sigma: ",med,sigma

    # truncate at 5 sigma and iterate once
    values = values[(values<(med + 5.0*sigma))*(values>(med-5.0*sigma))]

    # calculate MAD,sigma
    med = numpy.median(values)
    mad = numpy.median(numpy.abs(values-med))
    sigma = 1.4826 * mad
    print "calc_threshold: 2nd iteration Median,sigma: ",med,sigma

    return med,sigma

def mkSelPsfCat(expid, tag="SVA1_FINALCUT",
        base_catalog_directory="/nfs/slac/g/ki/ki22/roodman/dmdata",
        filelist=None,
        getIn=True, deleteIn=True, fraction=0.2, ccd=0,
        download_catalog=False, download_psfcat=True,
        download_image=False, download_background=False,
        **kwargs):
    """ make a -psfcat.fits catalog with just stars used by PSFex
    1) call psfex with option to make _out.cat
    2) readin _out.cat pick these stars out of -psfcat.fits
    3) write out selected stars into -selpsfcat.fits
    4) if needed delete the -psfcat.fits catalog
    """

    # make/move to data directory
    rungroup = int(expid/1000) * 1000
    directory = os.path.join(base_catalog_directory,tag,"psfcat/%08d/%08d" % (rungroup,expid))
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)

    # run psfex & build selpsfcat
    if ccd == 0:
        ccdrange = range(1, 62 + 1)
    else:
        ccdrange = [ccd]

    if not filelist:
        if ccd == 0:
            filelist = directory + "/filelist_%d.out" % (expid)
        else:
            filelist = directory + "/filelist_%d_%d.out" % (expid, ccd)

    for i in ccdrange:

        if getIn:
            #getPsfCat(expid,tag,i)
            download_desdm(expid, directory, filelist=filelist,
                           tag=tag, ccd=i,
                           download_catalog=download_catalog,
                           download_psfcat=download_psfcat,
                           download_image=download_image,
                           download_background=download_background)

        # do psfcat part
        if download_psfcat:
            filename = "DECam_%08d_%02d_psfcat.fits" % (expid,i)
            catname = "DECam_%08d_%02d_outcat.fits" % (expid,i)
            filename_fin = "DECam_%08d_%02d_psfcat_validation_subtracted.fits" %  \
                    (expid,i)
            catname_fin = "DECam_%08d_%02d_outcat_validation_subtracted.fits" %  \
                    (expid,i)

            selname = "DECam_%08d_%02d_selpsfcat.fits" % (expid,i)
            valname = "DECam_%08d_%02d_valpsfcat.fits" % (expid,i)
            #sexcatname = "DECam_%08d_%02d_cat.fits" % (expid,i)


            if os.path.exists(filename):
                cmd = "/nfs/slac/g/ki/ki22/roodman/DESDM/eups/packages/Linux64/psfex/3.17.0+0/bin/psfex %s -c /u/ec/roodman/Astrophysics/PSF/desdm-plus.psfex -OUTCAT_NAME %s" % (filename,catname)
                print cmd
                os.system(cmd)

                # filter out a validation cat
                buildValidationCat(expid, i, filename, catname, filename_fin,
                                   valname, fraction=fraction)

                # build your final actual selpsfcat
                cmd = "/nfs/slac/g/ki/ki22/roodman/DESDM/eups/packages/Linux64/psfex/3.17.0+0/bin/psfex %s -c /u/ec/roodman/Astrophysics/PSF/desdm-plus.psfex -OUTCAT_NAME %s" % (filename_fin, catname_fin)
                print cmd
                os.system(cmd)

                buildSelPsfCat(expid,i,filename_fin,catname_fin,selname)

                if deleteIn:
                    os.remove(filename)
                    os.remove(filename_fin)
                    os.remove(catname)
                    os.remove(catname_fin)
            else:
                print(filename + ' does not exist!')

def buildSelPsfCat(expid,iext,filename,catname,selname):
    """ actually build the selpsfcat file
    """

    psfcathdu = pyfits.open(filename)
    psfimhead = psfcathdu[1]
    psfcat = psfcathdu[2].data
    npsfcat = psfcat.shape[0]

    outcathdu = pyfits.open(catname)
    outcat = outcathdu[2].data
    noutcat = outcat.shape[0]

    # loop over selected stars, build a mask for them in the psfcat
    mask = numpy.zeros((npsfcat),dtype=bool)
    sourceArr = outcat['SOURCE_NUMBER']
    for i in range(noutcat):
        isource = sourceArr[i] - 1   #need to start index at 0
        mask[isource] = True

    # mask the recordArrays
    newcat = psfcat[mask]

    # collect all pixels from all VIGNET's and find 1sigma background level
    allpixels = newcat['VIGNET'].flatten()
    median,sigma = calc_threshold(allpixels)

    # merge additional information - both from PSFex and from moment calculation
    indices = numpy.zeros((noutcat),dtype=int) # index into outcat arrays, ordered to match newcat
    iwhich = numpy.array(range(npsfcat),dtype=int)
    iwhich = iwhich[mask]     # index into psfcat, ordered to match newcat

    # loop over newcat
    nrows = newcat.shape[0]
    for irow in range(nrows):
        isource = iwhich[irow] + 1 #sourceArr starts at 1
        # find this guy in the outcat, since the ordering is different
        indices[irow] = find_nearest(sourceArr,isource)

    # build arrays for the FITS table
    extens = outcat['EXTENSION'][indices]
    normpsf = outcat['NORM_PSF'][indices]
    chi2psf = outcat['CHI2_PSF'][indices]
    resipsf = outcat['RESI_PSF'][indices]
    delximg = outcat['DELTAX_IMAGE'][indices]
    delyimg = outcat['DELTAY_IMAGE'][indices]

    # calculate moments with Chris's code
    WF = Wavefront()
    e0 = numpy.zeros((noutcat))
    e1 = numpy.zeros((noutcat))
    e2 = numpy.zeros((noutcat))
    flux = numpy.zeros((noutcat))
    delta1 = numpy.zeros((noutcat))
    delta2 = numpy.zeros((noutcat))
    zeta1 = numpy.zeros((noutcat))
    zeta2 = numpy.zeros((noutcat))
    a4 = numpy.zeros((noutcat))
    fwhm_adaptive = numpy.zeros((noutcat))
    irowArr = numpy.zeros((noutcat),dtype=numpy.int32)

    for irow in range(nrows):
        row = newcat[irow]
        irowArr[irow] = irow
        # call CPD code to calculate moments
        stamp = numpy.copy(row['VIGNET'])
        stamp = stamp.astype(numpy.float64)
        moments = WF.moments(stamp, background=median, threshold=median+sigma)
        moments = convert_moments(moments)

        e0[irow] = moments['e0']
        e1[irow] = moments['e1']
        e2[irow] = moments['e2']
        flux[irow] = moments['flux']
        delta1[irow] = moments['delta1']
        delta2[irow] = moments['delta2']
        zeta1[irow] = moments['zeta1']
        zeta2[irow] = moments['zeta2']

        a4[irow] = moments['a4']
        fwhm_adaptive[irow] = moments['fwhm']

    # build 2nd table from extra columns
    c0 = pyfits.Column(name='IROW',format='1J',array=irowArr)
    c1 = pyfits.Column(name='NORM_PSF',format='1E',array=normpsf,unit='count')
    c2 = pyfits.Column(name='CHI2_PSF',format='1E',array=chi2psf)
    c3 = pyfits.Column(name='RESI_PSF',format='1E',array=resipsf)
    c4 = pyfits.Column(name='DELTAX_IMAGE',format='1E',array=delximg,unit='pixel')
    c5 = pyfits.Column(name='DELTAY_IMAGE',format='1E',array=delyimg,unit='pixel')
    c6 = pyfits.Column(name='e0',format='1E',array=e0,unit='arcsec^2')
    c7 = pyfits.Column(name='e1',format='1E',array=e1,unit='arcsec^2')
    c8 = pyfits.Column(name='e2',format='1E',array=e2,unit='arcsec^2')
    c9 = pyfits.Column(name='FLUX_ADAPTIVE',format='1E',array=flux,unit='counts')
    c10 = pyfits.Column(name='delta1',format='1E',array=delta1,unit='arcsec^3')
    c11 = pyfits.Column(name='delta2',format='1E',array=delta2,unit='arcsec^3')
    c12 = pyfits.Column(name='zeta1',format='1E',array=zeta1,unit='arcsec^3')
    c13 = pyfits.Column(name='zeta2',format='1E',array=zeta2,unit='arcsec^3')
    c14 = pyfits.Column(name='A4_ADAPTIVE',format='1E',array=a4)
    c15 = pyfits.Column(name='FWHM_ADAPTIVE',format='1E',array=fwhm_adaptive,unit='pixel')

    # add some medians and image information too
    e0Median = numpy.median(e0)
    e1Median = numpy.median(e1)
    e2Median = numpy.median(e2)
    delta1Median = numpy.median(delta1)
    delta2Median = numpy.median(delta2)
    zeta1Median = numpy.median(zeta1)
    zeta2Median = numpy.median(zeta2)
    #fwhm = 2.355 * numpy.sqrt(e0)
    # CPD 29.07.14: added empirical e0-fwhm relation
    fwhm = numpy.sqrt(4.220 * e0 + 0.151)
    fwhmMedian = numpy.median(fwhm)

    arr20 = numpy.ones((nrows),dtype=numpy.int32) * expid
    c20 = pyfits.Column(name='expid',format='1J',array=arr20)
    arr21 = numpy.ones((nrows),dtype=numpy.int16) * iext
    c21 = pyfits.Column(name='ext',format='1I',array=arr21)
    c22 = pyfits.Column(name='fwhm',format='1E',array=fwhm,unit='arcsec')
    arr23 = numpy.ones((nrows),dtype=numpy.float32) * fwhmMedian
    c23 = pyfits.Column(name='fwhmMedian',format='1E',array=arr23,unit='arcsec')
    arr24 = numpy.ones((nrows),dtype=numpy.float32) * e0Median
    c24 = pyfits.Column(name='e0Median',format='1E',array=arr24,unit='arcsec^2')
    arr25 = numpy.ones((nrows),dtype=numpy.float32) * e1Median
    c25 = pyfits.Column(name='e1Median',format='1E',array=arr25,unit='arcsec^2')
    arr26 = numpy.ones((nrows),dtype=numpy.float32) * e2Median
    c26 = pyfits.Column(name='e2Median',format='1E',array=arr26,unit='arcsec^2')
    arr27 = numpy.ones((nrows),dtype=numpy.float32) * delta1Median
    c27 = pyfits.Column(name='delta1Median',format='1E',array=arr27,unit='arcsec^3')
    arr28 = numpy.ones((nrows),dtype=numpy.float32) * delta2Median
    c28 = pyfits.Column(name='delta2Median',format='1E',array=arr28,unit='arcsec^3')
    arr29 = numpy.ones((nrows),dtype=numpy.float32) * zeta1Median
    c29 = pyfits.Column(name='zeta1Median',format='1E',array=arr29,unit='arcsec^3')
    arr30 = numpy.ones((nrows),dtype=numpy.float32) * zeta2Median
    c30 = pyfits.Column(name='zeta2Median',format='1E',array=arr30,unit='arcsec^3')

    coldefs = pyfits.ColDefs([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10,
         c11, c12, c13, c14, c15, c20, c21, c22, c23, c24, c25, c26, c27, c28,
         c29, c30])
    newt2hdu = pyfits.new_table(coldefs)

    # fill new table
    newthdu = pyfits.new_table(newcat.columns)
    newthdu.data = newcat

    # merge the two tables
    newtable = newthdu.columns + newt2hdu.columns
    # for some reason it is in the conversion newtable to newtablehdu that the
    # vignet parameter goes wrong; can check via newtable.columns[0].array vs
    # newtablehdu.data['VIGNET']
    newtablehdu = pyfits.new_table(newtable)
    # try to fix that
    newtablehdu.data['VIGNET'] = newcat['VIGNET']

    # insert table name (NEEDED FOR PSFEX!!!)
    # newtablehdu.header.insert(8, ('EXTNAME', 'LDAC_OBJECTS', 'TABLE NAME'))
    newtablehdu.header.set('EXTNAME', 'LDAC_OBJECTS', 'a name')

    # make output fits file
    prihdr = pyfits.Header()
    prihdu = pyfits.PrimaryHDU(header=prihdr)
    thdulist = pyfits.HDUList([prihdu,psfimhead,newtablehdu])
    thdulist.writeto(selname,clobber=True)

def buildValidationCat(expid, iext, filename, catname,
                       filename_out, valname, fraction=0.2):

    """ build the validation catalog
    """

    psfcathdu = pyfits.open(filename)
    psfimhead = psfcathdu[1]
    psfcat = psfcathdu[2].data
    npsfcat = psfcat.shape[0]

    outcathdu = pyfits.open(catname)
    outcat = outcathdu[2].data
    noutcat = outcat.shape[0]

    # loop over selected stars, build a mask for them to put into valname
    mask = numpy.zeros((npsfcat), dtype=bool)
    mask_all = numpy.zeros((npsfcat), dtype=bool)

    # create validation selection
    validation_sample = numpy.zeros((npsfcat), dtype=bool)
    validation_sample[:numpy.int(fraction * npsfcat)] = True
    numpy.random.shuffle(validation_sample)


    sourceArr = outcat['SOURCE_NUMBER']
    for i in xrange(noutcat):
        isource = sourceArr[i] - 1   #need to start index at 0
        # decide whether to put into validation (so mask them) or not
        if validation_sample[i]:
            mask[isource] = True
        mask_all[isource] = True


    # collect all pixels from all VIGNET's and find 1sigma background level
    allpixels = psfcat[mask_all]['VIGNET'].flatten()
    median,sigma = calc_threshold(allpixels)

    # get rid of the validation entries
    psfcathdu[2].data = psfcat[~mask]
    psfcathdu.writeto(filename_out, clobber=True)


    # now the process of creating the validation catalog:

    # create validation cat
    valcat = psfcat[mask]
    nrows = valcat.shape[0]
    # psfcathdu[2].data = valcat
    # psfcathdu.writeto(valname, clobber=True)


    # calculate moments with Chris's code
    WF = Wavefront()
    e0 = numpy.zeros((nrows))
    e1 = numpy.zeros((nrows))
    e2 = numpy.zeros((nrows))
    flux = numpy.zeros((nrows))
    delta1 = numpy.zeros((nrows))
    delta2 = numpy.zeros((nrows))
    zeta1 = numpy.zeros((nrows))
    zeta2 = numpy.zeros((nrows))
    a4 = numpy.zeros((nrows))
    fwhm_adaptive = numpy.zeros((nrows))

    for irow in range(nrows):
        row = valcat[irow]
        # call CPD code to calculate moments
        stamp = numpy.copy(row['VIGNET'])
        stamp = stamp.astype(numpy.float64)
        moments = WF.moments(stamp, background=median, threshold=median+sigma)
        moments = convert_moments(moments)

        e0[irow] = moments['e0']
        e1[irow] = moments['e1']
        e2[irow] = moments['e2']
        flux[irow] = moments['flux']
        delta1[irow] = moments['delta1']
        delta2[irow] = moments['delta2']
        zeta1[irow] = moments['zeta1']
        zeta2[irow] = moments['zeta2']

        a4[irow] = moments['a4']
        fwhm_adaptive[irow] = moments['fwhm']


    # build 2nd table from extra columns
    c6 = pyfits.Column(name='e0',format='1E',array=e0,unit='arcsec^2')
    c7 = pyfits.Column(name='e1',format='1E',array=e1,unit='arcsec^2')
    c8 = pyfits.Column(name='e2',format='1E',array=e2,unit='arcsec^2')
    c9 = pyfits.Column(name='FLUX_ADAPTIVE',format='1E',array=flux,unit='counts')
    c10 = pyfits.Column(name='delta1',format='1E',array=delta1,unit='arcsec^3')
    c11 = pyfits.Column(name='delta2',format='1E',array=delta2,unit='arcsec^3')
    c12 = pyfits.Column(name='zeta1',format='1E',array=zeta1,unit='arcsec^3')
    c13 = pyfits.Column(name='zeta2',format='1E',array=zeta2,unit='arcsec^3')
    c14 = pyfits.Column(name='A4_ADAPTIVE',format='1E',array=a4)
    c15 = pyfits.Column(name='FWHM_ADAPTIVE',format='1E',array=fwhm_adaptive,unit='pixel')

    # add some medians and image information too
    e0Median = numpy.median(e0)
    e1Median = numpy.median(e1)
    e2Median = numpy.median(e2)
    delta1Median = numpy.median(delta1)
    delta2Median = numpy.median(delta2)
    zeta1Median = numpy.median(zeta1)
    zeta2Median = numpy.median(zeta2)
    #fwhm = 2.355 * numpy.sqrt(e0)
    # CPD 29.07.14: added empirical e0-fwhm relation
    fwhm = numpy.sqrt(4.220 * e0 + 0.151)
    fwhmMedian = numpy.median(fwhm)

    arr20 = numpy.ones((nrows),dtype=numpy.int32) * expid
    c20 = pyfits.Column(name='expid',format='1J',array=arr20)
    arr21 = numpy.ones((nrows),dtype=numpy.int16) * iext
    c21 = pyfits.Column(name='ext',format='1I',array=arr21)
    c22 = pyfits.Column(name='fwhm',format='1E',array=fwhm,unit='arcsec')
    arr23 = numpy.ones((nrows),dtype=numpy.float32) * fwhmMedian
    c23 = pyfits.Column(name='fwhmMedian',format='1E',array=arr23,unit='arcsec')
    arr24 = numpy.ones((nrows),dtype=numpy.float32) * e0Median
    c24 = pyfits.Column(name='e0Median',format='1E',array=arr24,unit='arcsec^2')
    arr25 = numpy.ones((nrows),dtype=numpy.float32) * e1Median
    c25 = pyfits.Column(name='e1Median',format='1E',array=arr25,unit='arcsec^2')
    arr26 = numpy.ones((nrows),dtype=numpy.float32) * e2Median
    c26 = pyfits.Column(name='e2Median',format='1E',array=arr26,unit='arcsec^2')
    arr27 = numpy.ones((nrows),dtype=numpy.float32) * delta1Median
    c27 = pyfits.Column(name='delta1Median',format='1E',array=arr27,unit='arcsec^3')
    arr28 = numpy.ones((nrows),dtype=numpy.float32) * delta2Median
    c28 = pyfits.Column(name='delta2Median',format='1E',array=arr28,unit='arcsec^3')
    arr29 = numpy.ones((nrows),dtype=numpy.float32) * zeta1Median
    c29 = pyfits.Column(name='zeta1Median',format='1E',array=arr29,unit='arcsec^3')
    arr30 = numpy.ones((nrows),dtype=numpy.float32) * zeta2Median
    c30 = pyfits.Column(name='zeta2Median',format='1E',array=arr30,unit='arcsec^3')


    coldefs = pyfits.ColDefs([c6, c7, c8, c9, c10,
         c11, c12, c13, c14, c15, c20, c21, c22, c23, c24, c25, c26, c27, c28,
         c29, c30])

    newt2hdu = pyfits.new_table(coldefs)

    # fill new table
    newthdu = pyfits.new_table(valcat.columns)
    newthdu.data = valcat

    # merge the two tables
    newtable = newthdu.columns + newt2hdu.columns
    # for some reason it is in the conversion newtable to newtablehdu that the
    # vignet parameter goes wrong; can check via newtable.columns[0].array vs
    # newtablehdu.data['VIGNET']
    newtablehdu = pyfits.new_table(newtable)
    # try to fix that
    newtablehdu.data['VIGNET'] = valcat['VIGNET']

    # insert table name (NEEDED FOR PSFEX!!!)
    newtablehdu.header.set('EXTNAME', 'LDAC_OBJECTS', 'a name')

    # make output fits file
    prihdr = pyfits.Header()
    prihdu = pyfits.PrimaryHDU(header=prihdr)
    thdulist = pyfits.HDUList([prihdu,psfimhead,newtablehdu])
    thdulist.writeto(valname,clobber=True)

#  run from command line
#
if __name__ == "__main__":

    # collect the options
    options = parser.parse_args()
    aDict = vars(options)
    expid = aDict['expid']

    # do it!
    mkSelPsfCat(**aDict)

