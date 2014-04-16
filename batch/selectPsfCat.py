#!/usr/bin/env python

#
# $Rev:: 177                                                          $:  
# $Author:: roodman                                                   $:  
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
import pdb

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

def mkSelPsfCat(expnum,tag="SVA1_FINALCUT",basedir="/nfs/slac/g/ki/ki22/roodman/dmdata",getIn=True,deleteIn=False):
    """ make a -psfcat.fits catalog with just stars used by PSFex 
    1) call psfex with option to make _out.cat
    2) readin _out.cat pick these stars out of -psfcat.fits
    3) write out selected stars into -selpsfcat.fits
    4) if needed delete the -psfcat.fits catalog
    """

    # make/move to data directory
    rungroup = int(expnum/1000) * 1000
    directory = os.path.join(basedir,tag,"psfcat/%08d/%08d" % (rungroup,expnum))
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)

    if getIn:
        getPsfCat(expnum,tag)

    # run psfex & build selpsfcat
    for i in range(1,62+1):
        filename = "DECam_%08d_%02d_psfcat.fits" % (expnum,i)
        catname = "DECam_%08d_%02d_outcat.fits" % (expnum,i)
        selname = "DECam_%08d_%02d_selpsfcat.fits" % (expnum,i)

        if os.path.exists(filename):
            cmd = "psfex %s -c /u/ec/roodman/Astrophysics/PSF/desdm-plus.psfex -OUTCAT_NAME %s" % (filename,catname)
            print cmd
            os.system(cmd)

            buildSelPsfCat(expnum,i,filename,catname,selname)

            if deleteIn:
                os.remove(filename)


            
def buildSelPsfCat(expnum,iext,filename,catname,selname):
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
    irowArr = numpy.zeros((noutcat),dtype=numpy.int32)

    for irow in range(nrows):
        row = newcat[irow]
        irowArr[irow] = irow
        # call CPD code to calculate moments
        stamp = row['VIGNET']
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
    c9 = pyfits.Column(name='flux',format='1E',array=flux,unit='counts')
    c10 = pyfits.Column(name='delta1',format='1E',array=delta1,unit='arcsec^3')
    c11 = pyfits.Column(name='delta2',format='1E',array=delta2,unit='arcsec^3')
    c12 = pyfits.Column(name='zeta1',format='1E',array=zeta1,unit='arcsec^3')
    c13 = pyfits.Column(name='zeta2',format='1E',array=zeta2,unit='arcsec^3')

    # add some medians and image information too
    e0Median = numpy.median(e0)
    e1Median = numpy.median(e1)
    e2Median = numpy.median(e2)
    delta1Median = numpy.median(delta1)
    delta2Median = numpy.median(delta2)
    zeta1Median = numpy.median(zeta1)
    zeta2Median = numpy.median(zeta2)
    fwhm = 2.355 * numpy.sqrt(e0)
    fwhmMedian = numpy.median(fwhm)

    arr20 = numpy.ones((nrows),dtype=numpy.int32) * expnum
    c20 = pyfits.Column(name='expnum',format='1J',array=arr20)
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

    coldefs = pyfits.ColDefs([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30])
    newt2hdu = pyfits.new_table(coldefs)
                
    # fill new table
    newthdu = pyfits.new_table(newcat.columns)
    newthdu.data = newcat

    # merge the two tables
    newtable = newthdu.columns + newt2hdu.columns
    newtablehdu = pyfits.new_table(newtable)
    
    # make output fits file            
    prihdr = pyfits.Header()
    prihdu = pyfits.PrimaryHDU(header=prihdr)
    thdulist = pyfits.HDUList([prihdu,psfimhead,newtablehdu])
    thdulist.writeto(selname,clobber=True)


def getPsfCat(expnum,tag):
    """ download -psfcat.fits files, assumes we're in the desired directory already
    """

    username = "roodman"
    password = "roo70chips"

    # exposure table has expnum, and id, and this id is called exposureid in the image table
    # image table has exposureid, id, run and imagetype
    # runtag table has the run and tag
    # filepath table has the id and filepath
    # 
    cmd = 'trivialAccess -u %s -p %s -d dessci -c "select f.path from filepath f, image i, exposure e, runtag t where f.id = i.id and i.run = t.run and t.tag = \'%s\' and e.expnum = %d and i.imagetype = \'red\' and i.exposureid = e.id' % (username,password,tag,expnum)

    outname = "tempfilelist_%d.out" % (expnum)    
    cmd = cmd + '" > %s' % (outname)
    print cmd
    subprocess.call(cmd,shell=True)

    lines = [line.rstrip('\r\n') for line in open(outname)]
    junk = lines.pop(0)

    psfcatfiles=[]
    for line in lines:
        parts=line.split('.fits')
        psfcatfiles.append('https://desar2.cosmology.illinois.edu/DESFiles/desardata/' + parts[0]+'_psfcat.fits')
    for i in range(0,len(psfcatfiles)):
        cmd = 'wget --user='+username+' --password='+password+' --no-check-certificate -nv -nc ' + psfcatfiles[i]
        print cmd
        subprocess.call(cmd,shell=True)

       


def filterPsfCat(expnum,fwhmrange=[2.0,30.0],variability=0.2,minsn=20.,maxellip=.3,flagmask=0x00FE):
    """ readin psfcat table for all 62 (or 61) CCDs, filter with desired cuts, and output new table of just the selected stars
    """

    rungroup = int(expnum/1000) * 1000
    basename = "/nfs/slac/g/ki/ki22/roodman/desdm/SVA1-FINALCUT/%08d/%08d/DECam_%08d" % (rungroup,expnum,expnum)
    basename = "DECam_%08d" % (expnum)
    for iCCD in range(1,1+1):
        name = basename + "_%02d_psfcat.fits" % (iCCD)
        
        if os.path.exists(name):
            hdu = pyfits.open(name)
            psfcat = hdu[2].data
            nobjects = psfcat.shape[0]

            # now apply cuts, and collect array of halflight radius
            halflightArr = []
            indexArr = []
            
            halflightpower = numpy.power(psfcat['FLUX_RADIUS'],2)

            result0 = numpy.logical_and(psfcat['SNR_WIN']>minsn,psfcat['ELONGATION']<(1.+maxellip)/(1-maxellip))
            result1 = numpy.logical_and(psfcat['SNR_WIN']<1e8,result0)
            result2 = numpy.logical_and(halflightpower>=fwhmrange[0],halflightpower<fwhmrange[1])
            result3 = numpy.logical_and(result1,result2)
            result4 = numpy.logical_and(result3,psfcat['FLAGS']&flagmask==0)

            # collect objects passing cuts
            for iObj in range(nobjects):
                if result4[iObj]:
                    halflightArr.append(psfcat['FLUX_RADIUS'][iObj])
                    indexArr.append(iObj)

            # now analyze the array of halflight radius
            # first find something akin to the mode and a new min,max

            # sort the values
            indSort = numpy.argsort(halflightArr)

            # find the offset between pairs, nw
            nfwhm = len(halflightArr)
            nw = nfwhm/4
            if nw<4:
                nw = 1

            # loop over pairs, separated by nw, find smallest difference between pairs
            mindiff = 1.e6
            for iPair in range(nfwhm-nw):
                diff = halflightArr[indSort[iPair+nw]] - halflightArr[indSort[iPair]]
                if diff<mindiff:
                    mindiff = diff
                    mode = (halflightArr[indSort[iPair]] + halflightArr[indSort[iPair+nw]])/2.

            # get min,max 
            factor = numpy.power(1.0+variability,1./3.)  #why this odd power!!
            minout = mode/factor
            maxout = mode*factor*factor

            # don't let minout or maxout exceed allowable limits
            # this should never happen, I think

            # now loop through values, selecting only ones between minout,maxout
            # and build mask of desired entries
            result5 = numpy.logical_and(psfcat['FLUX_RADIUS']>minout,psfcat['FLUX_RADIUS']<maxout)
            mask = numpy.logical_and(result4,result5)

            # mask the recordArray
            newcat = psfcat[mask]
                
            # fill new table
            newthdu = pyfits.new_table(newcat.columns)
            newthdu.data = newcat

            # make output fits file
            
            prihdr = pyfits.Header()
            prihdr['MODE'] = mode
            prihdr['MINOUT'] = minout
            prihdr['MAXOUT'] = maxout
            prihdu = pyfits.PrimaryHDU(header=prihdr)
            thdulist = pyfits.HDUList([prihdu,newthdu])
            thdulist.writeto('output.fits')
             
            
        
#  run from command line
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='selectPsfCat')
    parser.add_argument("-e", "--expnum",
                  dest="expnum",
                  type=int,
                  help="sispi exposure number")
    parser.add_argument("-t", "--tag",
                  dest="tag",
                  default="",
                  help="SVA1_FINALCUT or Y1N_FIRSTCUT")
    parser.add_argument("-ng","--getIn",
                    dest="getIn",
                    default=True,action="store_false",
                    help="select to not copy files")
    parser.add_argument("-d","--deleteIn",
                    dest="deleteIn",
                    default=False,action="store_true",
                    help="select to delete input files")

    # collect the options
    options = parser.parse_args()
    aDict = vars(options)  #converts the object options to dictionary of key:value

    # do it!
    mkSelPsfCat(**aDict)

