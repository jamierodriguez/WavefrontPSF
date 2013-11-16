#!/usr/bin/env python
# $Rev:: 156                                                          $:  
# $Author:: roodman                                                   $:  
# $LastChangedDate:: 2013-07-02 16:42:13 -0700 (Tue, 02 Jul 2013)     $: 
#
# Make one calculated image, either from Zemax Zernike array
# or from Zemax WFM file
#
import numpy
import astropy.fits.io as pyfits
import sys
from donutlib.donutengine import donutengine


class makedonut(object):

    def __init__(self,**inputDict):

        # initialize the parameter Dictionary, and update defaults from inputDict
        self.paramDict = {"inputFile":"",
                     "writeToFits":False,
                     "outputPrefix":"testone",
                     "xDECam":0.0,
                     "yDECam":0.0,
                     "debugFlag":False,
                     "rootFlag":False,
                     "printLevel":0,
                     "iTelescope":0,
                     "waveLength":700.0e-9,
                     "nZernikeTerms":11,
                     "nbin":128,
                     "nPixels":32,
                     "gridCalcMode":True,
                     "pixelOverSample":4,
                     "scaleFactor":2.,                 
                     "randomFlag":False,
                     "gain":1.0}

        self.paramDict.update(inputDict)

        # declare fit function
        self.gFitFunc = donutengine(**self.paramDict)

        # check values of bins - prevent errors
        if self.gFitFunc._nbin != self.gFitFunc._nPixels * self.gFitFunc._pixelOverSample:
            print "makedonut:  ERROR in values of nbin,nPixels,pixelOverSample!!!"
            sys.exit(0)

        # also require that _Lu > 2R, ie. that pupil fits in pupil plane
        # this translates into requiring that (Lambda F / pixelSize) * pixelOverSample * scaleFactor > 1
        if self.gFitFunc._pixelOverSample * self.gFitFunc.getScaleFactor() * (700.e-9 * 2.9 / 15.e-6) < 1. :
            print "makedonut:  ERROR pupil doesn't fit!!!"
            sys.exit(0)
            


    def make(self,inputZernikeArray,rzero=0.125,nEle=1.e6,background=4000.0,xDECam=0.,yDECam=0.):

        # parameters for Donuts
        par = numpy.zeros(self.gFitFunc.npar)
        par[self.gFitFunc.ipar_rzero] = rzero
        par[self.gFitFunc.ipar_nEle] = nEle
        par[self.gFitFunc.ipar_bkgd] = background

        for iZ in range(len(inputZernikeArray)-1):            
            par[self.gFitFunc.ipar_ZernikeFirst+iZ] += inputZernikeArray[iZ+1] 

        self.gFitFunc.setXYDECam(xDECam,yDECam)

        # now make the Donut
        self.gFitFunc.calcAll(par)

        # randomize
        theImage = self.gFitFunc.getvImage().copy()
        if self.paramDict["randomFlag"]:
            postageshape = theImage.shape
            nranval = numpy.random.normal(0.0,1.0,postageshape)
            imarr = theImage + nranval*numpy.sqrt(theImage)
        else:
            imarr = theImage

        # apply gain
        if self.paramDict["gain"] != 1.0:
            imarr = imarr / self.paramDict["gain"]

        # make sure that imarr has dtype = float32
        imarr = numpy.float64(imarr)

        #output the calculated image
        # calculated Donut
        if self.paramDict["writeToFits"]:
            hdu = pyfits.PrimaryHDU(imarr)
            prihdr =  hdu.header

            prihdr.update("SCALE",0.270,"Arsec/pixel")
            prihdr.update("XDECAM",self.paramDict["xDECam"],"Target xposition (mm) in focal plane")
            prihdr.update("YDECAM",self.paramDict["yDECam"],"Target yposition (mm) in focal plane")
            prihdr.update("FILTER",3,"Filter number 1-6=ugrizY")
            prihdr.update("FILTNAME","r","Filter name")
                  
            prihdr.update("GAIN",self.paramDict["gain"],"Np.e./ADU")
            prihdr.update("nEle",par[self.gFitFunc.ipar_nEle],"Number of photo-electrons")
            prihdr.update("rzero",par[self.gFitFunc.ipar_rzero],"Fried Parameters [m]")
            prihdr.update("bkgd",par[self.gFitFunc.ipar_bkgd],"Background")
        
        else:
            return imarr

