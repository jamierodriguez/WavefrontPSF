#
# $Rev::                                                              $:  
# $Author::                                                           $:  
# $LastChangedDate::                                                  $:
#
import numpy
import scipy
import time
import astropy.fits.io as pyfits
from donutlib.PointMesh import PointMesh
from donutlib.decamutil import decaminfo 
from donutlib.decamutil import mosaicinfo
from ROOT import TTree, TFile, gROOT#, TCanvas, gStyle, TGraph2D, TGraphErrors, SetOwnership
from ROOT import SetOwnership
#from hplotlib_cpd import hfillhist
import pdb

class donutana(object):
    """ donutana is a class used to analyze the results from donut fits for the DES experiment.
    """

    def __init__(self,**inputDict):

        # default values for calibration information

        #  see doit.txt file 10/14/2012
        #  comes from 20120914seq1 suplmented by 20121001seq1 for tilts
        #  I've rounded to 2 sig. figures, removed elements to correct for 15deg rotation,
        #  symmetrized, then added the LR sub-matrix to match Zemax,
        #  then finally inverted and rounded.
        #  
        #  hexapodM = 
        #      matrix ([[  0.0e+00,  -2.6e+05,   4.5e+03,   0.0e+00],
	#               [ -2.6e+05,  -0.0e+00,  -0.0e+00,  -4.5e+03],
       	#               [ -5.3e+04,  -0.0e+00,  -0.0e+00,  -9.8e+01],
        #               [  0.0e+00,   5.3e+04,  -9.8e+01,   0.0e+00]])

        hexapodArrayDiagonal = numpy.array(((  0.0e+00,  -2.6e+05,   4.5e+03,   0.0e+00),
                                    ( -2.6e+05,   0.0e+00,   0.0e+00,  -4.5e+03),
                                    ( -5.3e+04,   0.0e+00,   0.0e+00,  -9.8e+01),
                                    (  0.0e+00,   5.3e+04,  -9.8e+01,   0.0e+00)))

        # now replace this with the array from the actual measurements, which
        # still includes the 15deg hexapod rotation

        hexapodArrayRotated =  numpy.array(((-1.9e5, -1.6e5,  4.1e3, -1.3e3),
                                     (-1.6e5,  1.9e5, -1.3e3, -4.1e3),
                                     (-4.8e4,  1.6e4,  -30. ,  -88.),
                                     ( 1.6e4,  4.8e4,  -88. ,  30. )))

        hexapodArray20121020 = numpy.array(((  0.00e+00 ,  1.07e+05 ,  4.54e+03 ,  0.00e+00),
                                            (  1.18e+05 , -0.00e+00 ,  0.00e+00 , -4.20e+03),
                                            ( -4.36e+04 ,  0.00e+00 ,  0.00e+00 , -8.20e+01),
                                            (  0.00e+00 ,  4.42e+04 , -8.10e+01 ,  0.00e+00) ))
                 
        self.paramDict  = {"z4Conversion":172.0,
                           "deltaZSetPoint":0.0,
                           "alignmentMatrix":hexapodArray20121020,
                           "zPointsFile":"",
                           "z5PointsFile":"",
                           "z6PointsFile":"",
                           "z7PointsFile":"",
                           "z8PointsFile":"",
                           "z9PointsFile":"",
                           "z10PointsFile":"",
                           "z11PointsFile":"",
                           "nInterpGrid":8,
                           "interpMethod":"idw",
                           "donutCutString":"",
                           "sensorSet":"FandAOnly",    # or "ScienceOnly" or "Mosaic"
                           "unVignettedOnly":False,
                           "doTrefoil":False,
                           "doSpherical":False,
                           "histFlag":False,
                           "debugFlag":False}
        
        # update paramDict from inputDict
        self.paramDict.update(inputDict)

        # get DECam geometry information
        if self.paramDict["sensorSet"] == "ScienceOnly" or self.paramDict["sensorSet"] == "FandAOnly"  :
            self.infoObj = decaminfo()
        else:
            print "HEY FOOL, set either ScienceOnly or FandAOnly !!!!"
            exit()
            self.infoObj = mosaicinfo()
            
        self.info = self.infoObj.infoDict

        # fill PointMesh coordinate list and gridDict
        self.coordList = []
        self.gridDict = {}

        #  Code for crude vignetting cut - now obsolete
        if self.paramDict["unVignettedOnly"]:
            limitsFA = {"FS1":[0,1024],"FS2":[0,1024],"FS3":[0,1024],"FS4":[0,1024],
                        "FN1":[-1024,0],"FN2":[-1024,0],"FN3":[-1024,0],"FN4":[-1024,0]}
        else:
            limitsFA = {"FS1":[-1024,1024],"FS2":[-1024,1024],"FS3":[-1024,1024],"FS4":[-1024,1024],
                        "FN1":[-1024,1024],"FN2":[-1024,1024],"FN3":[-1024,1024],"FN4":[-1024,1024]}

        # build list of ccds - options are FandAOnly, ScienceOnly, All
        if self.paramDict["sensorSet"] == "FandAOnly" :
            for ccd in self.info.keys():
                ccdinfo = self.info[ccd]
                if  ccdinfo["FAflag"]:
                    self.coordList.append(ccd)
        elif self.paramDict["sensorSet"] == "ScienceOnly":
            for ccd in self.info.keys():
                ccdinfo = self.info[ccd]
                if  not ccdinfo["FAflag"]:
                    self.coordList.append(ccd)
        elif  self.paramDict["sensorSet"] == "Both":
            for ccd in self.info.keys():
                self.coordList.append(ccd)
        else:
            print "donutana.py: HEY FOOL, you have to specify FandAOnly, ScienceOnly, or Both"
            exit()


        # loop over ccds
        for ccd in self.coordList:
            ccdinfo = self.info[ccd]
            # either FA sensors or Science sensors
            if  ( ccdinfo["FAflag"] ):

                xlo = ccdinfo["xCenter"] - 1024 * self.infoObj.mmperpixel
                xhi = ccdinfo["xCenter"] + 1024 * self.infoObj.mmperpixel
                
                ylo = ccdinfo["yCenter"] - 1024 * self.infoObj.mmperpixel
                yhi = ccdinfo["yCenter"] + 1024 * self.infoObj.mmperpixel

                # fill gridDict
                self.gridDict[ccd] = [self.paramDict["nInterpGrid"],ylo,yhi,self.paramDict["nInterpGrid"],xlo,xhi]

            elif ( self.paramDict["sensorSet"] == "ScienceOnly" and not ccdinfo["FAflag"] ):

                xlo = ccdinfo["xCenter"] - 1024 * self.infoObj.mmperpixel
                xhi = ccdinfo["xCenter"] + 1024 * self.infoObj.mmperpixel
                ylo = ccdinfo["yCenter"] - 2048 * self.infoObj.mmperpixel
                yhi = ccdinfo["yCenter"] + 2048 * self.infoObj.mmperpixel

                # fill gridDict
                self.gridDict[ccd] = [self.paramDict["nInterpGrid"],ylo,yhi,self.paramDict["nInterpGrid"],xlo,xhi]

            #elif self.paramDict["sensorSet"] == "Mosaic" :
            #    xlo = ccdinfo["xCenter"] - 1024 * self.infoObj.mmperpixel
            #    xhi = ccdinfo["xCenter"] + 1024 * self.infoObj.mmperpixel
            #    ylo = ccdinfo["yCenter"] - 2048 * self.infoObj.mmperpixel
            #    yhi = ccdinfo["yCenter"] + 2048 * self.infoObj.mmperpixel
            #
            #    # fill gridDict
            #    self.gridDict[ccd] = [self.paramDict["nInterpGrid"],ylo,yhi,self.paramDict["nInterpGrid"],xlo,xhi]

                

        # build the reference meshes
        if self.paramDict["zPointsFile"] != "":
            self.zRefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["zPointsFile"],myMethod=self.paramDict["interpMethod"])
        if self.paramDict["z5PointsFile"] != "":
            self.z5RefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["z5PointsFile"],myMethod=self.paramDict["interpMethod"])
        if self.paramDict["z6PointsFile"] != "":
            self.z6RefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["z6PointsFile"],myMethod=self.paramDict["interpMethod"])
        if self.paramDict["z7PointsFile"] != "":
            self.z7RefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["z7PointsFile"],myMethod=self.paramDict["interpMethod"])
        if self.paramDict["z8PointsFile"] != "":
            self.z8RefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["z8PointsFile"],myMethod=self.paramDict["interpMethod"])
        if self.paramDict["z9PointsFile"] != "":
            self.z9RefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["z9PointsFile"],myMethod=self.paramDict["interpMethod"])
        if self.paramDict["z10PointsFile"] != "":
            self.z10RefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["z10PointsFile"],myMethod=self.paramDict["interpMethod"])
        if self.paramDict["z11PointsFile"] != "":
            self.z11RefMesh = PointMesh(self.coordList,self.gridDict,pointsFile=self.paramDict["z11PointsFile"],myMethod=self.paramDict["interpMethod"])

        # matrix for Hexapod calculation
        self.alignmentMatrix = numpy.matrix(self.paramDict["alignmentMatrix"])
        self.alignmentMatrixSquared = numpy.matrix(self.paramDict["alignmentMatrix"]*self.paramDict["alignmentMatrix"])

        # store some constants
        arcsecperrad = 3600.0 * 180. / numpy.pi
        mmpermicron = 0.001
        self.zangleconv = arcsecperrad * mmpermicron

    def fillPoints(self,dataDict,extraCut=""):
        # fill Points dictionaries for PointMeshes from the list of Donut dictionaries
        zPnts = {}
        z5Pnts = {}
        z6Pnts = {}
        z7Pnts = {}
        z8Pnts = {}
        if self.paramDict["doTrefoil"]:
            z9Pnts = {}
            z10Pnts = {}
        if self.paramDict["doSpherical"]:
            z11Pnts = {}
            

        # need a blank list for each ccd for each of these 5 meshes
        for coord in self.coordList:
            zPnts[coord] = []
            z5Pnts[coord] = []
            z6Pnts[coord] = []
            z7Pnts[coord] = []
            z8Pnts[coord] = []
            if self.paramDict["doTrefoil"]:
                z9Pnts[coord] = []
                z10Pnts[coord] = []
            if self.paramDict["doSpherical"]:
                z11Pnts[coord] = []
                
        
        # fill dicts of Pnts from List of Donut information dictionaries
        for donut in dataDict:

            if type(donut) == dict or type(donut) == pyfits.header.Header:
                try:
                    extname = donut["EXTNAME"]
                except:
                    extname = str(donut["IEXT"]-1) 
                    
                ifile = donut["IFILE"]
                zern4 = donut["ZERN4"]
                zern5 = donut["ZERN5"]
                zern6 = donut["ZERN6"]
                zern7 = donut["ZERN7"]
                zern8 = donut["ZERN8"]
                if self.paramDict["doTrefoil"]:
                    zern9 = donut["ZERN9"]
                    zern10 = donut["ZERN10"]
                ix = donut["IX"]
                iy = donut["IY"]
                chi2 = donut["CHI2"]
##                fitstat = donut["FITSTAT"]
                nele = donut["NELE"]
##                bkgd = donut["BKGD"]
                
            else:
                
#                if type(donut.iext) == float or type(donut.iext) == int :
#                    extname = str(int(donut.iext)-1)   #KLUDGE for MOSAIC
#                elif type(donut.iext) == str:
#                    extname = donut.iext

                extname = donut.extname
                try:
                    ifile = donut.ifile
                except:
                    ifile = 0

                # for focus scans
                try:
                    ifocus = donut.ifocus
                except:
                    ifocus = 0
                    
                zern4 = donut.zern4
                zern5 = donut.zern5 
                zern6 = donut.zern6 
                zern7 = donut.zern7 
                zern8 = donut.zern8 
                zern9 = donut.zern9
                zern10 = donut.zern10
                ix = donut.ix
                iy = donut.iy

                try:
                    chi2 = donut.chi2
                    fitstat = donut.fitstat
                    nele = donut.nele
                    bkgd = donut.bkgd
                except:
                    chi2 = 1.
                    fitstat = 1
                    nele = 1.
                    bkgd = 0.

                try:
                    zern11 = donut.zern11
                except:
                    zern11 = 0.

            # apply cut (could use compile command here)
            good = False
            if self.paramDict["donutCutString"]=="" and extraCut=="":
                good = True
            elif self.paramDict["donutCutString"]!="" and extraCut=="":
                if eval(self.paramDict["donutCutString"]):
                    good = True
            else:
                if eval(self.paramDict["donutCutString"] + " and " + extraCut):
                    good = True

            # good Donut
            if good and zPnts.has_key(extname):
                xval,yval = self.infoObj.getPosition(extname,ix,iy)
                #KLUDGE KLUDGE
                dz = zern4 * self.paramDict["z4Conversion"]
                wgt = 1.0

                # append to relevant list
                zPoint = [xval,yval,dz,wgt]
                zPnts[extname].append(zPoint)
                
                z5Point = [xval,yval,zern5,wgt]
                z5Pnts[extname].append(z5Point)

                z6Point = [xval,yval,zern6,wgt]
                z6Pnts[extname].append(z6Point)

                z7Point = [xval,yval,zern7,wgt]
                z7Pnts[extname].append(z7Point)

                z8Point = [xval,yval,zern8,wgt]
                z8Pnts[extname].append(z8Point)

                if self.paramDict["doTrefoil"]:

                    z9Point = [xval,yval,zern9,wgt]
                    z9Pnts[extname].append(z9Point)
                    
                    z10Point = [xval,yval,zern10,wgt]
                    z10Pnts[extname].append(z10Point)

                if self.paramDict["doSpherical"]:

                    z11Point = [xval,yval,zern11,wgt]
                    z11Pnts[extname].append(z11Point)


        # convert all the lists to numpy arrays
        zPoints = {}
        z5Points = {}
        z6Points = {}
        z7Points = {}
        z8Points = {}
        if self.paramDict["doTrefoil"]:
            z9Points = {}
            z10Points = {}
        if self.paramDict["doSpherical"]:
            z11Points = {}
                            
        for coord in self.coordList:
            zPoints[coord] = numpy.array(zPnts[coord])
            z5Points[coord] = numpy.array(z5Pnts[coord])
            z6Points[coord] = numpy.array(z6Pnts[coord])
            z7Points[coord] = numpy.array(z7Pnts[coord])
            z8Points[coord] = numpy.array(z8Pnts[coord])
            if self.paramDict["doTrefoil"]:
                z9Points[coord] = numpy.array(z9Pnts[coord])
                z10Points[coord] = numpy.array(z10Pnts[coord])
            if self.paramDict["doSpherical"]:
                z11Points[coord] = numpy.array(z11Pnts[coord])

        # return
        if self.paramDict["doTrefoil"] and self.paramDict["doSpherical"]:
            return zPoints,z5Points,z6Points,z7Points,z8Points,z9Points,z10Points,z11Points
        elif self.paramDict["doTrefoil"]:
            return zPoints,z5Points,z6Points,z7Points,z8Points,z9Points,z10Points
        else:
            return zPoints,z5Points,z6Points,z7Points,z8Points

    def makeMeshes(self,donutData,extraCut="",myMethod='none'):
        # make the meshes from the data

        # fill dictionaries of Points, separated by detector
        if self.paramDict["doTrefoil"] and self.paramDict["doSpherical"]:
            zPoints,z5Points,z6Points,z7Points,z8Points,z9Points,z10Points,z11Points = self.fillPoints(donutData,extraCut)
        elif self.paramDict["doTrefoil"]:
            zPoints,z5Points,z6Points,z7Points,z8Points,z9Points,z10Points = self.fillPoints(donutData,extraCut)
        else:
            zPoints,z5Points,z6Points,z7Points,z8Points = self.fillPoints(donutData,extraCut)
        
        # build meshes for the input data (don't need interpolation grids)
        zMesh = PointMesh(self.coordList,self.gridDict,pointsArray=zPoints,myMethod=myMethod)
        z5Mesh = PointMesh(self.coordList,self.gridDict,pointsArray=z5Points,myMethod=myMethod)
        z6Mesh = PointMesh(self.coordList,self.gridDict,pointsArray=z6Points,myMethod=myMethod)
        z7Mesh = PointMesh(self.coordList,self.gridDict,pointsArray=z7Points,myMethod=myMethod)
        z8Mesh = PointMesh(self.coordList,self.gridDict,pointsArray=z8Points,myMethod=myMethod)
        if self.paramDict["doTrefoil"]:
            z9Mesh = PointMesh(self.coordList,self.gridDict,pointsArray=z9Points,myMethod=myMethod)
            z10Mesh = PointMesh(self.coordList,self.gridDict,pointsArray=z10Points,myMethod=myMethod)
        if self.paramDict["doSpherical"]:
            z11Mesh = PointMesh(self.coordList,self.gridDict,pointsArray=z11Points,myMethod=myMethod)
            
        if self.paramDict["doTrefoil"] and self.paramDict["doSpherical"]:
            return zMesh,z5Mesh,z6Mesh,z7Mesh,z8Mesh,z9Mesh,z10Mesh,z11Mesh
        elif self.paramDict["doTrefoil"]:
            return zMesh,z5Mesh,z6Mesh,z7Mesh,z8Mesh,z9Mesh,z10Mesh
        else:
            return zMesh,z5Mesh,z6Mesh,z7Mesh,z8Mesh


    ## def analyzeDonuts(self,donutData,extraCut="",doCull=False,cullCut=0.90):
    ##        # analyzeDonuts takes a list of dictionaries with donut information
    ##        # as input, containing the results of the donut fits,
   
    ##        if self.paramDict["doTrefoil"] and self.paramDict["doSpherical"]:
    ##            zMesh,z5Mesh,z6Mesh,z7Mesh,z8Mesh,z9Mesh,z10Mesh,z11Mesh = self.makeMeshes(donutData,extraCut=extraCut)
    ##        elif self.paramDict["doTrefoil"]:
    ##            zMesh,z5Mesh,z6Mesh,z7Mesh,z8Mesh,z9Mesh,z10Mesh = self.makeMeshes(donutData,extraCut=extraCut)
    ##        else:
    ##            zMesh,z5Mesh,z6Mesh,z7Mesh,z8Mesh = self.makeMeshes(donutData,extraCut=extraCut)
   
    ##        # then we analyze the Zernike polynomials by comparing them to stored references
    ##        # analyzeDonuts returns a dictionary containing deltas and rotation angles for focus, astigmatism and coma
    ##        # as well as the hexapod adjustments
    ##        # it also make a Canvas of plots for study
   
    ##        # fit data meshes to reference meshes
    ##        zResultDict = self.fitToRefMesh(self.zRefMesh,zMesh,self.zangleconv)
    ##        z5ResultDict = self.fitToRefMesh(self.z5RefMesh,z5Mesh)
    ##        z6ResultDict = self.fitToRefMesh(self.z6RefMesh,z6Mesh)
    ##        z7ResultDict = self.fitToRefMesh(self.z7RefMesh,z7Mesh)
    ##        z8ResultDict = self.fitToRefMesh(self.z8RefMesh,z8Mesh)
    ##        if self.paramDict["doTrefoil"]:
    ##            z9ResultDict = self.fitToRefMesh(self.z9RefMesh,z9Mesh)
    ##            z10ResultDict = self.fitToRefMesh(self.z10RefMesh,z10Mesh)
    ##        if self.paramDict["doSpherical"]:
    ##            z11ResultDict = self.fitToRefMesh(self.z11RefMesh,z11Mesh)
   
    ##        # if we want to cull, cull and refit!
    ##        # use the fitted weight to cull - culltype="fit" 
    ##        if doCull:
    ##            dictOfMeshes = {}
    ##            dictOfMeshes["z"] = zMesh
    ##            dictOfMeshes["z5"] = z5Mesh
    ##            dictOfMeshes["z6"] = z6Mesh
    ##            dictOfMeshes["z7"] = z7Mesh
    ##            dictOfMeshes["z8"] = z8Mesh
    ##            if self.paramDict["doTrefoil"]:
    ##                dictOfMeshes["z9"] = z9Mesh
    ##                dictOfMeshes["z10"] = z10Mesh
    ##            if self.paramDict["doSpherical"]:
    ##                dictOfMeshes["z11"] = z11Mesh
    ##                
    ##            # this will take the wgt's from the fit and take their product
    ##            # for each point, and cull at a product of cullCut
    ##            self.cullAllMeshes(dictOfMeshes,cuttype="fit",cullCut=cullCut)
   
    ##            zResultDict = self.fitToRefMesh(self.zRefMesh,zMesh,self.zangleconv)
    ##            z5ResultDict = self.fitToRefMesh(self.z5RefMesh,z5Mesh)
    ##            z6ResultDict = self.fitToRefMesh(self.z6RefMesh,z6Mesh)
    ##            z7ResultDict = self.fitToRefMesh(self.z7RefMesh,z7Mesh)
    ##            z8ResultDict = self.fitToRefMesh(self.z8RefMesh,z8Mesh)
    ##            if self.paramDict["doTrefoil"]:
    ##                z9ResultDict = self.fitToRefMesh(self.z9RefMesh,z9Mesh)
    ##                z10ResultDict = self.fitToRefMesh(self.z10RefMesh,z10Mesh)
    ##            if self.paramDict["doSpherical"]:
    ##                z11ResultDict = self.fitToRefMesh(self.z11RefMesh,z11Mesh)
   
    ##        # make nulls for z9,z10 if doTrefoil is false
    ##        if not self.paramDict["doTrefoil"]:
    ##            z9ResultDict = None
    ##            z10ResultDict = None
    ##        if not self.paramDict["doSpherical"]:
    ##            z11ResultDict = None
    ##            
    ##        # analyze this data and extract the hexapod coefficients
    ##        donutDict = self.calcHexapod(zResultDict,z5ResultDict,z6ResultDict,z7ResultDict,z8ResultDict,z9ResultDict,z10ResultDict,z11ResultDict)
    ##        if len(donutDict)==0:
    ##            goodCalc = False
    ##        else:
    ##            goodCalc = True
   
    ##        # add the individual fit results here too
    ##        donutDict["zResultDict"] = zResultDict
    ##        donutDict["z5ResultDict"] = z5ResultDict
    ##        donutDict["z6ResultDict"] = z6ResultDict
    ##        donutDict["z7ResultDict"] = z7ResultDict
    ##        donutDict["z8ResultDict"] = z8ResultDict
   
    ##        if self.paramDict["doTrefoil"]:
    ##            donutDict["z9ResultDict"] = z9ResultDict
    ##            donutDict["z10ResultDict"] = z10ResultDict
   
    ##        if self.paramDict["doSpherical"]:
    ##            donutDict["z11ResultDict"] = z11ResultDict
   
    ##        # and add the meshes too
    ##        donutDict["zMesh"] = zMesh
    ##        donutDict["z5Mesh"] = z5Mesh
    ##        donutDict["z6Mesh"] = z6Mesh
    ##        donutDict["z7Mesh"] = z7Mesh
    ##        donutDict["z8Mesh"] = z8Mesh
    ##        if self.paramDict["doTrefoil"]:            
    ##            donutDict["z9Mesh"] = z9Mesh
    ##            donutDict["z10Mesh"] = z10Mesh
    ##        if self.paramDict["doSpherical"]:   
    ##            donutDict["z11Mesh"] = z11Mesh
   
    ##        # make a Canvas of plots for this image
    ##        # plot Histogram of Difference before fit, after fit, and after fit vs. X,Y position
    ##        ## if self.paramDict["histFlag"] and zResultDict.has_key("deltaArrayBefore") and goodCalc:
   
    ##        ##     # setup plots
    ##        ##     gStyle.SetStatH(0.32)
    ##        ##     gStyle.SetStatW(0.4)
    ##        ##     gStyle.SetOptStat(1111111)
    ##        ##     gStyle.SetMarkerStyle(20)
    ##        ##     gStyle.SetMarkerSize(0.5)
    ##        ##     gStyle.SetPalette(1)            
    ##        ##     gROOT.ForceStyle()
    ##        ##     
   
    ##        ##     # z plots
    ##        ##     #  
    ##        ##     hZBefore = hfillhist("zBefore","Delta Z, Before Fit",zResultDict["deltaArrayBefore"],200,-200.0,200.0)
    ##        ##     hZAfter = hfillhist("zAfter","Delta Z, After Fit",zResultDict["deltaArrayAfter"],200,-200.0,200.0)
    ##        ##     hZBefore2D = TGraph2D("zBefore2D","Delta Z, Before Fit, vs. Position;X[mm];Y[mm]",zResultDict["deltaArrayBefore"].shape[0],zResultDict["deltaArrayX"],zResultDict["deltaArrayY"],zResultDict["deltaArrayBefore"])
    ##        ##     hZAfter2D = TGraph2D("zAfter2D","Delta Z, After Fit, vs. Position;X[mm];Y[mm]",zResultDict["deltaArrayAfter"].shape[0],zResultDict["deltaArrayX"],zResultDict["deltaArrayY"],zResultDict["deltaArrayAfter"])
   
    ##        ##     nWavesBefore = 1.0
    ##        ##     nWavesAfter = 0.2
    ##        ##     # zern5 plots
    ##        ##     hZ5Before = hfillhist("z5Before","Delta Zern5, Before Fit",z5ResultDict["deltaArrayBefore"],200,-nWavesBefore,nWavesBefore)
    ##        ##     hZ5After = hfillhist("z5After","Delta Zern5, After Fit",z5ResultDict["deltaArrayAfter"],200,-nWavesAfter,nWavesAfter)
    ##        ##     hZ5Before2D = TGraph2D("z5Before2D","Delta Zern5, Before Fit, vs. Position;X[mm];Y[mm]",z5ResultDict["deltaArrayBefore"].shape[0],z5ResultDict["deltaArrayX"],z5ResultDict["deltaArrayY"],z5ResultDict["deltaArrayBefore"])
    ##        ##     hZ5After2D = TGraph2D("z5After2D","Delta Zern5, After Fit, vs. Position;X[mm];Y[mm]",z5ResultDict["deltaArrayAfter"].shape[0],z5ResultDict["deltaArrayX"],z5ResultDict["deltaArrayY"],z5ResultDict["deltaArrayAfter"])
   
    ##        ##     # z6 plots
    ##        ##     hZ6Before = hfillhist("z6Before","Delta Zern6, Before Fit",z6ResultDict["deltaArrayBefore"],200,-nWavesBefore,nWavesBefore)
    ##        ##     hZ6After = hfillhist("z6After","Delta Zern6, After Fit",z6ResultDict["deltaArrayAfter"],200,-nWavesAfter,nWavesAfter)
    ##        ##     hZ6Before2D = TGraph2D("z6Before2D","Delta Zern6, Before Fit, vs. Position;X[mm];Y[mm]",z6ResultDict["deltaArrayBefore"].shape[0],z6ResultDict["deltaArrayX"],z6ResultDict["deltaArrayY"],z6ResultDict["deltaArrayBefore"])
    ##        ##     hZ6After2D = TGraph2D("z6After2D","Delta Zern6, After Fit, vs. Position;X[mm];Y[mm]",z6ResultDict["deltaArrayAfter"].shape[0],z6ResultDict["deltaArrayX"],z6ResultDict["deltaArrayY"],z6ResultDict["deltaArrayAfter"])
   
    ##        ##     # z7 plots
    ##        ##     hZ7Before = hfillhist("z7Before","Delta Zern7, Before Fit",z7ResultDict["deltaArrayBefore"],200,-nWavesBefore,nWavesBefore)
    ##        ##     hZ7After = hfillhist("z7After","Delta Zern7, After Fit",z7ResultDict["deltaArrayAfter"],200,-nWavesAfter,nWavesAfter)
    ##        ##     hZ7Before2D = TGraph2D("z7Before2D","Delta Zern7, Before Fit, vs. Position;X[mm];Y[mm]",z7ResultDict["deltaArrayBefore"].shape[0],z7ResultDict["deltaArrayX"],z7ResultDict["deltaArrayY"],z7ResultDict["deltaArrayBefore"])
    ##        ##     hZ7After2D = TGraph2D("z7After2D","Delta Zern7, After Fit, vs. Position;X[mm];Y[mm]",z7ResultDict["deltaArrayAfter"].shape[0],z7ResultDict["deltaArrayX"],z7ResultDict["deltaArrayY"],z7ResultDict["deltaArrayAfter"])
   
    ##        ##     # z8 plots
    ##        ##     hZ8Before = hfillhist("z8Before","Delta Zern8, Before Fit",z8ResultDict["deltaArrayBefore"],200,-nWavesBefore,nWavesBefore)
    ##        ##     hZ8After = hfillhist("z8After","Delta Zern8, After Fit",z8ResultDict["deltaArrayAfter"],200,-nWavesAfter,nWavesAfter)
    ##        ##     hZ8Before2D = TGraph2D("z8Before2D","Delta Zern8, Before Fit, vs. Position;X[mm];Y[mm]",z8ResultDict["deltaArrayBefore"].shape[0],z8ResultDict["deltaArrayX"],z8ResultDict["deltaArrayY"],z8ResultDict["deltaArrayBefore"])
    ##        ##     hZ8After2D = TGraph2D("z8After2D","Delta Zern8, After Fit, vs. Position;X[mm];Y[mm]",z8ResultDict["deltaArrayAfter"].shape[0],z8ResultDict["deltaArrayX"],z8ResultDict["deltaArrayY"],z8ResultDict["deltaArrayAfter"])
   
    ##        ##     # z9 and z10 plots
    ##        ##     if self.paramDict["doTrefoil"]:
    ##        ##         hZ9Before = hfillhist("z9Before","Delta Zern9, Before Fit",z9ResultDict["deltaArrayBefore"],200,-nWavesBefore,nWavesBefore)
    ##        ##         hZ9After = hfillhist("z9After","Delta Zern9, After Fit",z9ResultDict["deltaArrayAfter"],200,-nWavesAfter,nWavesAfter)
    ##        ##         hZ9Before2D = TGraph2D("z9Before2D","Delta Zern9, Before Fit, vs. Position;X[mm];Y[mm]",z9ResultDict["deltaArrayBefore"].shape[0],z9ResultDict["deltaArrayX"],z9ResultDict["deltaArrayY"],z9ResultDict["deltaArrayBefore"])
    ##        ##         hZ9After2D = TGraph2D("z9After2D","Delta Zern9, After Fit, vs. Position;X[mm];Y[mm]",z9ResultDict["deltaArrayAfter"].shape[0],z9ResultDict["deltaArrayX"],z9ResultDict["deltaArrayY"],z9ResultDict["deltaArrayAfter"])
   
    ##        ##         hZ10Before = hfillhist("z10Before","Delta Zern10, Before Fit",z10ResultDict["deltaArrayBefore"],200,-nWavesBefore,nWavesBefore)
    ##        ##         hZ10After = hfillhist("z10After","Delta Zern10, After Fit",z10ResultDict["deltaArrayAfter"],200,-nWavesAfter,nWavesAfter)
    ##        ##         hZ10Before2D = TGraph2D("z10Before2D","Delta Zern10, Before Fit, vs. Position;X[mm];Y[mm]",z10ResultDict["deltaArrayBefore"].shape[0],z10ResultDict["deltaArrayX"],z10ResultDict["deltaArrayY"],z10ResultDict["deltaArrayBefore"])
    ##        ##         hZ10After2D = TGraph2D("z10After2D","Delta Zern10, After Fit, vs. Position;X[mm];Y[mm]",z10ResultDict["deltaArrayAfter"].shape[0],z10ResultDict["deltaArrayX"],z10ResultDict["deltaArrayY"],z10ResultDict["deltaArrayAfter"])
   
    ##        ##     # z11 plots
    ##        ##     if self.paramDict["doSpherical"]:
    ##        ##         hZ11Before = hfillhist("z11Before","Delta Zern11, Before Fit",z11ResultDict["deltaArrayBefore"],200,-nWavesBefore,nWavesBefore)
    ##        ##         hZ11After = hfillhist("z11After","Delta Zern11, After Fit",z11ResultDict["deltaArrayAfter"],200,-nWavesAfter,nWavesAfter)
    ##        ##         hZ11Before2D = TGraph2D("z11Before2D","Delta Zern11, Before Fit, vs. Position;X[mm];Y[mm]",z11ResultDict["deltaArrayBefore"].shape[0],z11ResultDict["deltaArrayX"],z11ResultDict["deltaArrayY"],z11ResultDict["deltaArrayBefore"])
    ##        ##         hZ11After2D = TGraph2D("z11After2D","Delta Zern11, After Fit, vs. Position;X[mm];Y[mm]",z11ResultDict["deltaArrayAfter"].shape[0],z11ResultDict["deltaArrayX"],z11ResultDict["deltaArrayY"],z11ResultDict["deltaArrayAfter"])
    ##        ##         
    ##        ##     # the Canvas
   
    ##        ##     # unique name for our canvas
    ##        ##     tstr = "canvas" + str(time.time())
   
    ##        ##     if self.paramDict["doTrefoil"]:
    ##        ##         canvas = TCanvas(tstr,tstr,2100,1000)
    ##        ##         canvas.Divide(7,4)
    ##        ##     else:
    ##        ##         canvas = TCanvas(tstr,tstr,1500,1000)
    ##        ##         canvas.Divide(5,4)
   
    ##        ##     # add the plots to a list
    ##        ##     if self.paramDict["doTrefoil"]:
    ##        ##         plotList = [hZBefore,hZ5Before,hZ6Before,hZ7Before,hZ8Before,hZ9Before,hZ10Before,hZAfter,hZ5After,hZ6After,hZ7After,hZ8After,hZ9After,hZ10After,hZBefore2D,hZ5Before2D,hZ6Before2D,hZ7Before2D,hZ8Before2D,hZ9Before2D,hZ10Before2D,hZAfter2D,hZ5After2D,hZ6After2D,hZ7After2D,hZ8After2D,hZ9After2D,hZ10After2D]
    ##        ##     else:
    ##        ##         plotList = [hZBefore,hZ5Before,hZ6Before,hZ7Before,hZ8Before,hZAfter,hZ5After,hZ6After,hZ7After,hZ8After,hZBefore2D,hZ5Before2D,hZ6Before2D,hZ7Before2D,hZ8Before2D,hZAfter2D,hZ5After2D,hZ6After2D,hZ7After2D,hZ8After2D]
   
    ##        ##     # plot em
    ##        ##     if self.paramDict["doTrefoil"]:
    ##        ##         for pad in range(28):
    ##        ##             canvas.cd(pad+1)
    ##        ##             if pad<14 :
    ##        ##                 plotList[pad].Draw()
    ##        ##             else:
    ##        ##                 plotList[pad].Draw("zcolpcol")
    ##        ##     else:
    ##        ##         for pad in range(20):
    ##        ##             canvas.cd(pad+1)
    ##        ##             if pad<10 :
    ##        ##                 plotList[pad].Draw()
    ##        ##             else:
    ##        ##                 plotList[pad].Draw("zcolpcol")
   
    ##        ##     # set it so that python doesn't own these ROOT object
    ##        ##     for plot in plotList:
    ##        ##         SetOwnership(plot,False)
   
    ##        ##     # save canvas in the output Dictionary
    ##        ##     donutDict["canvas"] = canvas
   
    ##        # all done
    ##        return donutDict
            
            

    def calcHexapod(self,zResultDict,z5ResultDict,z6ResultDict,z7ResultDict,z8ResultDict,z9ResultDict,z10ResultDict,z11ResultDict):

        # if any problems, print out error message
        try:

            # build aberration Column vector
            # average the redundant rotations of Astigmatism
            aveZern5ThetaX = 0.5 * (z5ResultDict["thetax"] + z6ResultDict["thetay"])
            aveZern6ThetaX = 0.5 * (z6ResultDict["thetax"] - z5ResultDict["thetay"])

            aveZern5ThetaXErr = numpy.sqrt( 0.25 * (z5ResultDict["thetaxErr"]*z5ResultDict["thetaxErr"] + z6ResultDict["thetayErr"]*z6ResultDict["thetayErr"]) )
            aveZern6ThetaXErr = numpy.sqrt( 0.25 * (z6ResultDict["thetaxErr"]*z6ResultDict["thetaxErr"] + z5ResultDict["thetayErr"]*z5ResultDict["thetayErr"]) )

            # build a weighted average
            try:
                wgtaveZern5ThetaXIErr2 =  ( 1.0/numpy.power(z5ResultDict["thetaxErr"],2) + 1.0/numpy.power(z6ResultDict["thetayErr"],2) )
                wgtaveZern5ThetaX = ( z5ResultDict["thetax"]/numpy.power(z5ResultDict["thetaxErr"],2) + z6ResultDict["thetay"]/numpy.power(z6ResultDict["thetayErr"],2) ) / wgtaveZern5ThetaXIErr2
                wgtaveZern5ThetaXErr = numpy.sqrt(1.0/wgtaveZern5ThetaXIErr2)
                wgtaveZern6ThetaXIErr2 =  ( 1.0/numpy.power(z6ResultDict["thetaxErr"],2) + 1.0/numpy.power(z5ResultDict["thetayErr"],2) )
                wgtaveZern6ThetaX = ( z6ResultDict["thetax"]/numpy.power(z6ResultDict["thetaxErr"],2) - z5ResultDict["thetay"]/numpy.power(z5ResultDict["thetayErr"],2) ) / wgtaveZern6ThetaXIErr2
                wgtaveZern6ThetaXErr = numpy.sqrt(1.0/wgtaveZern6ThetaXIErr2)
            except:
                wgtaveZern5ThetaX = aveZern5ThetaX
                wgtaveZern6ThetaX = aveZern6ThetaX
                wgtaveZern5ThetaXErr = aveZern5ThetaXErr
                wgtaveZern6ThetaXErr = aveZern6ThetaXErr
                
            # use the regular average here
            aberrationList = (aveZern5ThetaX,aveZern6ThetaX,z7ResultDict["delta"],z8ResultDict["delta"])
            aberrationColVec = numpy.matrix(aberrationList).transpose()
            
            # build aberration Error Column vector
            aberrationErrList = (aveZern5ThetaXErr,aveZern6ThetaXErr,z7ResultDict["deltaErr"],z8ResultDict["deltaErr"])
            aberrationErrArr = numpy.array(aberrationErrList)
            aberrationErrSquaredArr = aberrationErrArr * aberrationErrArr
            aberrationErrSquaredColVec = numpy.matrix(aberrationErrSquaredArr).transpose()
            
            # calculate hexapod vector        
            hexapodM = self.alignmentMatrix * aberrationColVec
            hexapodVector = hexapodM.A

            # calculated hexapod vector error
            hexapodErrSquaredM = self.alignmentMatrixSquared * aberrationErrSquaredColVec
            hexapodErrSquaredVector = hexapodErrSquaredM.A
            hexapodErrVector = numpy.sqrt(hexapodErrSquaredVector)
            
            # fill output dictionary (these are now in the Hexapod coordinate system)
            donutDict = {}
            
            donutDict["dodz"] = zResultDict["delta"] - self.paramDict["deltaZSetPoint"]
            donutDict["dodzErr"] = zResultDict["deltaErr"]
            
            donutDict["dodx"] = hexapodVector[0][0]
            donutDict["dody"] = hexapodVector[1][0]
            donutDict["doxt"] = hexapodVector[2][0]
            donutDict["doyt"] = hexapodVector[3][0]
        
            donutDict["dodxErr"] = hexapodErrVector[0][0]
            donutDict["dodyErr"] = hexapodErrVector[1][0]
            donutDict["doxtErr"] = hexapodErrVector[2][0]
            donutDict["doytErr"] = hexapodErrVector[3][0]

            # and for forward compatibility ALSO put this in a sub-dictionary called "donut_summary"
            # but with the MINUS SIGN to make these corrections instead of measurements
            # fill all DB variables into donut_summary also AJR 12/1/2012
            donutSummary = {}

            donutSummary["dodz"] = -(zResultDict["delta"] - self.paramDict["deltaZSetPoint"])
            donutSummary["dodzerr"] = zResultDict["deltaErr"]
            
            donutSummary["dodx"] = -donutDict["dodx"]
            donutSummary["dody"] = -donutDict["dody"]
            donutSummary["doxt"] = -donutDict["doxt"]
            donutSummary["doyt"] = -donutDict["doyt"]
        
            donutSummary["dodxerr"] = donutDict["dodxErr"]
            donutSummary["dodyerr"] = donutDict["dodyErr"]
            donutSummary["doxterr"] = donutDict["doxtErr"]
            donutSummary["doyterr"] = donutDict["doytErr"]
            
            donutSummary["zdelta"] = zResultDict["delta"]
            donutSummary["zthetax"] = zResultDict["thetax"]
            donutSummary["zthetay"] = zResultDict["thetay"]
            donutSummary["zdeltaerr"] = zResultDict["deltaErr"]
            donutSummary["zthetaxerr"] = zResultDict["thetaxErr"]
            donutSummary["zthetayerr"] = zResultDict["thetayErr"]
            donutSummary["zmeandeltabefore"] = zResultDict["meanDeltaBefore"]
            donutSummary["zrmsdeltabefore"] = zResultDict["rmsDeltaBefore"]
            donutSummary["zmeandeltaafter"] = zResultDict["meanDeltaAfter"]
            donutSummary["zrmsdeltaafter"] = zResultDict["rmsDeltaAfter"]

            donutSummary["z5delta"] = z5ResultDict["delta"]
            donutSummary["z5thetax"] = z5ResultDict["thetax"]
            donutSummary["z5thetay"] = z5ResultDict["thetay"]
            donutSummary["z5deltaerr"] = z5ResultDict["deltaErr"]
            donutSummary["z5thetaxerr"] = z5ResultDict["thetaxErr"]
            donutSummary["z5thetayerr"] = z5ResultDict["thetayErr"]
            donutSummary["z5meandeltabefore"] = z5ResultDict["meanDeltaBefore"]
            donutSummary["z5rmsdeltabefore"] = z5ResultDict["rmsDeltaBefore"]
            donutSummary["z5meandeltaafter"] = z5ResultDict["meanDeltaAfter"]
            donutSummary["z5rmsdeltaafter"] = z5ResultDict["rmsDeltaAfter"]

            donutSummary["z6delta"] = z6ResultDict["delta"]
            donutSummary["z6thetax"] = z6ResultDict["thetax"]
            donutSummary["z6thetay"] = z6ResultDict["thetay"]
            donutSummary["z6deltaerr"] = z6ResultDict["deltaErr"]
            donutSummary["z6thetaxerr"] = z6ResultDict["thetaxErr"]
            donutSummary["z6thetayerr"] = z6ResultDict["thetayErr"]
            donutSummary["z6meandeltabefore"] = z6ResultDict["meanDeltaBefore"]
            donutSummary["z6rmsdeltabefore"] = z6ResultDict["rmsDeltaBefore"]
            donutSummary["z6meandeltaafter"] = z6ResultDict["meanDeltaAfter"]
            donutSummary["z6rmsdeltaafter"] = z6ResultDict["rmsDeltaAfter"]
            
            donutSummary["z7delta"] = z7ResultDict["delta"]
            donutSummary["z7thetax"] = z7ResultDict["thetax"]
            donutSummary["z7thetay"] = z7ResultDict["thetay"]
            donutSummary["z7deltaerr"] = z7ResultDict["deltaErr"]
            donutSummary["z7thetaxerr"] = z7ResultDict["thetaxErr"]
            donutSummary["z7thetayerr"] = z7ResultDict["thetayErr"]
            donutSummary["z7meandeltabefore"] = z7ResultDict["meanDeltaBefore"]
            donutSummary["z7rmsdeltabefore"] = z7ResultDict["rmsDeltaBefore"]
            donutSummary["z7meandeltaafter"] = z7ResultDict["meanDeltaAfter"]
            donutSummary["z7rmsdeltaafter"] = z7ResultDict["rmsDeltaAfter"]
            
            donutSummary["z8delta"] = z8ResultDict["delta"]
            donutSummary["z8thetax"] = z8ResultDict["thetax"]
            donutSummary["z8thetay"] = z8ResultDict["thetay"]
            donutSummary["z8deltaerr"] = z8ResultDict["deltaErr"]
            donutSummary["z8thetaxerr"] = z8ResultDict["thetaxErr"]
            donutSummary["z8thetayerr"] = z8ResultDict["thetayErr"]
            donutSummary["z8meandeltabefore"] = z8ResultDict["meanDeltaBefore"]
            donutSummary["z8rmsdeltabefore"] = z8ResultDict["rmsDeltaBefore"]
            donutSummary["z8meandeltaafter"] = z8ResultDict["meanDeltaAfter"]
            donutSummary["z8rmsdeltaafter"] = z8ResultDict["rmsDeltaAfter"]

            if self.paramDict["doTrefoil"]:
                
                donutSummary["z9delta"] = z9ResultDict["delta"]
                donutSummary["z9thetax"] = z9ResultDict["thetax"]
                donutSummary["z9thetay"] = z9ResultDict["thetay"]
                donutSummary["z9deltaerr"] = z9ResultDict["deltaErr"]
                donutSummary["z9thetaxerr"] = z9ResultDict["thetaxErr"]
                donutSummary["z9thetayerr"] = z9ResultDict["thetayErr"]
                donutSummary["z9meandeltabefore"] = z9ResultDict["meanDeltaBefore"]
                donutSummary["z9rmsdeltabefore"] = z9ResultDict["rmsDeltaBefore"]
                donutSummary["z9meandeltaafter"] = z9ResultDict["meanDeltaAfter"]
                donutSummary["z9rmsdeltaafter"] = z9ResultDict["rmsDeltaAfter"]

                donutSummary["z10delta"] = z10ResultDict["delta"]
                donutSummary["z10thetax"] = z10ResultDict["thetax"]
                donutSummary["z10thetay"] = z10ResultDict["thetay"]
                donutSummary["z10deltaerr"] = z10ResultDict["deltaErr"]
                donutSummary["z10thetaxerr"] = z10ResultDict["thetaxErr"]
                donutSummary["z10thetayerr"] = z10ResultDict["thetayErr"]
                donutSummary["z10meandeltabefore"] = z10ResultDict["meanDeltaBefore"]
                donutSummary["z10rmsdeltabefore"] = z10ResultDict["rmsDeltaBefore"]
                donutSummary["z10meandeltaafter"] = z10ResultDict["meanDeltaAfter"]
                donutSummary["z10rmsdeltaafter"] = z10ResultDict["rmsDeltaAfter"]

            else:
                
                donutSummary["z9delta"] = 0.
                donutSummary["z9thetax"] = 0.
                donutSummary["z9thetay"] = 0.
                donutSummary["z9deltaerr"] = 0.
                donutSummary["z9thetaxerr"] = 0.
                donutSummary["z9thetayerr"] = 0.
                donutSummary["z9meandeltabefore"] = 0.
                donutSummary["z9rmsdeltabefore"] = 0.
                donutSummary["z9meandeltaafter"] = 0.
                donutSummary["z9rmsdeltaafter"] = 0.

                donutSummary["z10delta"] = 0.
                donutSummary["z10thetax"] = 0.
                donutSummary["z10thetay"] = 0.
                donutSummary["z10deltaerr"] = 0.
                donutSummary["z10thetaxerr"] = 0.
                donutSummary["z10thetayerr"] = 0.
                donutSummary["z10meandeltabefore"] = 0.
                donutSummary["z10rmsdeltabefore"] = 0.
                donutSummary["z10meandeltaafter"] = 0.
                donutSummary["z10rmsdeltaafter"] = 0.

            # do Spherical if desired
            if self.paramDict["doSpherical"]:

                donutSummary["z11delta"] = z11ResultDict["delta"]
                donutSummary["z11thetax"] = z11ResultDict["thetax"]
                donutSummary["z11thetay"] = z11ResultDict["thetay"]
                donutSummary["z11deltaerr"] = z11ResultDict["deltaErr"]
                donutSummary["z11thetaxerr"] = z11ResultDict["thetaxErr"]
                donutSummary["z11thetayerr"] = z11ResultDict["thetayErr"]
                donutSummary["z11meandeltabefore"] = z11ResultDict["meanDeltaBefore"]
                donutSummary["z11rmsdeltabefore"] = z11ResultDict["rmsDeltaBefore"]
                donutSummary["z11meandeltaafter"] = z11ResultDict["meanDeltaAfter"]
                donutSummary["z11rmsdeltaafter"] = z11ResultDict["rmsDeltaAfter"]
                
            else:
                
                donutSummary["z11delta"] = 0.
                donutSummary["z11thetax"] = 0.
                donutSummary["z11thetay"] = 0.
                donutSummary["z11deltaerr"] = 0.
                donutSummary["z11thetaxerr"] = 0.
                donutSummary["z11thetayerr"] = 0.
                donutSummary["z11meandeltabefore"] = 0.
                donutSummary["z11rmsdeltabefore"] = 0.
                donutSummary["z11meandeltaafter"] = 0.
                donutSummary["z11rmsdeltaafter"] = 0.

            # get number of donuts ultimately used
            ndonuts_used = len(zResultDict["deltaArrayAfter"])
            donutSummary["ndonuts_used"] = ndonuts_used

            # store in the output dictionary
            donutDict["donut_summary"] = donutSummary

        except:
            print "donutana: Not enough information for calcHexapod"
            donutDict = {}


        return donutDict
        
        
        

    def fitToRefMesh(self,refMesh,otherMesh,angleconversion=1.0):
        # fit a mesh to a Reference mesh, return the fit parameters
        # and errors and convenient arrays for plots

        # resultsDict contents
        #      delta fit result
        #      thetax fit result
        #      thetay fit result
        #      error on delta fit result
        #      error on thetax fit result
        #      error on thetay fit result
        #      mean of differences before fit
        #      sigma of residual distribution
        #      array of differences before the fit
        #      array of differences after the fit

        resultDict = {}
                                     
        # fit the two meshes
        results,zDiff,xColumn,yColumn = refMesh.fitMesh(otherMesh,"RLM")

        if results!=None:
            # fill resultsDict
            resultDict["delta"] = results.params[2]
            resultDict["deltaErr"] = results.bse[2]
            
            # convert from [Values/Dictance] to another unit
            resultDict["thetax"] = results.params[0]*angleconversion
            resultDict["thetay"] = results.params[1]*angleconversion
            resultDict["thetaxErr"] = results.bse[0]*angleconversion
            resultDict["thetayErr"] = results.bse[1]*angleconversion

            # mean delta before the fit - no truncation now
            resultDict["meanDeltaBefore"] = scipy.mean(zDiff)
            resultDict["rmsDeltaBefore"] = scipy.std(zDiff)
            
            # rms of Differences after fit
            resultDict["meanDeltaAfter"] = scipy.mean(results.resid)
            resultDict["rmsDeltaAfter"] = scipy.std(results.resid)

            # array of differences before the fit
            resultDict["deltaArrayBefore"] = zDiff            

            # array of differences after the fit
            resultDict["deltaArrayAfter"] = results.resid
            resultDict["deltaArrayX"] = xColumn
            resultDict["deltaArrayY"] = yColumn
            
        return resultDict


    def cullAllMeshes(self,dictOfMeshes,cutval=3.0,cutval2=50.,cuttype="WgtNSig",cullCut=0.01):
        """ utility routine to cull a dictionary of Meshes, where all Meshes are of the same donuts """
        # first calculate nSigma or DtoInterp or  for each mesh
        for mesh in dictOfMeshes:
            thisMesh = dictOfMeshes[mesh]
            if cuttype=="WgtNSig":
                thisMesh.calcWgtNSig(nsigma=cutval)
            elif cuttype=="WgtDtoInterp":
                if mesh=="z":
                    thisMesh.calcWgtDtoInterp(maxdist=cutval2)
                else:
                    thisMesh.calcWgtDtoInterp(maxdist=cutval)
                

        # now we need to loop over the Coords
        # and coalesce the Wgts of the points

        aMesh = dictOfMeshes[dictOfMeshes.keys()[0]]   # this is just the first mesh
        dictOfWgts = {}
        for coord in aMesh.coordList:
            npts = aMesh.pointsArray[coord].shape[0]
            if npts>0 :
                thisWgt = numpy.ones(npts)
                # combine weights with the product 
                for mesh in dictOfMeshes:
                    thisMesh = dictOfMeshes[mesh]
                    thisWgt = thisWgt * thisMesh.pointsArray[coord][:,3]

                # now insert this weight for all meshes
                for mesh in dictOfMeshes:
                    thisMesh = dictOfMeshes[mesh]
                    thisMesh.pointsArray[coord][:,3] = thisWgt

        # finally cull each mesh!
        for mesh in dictOfMeshes:
            thisMesh = dictOfMeshes[mesh]
            thisMesh.cullMesh(cullCut)

        return 0

    def adjustAllMeshes(self,donutDict):
        """ utility routine to adjust all the meshes in a dict from analyzeDonut
        """
        
        zMesh = donutDict["zMesh"]
        zResultMesh = donutDict["zResultDict"]
        zMesh.adjustMesh(zResultMesh["thetax"],zResultMesh["thetay"],zResultMesh["delta"],self.zangleconv)

        z5Mesh = donutDict["z5Mesh"]
        z5ResultMesh = donutDict["z5ResultDict"]
        z5Mesh.adjustMesh(z5ResultMesh["thetax"],z5ResultMesh["thetay"],z5ResultMesh["delta"])
        z6Mesh = donutDict["z6Mesh"]
        z6ResultMesh = donutDict["z6ResultDict"]
        z6Mesh.adjustMesh(z6ResultMesh["thetax"],z6ResultMesh["thetay"],z6ResultMesh["delta"])
        z7Mesh = donutDict["z7Mesh"]
        z7ResultMesh = donutDict["z7ResultDict"]
        z7Mesh.adjustMesh(z7ResultMesh["thetax"],z7ResultMesh["thetay"],z7ResultMesh["delta"])
        z8Mesh = donutDict["z8Mesh"]
        z8ResultMesh = donutDict["z8ResultDict"]
        z8Mesh.adjustMesh(z8ResultMesh["thetax"],z8ResultMesh["thetay"],z8ResultMesh["delta"])
        if self.paramDict["doTrefoil"]:
            z9Mesh = donutDict["z9Mesh"]
            z9ResultMesh = donutDict["z9ResultDict"]
            z9Mesh.adjustMesh(z9ResultMesh["thetax"],z9ResultMesh["thetay"],z9ResultMesh["delta"])
            z10Mesh = donutDict["z10Mesh"]
            z10ResultMesh = donutDict["z10ResultDict"]
            z10Mesh.adjustMesh(z10ResultMesh["thetax"],z10ResultMesh["thetay"],z10ResultMesh["delta"])
        if self.paramDict["doSpherical"]:
            z11Mesh = donutDict["z11Mesh"]
            z11ResultMesh = donutDict["z11ResultDict"]
            z11Mesh.adjustMesh(z11ResultMesh["thetax"],z11ResultMesh["thetay"],z11ResultMesh["delta"])


    def anaMesh(self,aMesh,mesh="RLM"):
        """ utility routine to analyze one mesh: find its mean and x,y slopes
        """

        # mean should be ok!
        X,Y,Z = aMesh.getXYZpoints(self.coordList)
        delta = numpy.mean(Z)

        # column with just 1
        npoints = Z.shape[0]
        oneColumn = numpy.ones(npoints)

        # get slopes using a robust linear fit

        # first do Z vs. X to get thetaY
        # formula to fit is ThetaY*x + DeltaZ
        aMatrix = numpy.vstack([X,oneColumn]).T

        # make sure there is something to fit
        if npoints>3:
            # use Regression method or use sm.RLM for robust fitting
            if method=="OLS":
                linearModel = sm.OLS(Z,aMatrix)
            elif method=="RLM":
                linearModel = sm.RLM(Z,aMatrix)
            
            try:
                results = linearModel.fit()
                print "anaMesh: ThetaY = ",results.params[0]," +- ",results.bse[0]
                print "PrintMesh: DeltaZ = ",results.params[1]," +- ",results.bse[1]
                thetaY = results.params[0]
                deltaYfit = results.params[0]

            except:
                print "PointMesh.py:  fit failed"
                results = None
                thetaY = None
                deltaYfit = None
                                
        else:
            results = None
            thetaY = None
            deltaYfit = None


        # next do Z vs. Y to get thetaX
        # formula to fit is ThetaX*y + DeltaZ
        aMatrix = numpy.vstack([Y,oneColumn]).T

        # make sure there is something to fit
        if npoints>3:
            # use Regression method or use sm.RLM for robust fitting
            if method=="OLS":
                linearModel = sm.OLS(Z,aMatrix)
            elif method=="RLM":
                linearModel = sm.RLM(Z,aMatrix)
            
            try:
                results = linearModel.fit()
                print "anaMesh: ThetaX = ",results.params[0]," +- ",results.bse[0]
                print "PrintMesh: DeltaZ = ",results.params[1]," +- ",results.bse[1]
                thetaX = results.params[0]
                deltaXfit = results.params[0]

            except:
                print "PointMesh.py:  fit failed"
                results = None
                thetaX = None
                deltaXfit = None
                                
        else:
            results = None
            thetaX = None
            deltaXfit = None

        return {"delta":delta,"thetaX":thetaX,"thetaY":thetaY}


    def analyzeAllMeshes(self,donutDict):
        """ utility routine to analyze the meshes in the dict: to find the mean and x,y slopes
        """
        anaResults = {}

        zMesh = donutDict["zMesh"]
        anaResults["zMesh"] = anaMesh(zMesh)

        for i in range(4,11+1):
            meshName = "%dMesh" % (i)
            if donutDict.has_key(meshName):
                aMesh = donutDict[meshName]
                anaResults[meshName] = anaMesh(aMesh)

    def mergeAllMeshes(self,donutDict,donutDictOther):
        """ utility routine to adjust all the meshes in a dict from analyzeDonut
        """
        
        zMesh = donutDict["zMesh"]
        zMeshOther = donutDictOther["zMesh"]
        zMesh.mergeMesh(zMeshOther)

        z5Mesh = donutDict["z5Mesh"]
        z5MeshOther = donutDictOther["z5Mesh"]
        z5Mesh.mergeMesh(z5MeshOther)

        z6Mesh = donutDict["z6Mesh"]
        z6MeshOther = donutDictOther["z6Mesh"]
        z6Mesh.mergeMesh(z6MeshOther)

        z7Mesh = donutDict["z7Mesh"]
        z7MeshOther = donutDictOther["z7Mesh"]
        z7Mesh.mergeMesh(z7MeshOther)

        z8Mesh = donutDict["z8Mesh"]
        z8MeshOther = donutDictOther["z8Mesh"]
        z8Mesh.mergeMesh(z8MeshOther)

        if self.paramDict["doTrefoil"]:
        
            z9Mesh = donutDict["z9Mesh"]
            z9MeshOther = donutDictOther["z9Mesh"]
            z9Mesh.mergeMesh(z9MeshOther)
            
            z10Mesh = donutDict["z10Mesh"]
            z10MeshOther = donutDictOther["z10Mesh"]
            z10Mesh.mergeMesh(z10MeshOther)

        if self.paramDict["doSpherical"]:
        
            z11Mesh = donutDict["z11Mesh"]
            z11MeshOther = donutDictOther["z11Mesh"]
            z11Mesh.mergeMesh(z11MeshOther)



    def writeAllMeshes(self,donutDict,fileName="test"):        
        """ write all meshes to .dat files
        """

        zMesh = donutDict["zMesh"]
        zMesh.writePointsToFile("z4Mesh"+fileName+".dat")

        z5Mesh = donutDict["z5Mesh"]
        z5Mesh.writePointsToFile("z5Mesh"+fileName+".dat")
        z6Mesh = donutDict["z6Mesh"]
        z6Mesh.writePointsToFile("z6Mesh"+fileName+".dat")
        z7Mesh = donutDict["z7Mesh"]
        z7Mesh.writePointsToFile("z7Mesh"+fileName+".dat")
        z8Mesh = donutDict["z8Mesh"]
        z8Mesh.writePointsToFile("z8Mesh"+fileName+".dat")

        if self.paramDict["doTrefoil"]:
            z9Mesh = donutDict["z9Mesh"]
            z9Mesh.writePointsToFile("z9Mesh"+fileName+".dat")
            z10Mesh = donutDict["z10Mesh"]
            z10Mesh.writePointsToFile("z10Mesh"+fileName+".dat")

        if self.paramDict["doSpherical"]:
            z11Mesh = donutDict["z11Mesh"]
            z11Mesh.writePointsToFile("z11Mesh"+fileName+".dat")
            

