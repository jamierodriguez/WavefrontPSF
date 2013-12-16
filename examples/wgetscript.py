#! /usr/bin/env python
#
# $Rev::                                                              $:
# $Author::                                                           $:
# $LastChangedDate::                                                  $:
#
import argparse
import os
from scriptUtil import decodeNumberList

parser = argparse.ArgumentParser(prog='wgetscript')
parser.add_argument("-d", "--date", dest="date",type=int,
                  help="date")
parser.add_argument("-rid", "--runit", dest="runid", type=int,
                  help="runID, e.g. 20121203125526")
parser.add_argument("-min", "--minImage", dest="minImage",type=int,default=None,
                  help="minimum image no")
parser.add_argument("-max", "--maxImage", dest="maxImage",type=int,default=None,
                  help="maximum image no")
parser.add_argument("-f", "--inFault",default=False,action="store_true",
                  help="in Fault area")
parser.add_argument("-n", "--imageListString", dest="imageListString",default=None,
                  help="string with image lists, eg. 1:10,20,30:100 ")
parser.add_argument("-i", "--getImage", default=0,type=int,
                  help="download image, or just catalog?")

# unpack options
options = parser.parse_args()


# DTS area
dtsName = "DTS"
if options.inFault:
    dtsName = "DTS_FAULT"


if options.imageListString!=None:
    imageList = decodeNumberList(options.imageListString)
else:
    imageList = range(options.minImage,options.maxImage+1)

'''
e.g.
-d 20121207 -rid 20130117171214 -min 158981 -max 158999
-d 20130217 -rid 20130219111143 -min 178829 -max 178829

'''

for run in imageList:
    # get on the directory, make it if it isn't there
    #
    dataDirectory = "$CPD/catalogs/wgetscript/{0:08d}".format(run)
    dataDirectoryExp = os.path.expandvars(dataDirectory)

    # make directory if it doesn't exist
    if not os.path.exists(dataDirectoryExp):
        os.makedirs(dataDirectoryExp)

    # move there!
    os.chdir(dataDirectoryExp)
    for i in range(1, 63):

        # get cat
        command = "wget --no-check-certificate --http-user=cpd --http-password=cpd70chips -nc -nd -nH -r -k -p -np  --cut-dirs=3 https://desar2.cosmology.illinois.edu:7443/DESFiles/desardata/OPS/red/{0}_{3}/red/DECam_{1:08d}/DECam_{1:08d}_{2:02d}_cat.fits".format(options.runid, run, i, options.date)
        os.system(command)
        if options.getImage:
            # get image
            command = "wget --no-check-certificate --http-user=cpd --http-password=cpd70chips -nc -nd -nH -r -k -p -np  --cut-dirs=3 https://desar2.cosmology.illinois.edu:7443/DESFiles/desardata/OPS/red/{0}_{3}/red/DECam_{1:08d}/DECam_{1:08d}_{2:02d}.fits.fz".format(options.runid, run, i, options.date)
            os.system(command)
            # decompress image
            command = "funpack DECam_{0:08d}_{1:02d}.fits.fz".format(run, i)
            os.system(command)
            # remove old compressed image
            os.remove("DECam_{0:08d}_{1:02d}.fits.fz".format(run, i))
            ###  old command using anon ftp
            ###command = "wget ftp://desar.cosmology.illinois.edu/DESFiles/desardata/%s/src/%d/src/DECam_00%d.fits.fz" % (dtsName,options.date,run)
        else:
            if i == 1:
                # get the first image anyways
                command = "wget --no-check-certificate --http-user=cpd --http-password=cpd70chips -nc -nd -nH -r -k -p -np  --cut-dirs=3 https://desar2.cosmology.illinois.edu:7443/DESFiles/desardata/OPS/red/{0}_{3}/red/DECam_{1:08d}/DECam_{1:08d}_{2:02d}.fits.fz".format(options.runid, run, i, options.date)
                os.system(command)
                # decompress image
                command = "funpack DECam_{0:08d}_{1:02d}.fits.fz".format(run, i)
                os.system(command)
                # remove old compressed image
                os.remove("DECam_{0:08d}_{1:02d}.fits.fz".format(run, i))

