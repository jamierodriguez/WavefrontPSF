#! /usr/bin/env python
#
# $Rev::                                                              $:  
# $Author::                                                           $:  
# $LastChangedDate::                                                  $:  
#
#
#  This scripts has NO DEFAULTS of its own!!!
#  it uses the underlying defaults ONLY
#
import argparse
import os
from scriptUtil import decodeNumberList

parser = argparse.ArgumentParser(prog='submitPSFex')
parser.add_argument("-n", "--imageListString", dest="imageListString",default=None,
                  help="string with image lists, eg. 1:10,20,30:100 ")
parser.add_argument("-f", "--imageInputFile", dest="imageInputFile",default=None,
                  help="file name with csv of images ")
parser.add_argument("-nmin", "--nmin", dest="nmin",type=int,default=0,
                  help="minimum image number")
parser.add_argument("-nmax", "--nmax", dest="nmax",type=int,default=99999999,
                  help="maximum image number")

parser.add_argument("-q", "--queue",
                  dest="queue",type=str,
                  default="long",
                  help="queue to use")
parser.add_argument("-t", "--tag",
                  dest="tag",
                  default="",
                  help="SVA1_FINALCUT or Y1N_FIRSTCUT")
parser.add_argument("-d","--deleteIn",
                    dest="deleteIn",
                    default=False,action="store_true",
                    help="select to delete input files")

# unpack options
options = parser.parse_args()

# get image list
if options.imageListString != None:
    imageList = decodeNumberList(options.imageListString)
elif options.imageInputFile != None:
    inFile = open(options.imageInputFile)
    imageList = []
    for line in inFile.readlines():
        imageStr = line.split(",")
        i = int(imageStr[0])
        if i>= options.nmin and i<=options.nmax:
            imageList.append(i)


optionsString = ""
if options.deleteIn:
    optionsString = optionsString + " -d "


for expnum in imageList:

    command = "bsub -R rhel60 -C 0 -q %s -o logfiles/spc_%08d.log selectPsfCat.py -e %d -t %s %s " % (options.queue,expnum,expnum,options.tag,optionsString)
    print command
    os.system(command)
