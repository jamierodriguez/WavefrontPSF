#!/usr/bin/env python


import argparse
import subprocess
import os
import string


# find the first cut or final cut (or any other tag) corresponsing to a given exposure id
# uses the following DB tables:
# i) filepath, which contains the an id and a corresponding filepath
# ii) image, which also contains the id, and also the DESDM run #
# iii) exposure, which contains the id and also the expnum (ie. the exposure number used online)
# iv) tag, which contains the tag



parser = argparse.ArgumentParser(prog='queryDM')

parser.add_argument("-C", "--ccd", dest="ccd",type=int,default=None,help="CCD ")
parser.add_argument("-e", "--expnum", dest="expnum",type=int,help="Exposure Number")
parser.add_argument("-c", "--catalogOnly",dest="catalogOnly",action="store_true",default=False,
    help="only download catalogs")
parser.add_argument("-b", "--background",dest="background",action="store_true",default=False,
    help="also download background")
parser.add_argument("-p", "--psfcatOnly",dest="psfcatOnly",action="store_true",default=False,
    help="only download psf catalogs")
parser.add_argument("-t", "--tag",dest="tag",default="SVA1_FINALCUT",
    help="desired tag, eg are SVA1_FINALCUT or Y1N_FIRSTCUT")


# unpack options
options = parser.parse_args()

# TODO: Change this!
username = "cpd"
password = "cpd70chips"

# exposure table has expnum, and id, and this id is called exposureid in the image table
# image table has exposureid, id, run and imagetype
# runtag table has the run and tag
# filepath table has the id and filepath
# 
cmd = 'trivialAccess -u %s -p %s -d dessci -c "select f.path from filepath f, image i, exposure e, runtag t where f.id = i.id and i.run = t.run and t.tag = \'%s\' and e.expnum = %d and i.imagetype = \'red\' and i.exposureid = e.id' % (username,password,options.tag,options.expnum)

if (options.ccd is not None):
    cmd = cmd + ' and i.ccd = %d' % (options.ccd)

outname = "tempfilelist_%d.out" % (options.expnum)    
cmd = cmd + '" > %s' % (outname)

print cmd

subprocess.call(cmd,shell=True)

lines = [line.rstrip('\r\n') for line in open(outname)]
junk = lines.pop(0)

# make directory if it doesn't exist
# TODO: Change this!
dataDirectory = "/nfs/slac/g/ki/ki22/roodman/dmdata/%s/%08d" % (options.tag,options.expnum)

if not os.path.exists(dataDirectory):
    os.mkdir(dataDirectory)

# move there!
os.chdir(dataDirectory)



for line in lines:
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
        if (options.catalogOnly) :
            cmd = 'wget --user='+username+' --password='+password+' --no-check-certificate ' + catfiles[i]
        elif (options.psfcatOnly) :
            cmd = 'wget --user='+username+' --password='+password+' --no-check-certificate ' + psfcatfiles[i]
        else :
            cmd = 'wget --user='+username+' --password='+password+' --no-check-certificate ' + imfiles[i] +' ' +catfiles[i]

            if (options.background) :
                cmd = cmd + ' '+bkgfiles[i]

        print cmd

        subprocess.call(cmd,shell=True)

       

   
