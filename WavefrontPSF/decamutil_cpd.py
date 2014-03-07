#
# $Rev::                                                              $:
# $Author::                                                           $:
# $LastChangedDate::                                                  $:
#
# Utility methods for DECam
#

import numpy as np

class decaminfo(object):
    """ decaminfo is a class used to contain DECam geometry information and various utility routines
    """

    def info(self):
        # info returns a dictionary chock full of info on the DECam geometry
        # keyed by the CCD name

        infoDict = {}

        # store a dictionary for each CCD, keyed by the CCD name
        # AJR 9/14/2012 fixed these to agree with the DS9 coordinate system
        infoDict["S1"] =  {"xCenter":  -16.908,"yCenter":-191.670, "FAflag":False, "CCDNUM":25}
        infoDict["S2"]  = {"xCenter":  -16.908,"yCenter":-127.780, "FAflag":False, "CCDNUM":26}
        infoDict["S3"]  = {"xCenter":  -16.908,"yCenter": -63.890, "FAflag":False, "CCDNUM":27}
        infoDict["S4"]  = {"xCenter":  -16.908,"yCenter":   0.000, "FAflag":False, "CCDNUM":28}
        infoDict["S5"]  = {"xCenter":  -16.908,"yCenter":  63.890, "FAflag":False, "CCDNUM":29}
        infoDict["S6"]  = {"xCenter":  -16.908,"yCenter": 127.780, "FAflag":False, "CCDNUM":30}
        infoDict["S7"]  = {"xCenter":  -16.908,"yCenter": 191.670, "FAflag":False, "CCDNUM":31}
        infoDict["S8"]  = {"xCenter":  -50.724,"yCenter":-159.725, "FAflag":False, "CCDNUM":19}
        infoDict["S9"]  = {"xCenter":  -50.724,"yCenter": -95.835, "FAflag":False, "CCDNUM":20}
        infoDict["S10"] = {"xCenter":  -50.724,"yCenter": -31.945, "FAflag":False, "CCDNUM":21}
        infoDict["S11"] = {"xCenter":  -50.724,"yCenter":  31.945, "FAflag":False, "CCDNUM":22}
        infoDict["S12"] = {"xCenter":  -50.724,"yCenter":  95.835, "FAflag":False, "CCDNUM":23}
        infoDict["S13"] = {"xCenter":  -50.724,"yCenter": 159.725, "FAflag":False, "CCDNUM":24}
        infoDict["S14"] = {"xCenter":  -84.540,"yCenter":-159.725, "FAflag":False, "CCDNUM":13}
        infoDict["S15"] = {"xCenter":  -84.540,"yCenter": -95.835, "FAflag":False, "CCDNUM":14}
        infoDict["S16"] = {"xCenter":  -84.540,"yCenter": -31.945, "FAflag":False, "CCDNUM":15}
        infoDict["S17"] = {"xCenter":  -84.540,"yCenter":  31.945, "FAflag":False, "CCDNUM":16}
        infoDict["S18"] = {"xCenter":  -84.540,"yCenter":  95.835, "FAflag":False, "CCDNUM":17}
        infoDict["S19"] = {"xCenter":  -84.540,"yCenter": 159.725, "FAflag":False, "CCDNUM":18}
        infoDict["S20"] = {"xCenter": -118.356,"yCenter":-127.780, "FAflag":False, "CCDNUM":8 }
        infoDict["S21"] = {"xCenter": -118.356,"yCenter": -63.890, "FAflag":False, "CCDNUM":9 }
        infoDict["S22"] = {"xCenter": -118.356,"yCenter":   0.000, "FAflag":False, "CCDNUM":10}
        infoDict["S23"] = {"xCenter": -118.356,"yCenter":  63.890, "FAflag":False, "CCDNUM":11}
        infoDict["S24"] = {"xCenter": -118.356,"yCenter": 127.780, "FAflag":False, "CCDNUM":12}
        infoDict["S25"] = {"xCenter": -152.172,"yCenter": -95.835, "FAflag":False, "CCDNUM":4 }
        infoDict["S26"] = {"xCenter": -152.172,"yCenter": -31.945, "FAflag":False, "CCDNUM":5 }
        infoDict["S27"] = {"xCenter": -152.172,"yCenter":  31.945, "FAflag":False, "CCDNUM":6 }
        infoDict["S28"] = {"xCenter": -152.172,"yCenter":  95.835, "FAflag":False, "CCDNUM":7 }
        infoDict["S29"] = {"xCenter": -185.988,"yCenter": -63.890, "FAflag":False, "CCDNUM":1 }
        infoDict["S30"] = {"xCenter": -185.988,"yCenter":   0.000, "FAflag":False, "CCDNUM":2 }
        infoDict["S31"] = {"xCenter": -185.988,"yCenter":  63.890, "FAflag":False, "CCDNUM":3 }
        infoDict["N1"]  = {"xCenter": 16.908,  "yCenter":-191.670, "FAflag":False, "CCDNUM":32}
        infoDict["N2"]  = {"xCenter": 16.908,  "yCenter":-127.780, "FAflag":False, "CCDNUM":33}
        infoDict["N3"]  = {"xCenter": 16.908,  "yCenter": -63.890, "FAflag":False, "CCDNUM":34}
        infoDict["N4"]  = {"xCenter": 16.908,  "yCenter":   0.000, "FAflag":False, "CCDNUM":35}
        infoDict["N5"]  = {"xCenter": 16.908,  "yCenter":  63.890, "FAflag":False, "CCDNUM":36}
        infoDict["N6"]  = {"xCenter": 16.908,  "yCenter": 127.780, "FAflag":False, "CCDNUM":37}
        infoDict["N7"]  = {"xCenter": 16.908,  "yCenter": 191.670, "FAflag":False, "CCDNUM":38}
        infoDict["N8"]  = {"xCenter": 50.724,  "yCenter":-159.725, "FAflag":False, "CCDNUM":39}
        infoDict["N9"]  = {"xCenter": 50.724,  "yCenter": -95.835, "FAflag":False, "CCDNUM":40}
        infoDict["N10"] = {"xCenter": 50.724,  "yCenter": -31.945, "FAflag":False, "CCDNUM":41}
        infoDict["N11"] = {"xCenter": 50.724,  "yCenter":  31.945, "FAflag":False, "CCDNUM":42}
        infoDict["N12"] = {"xCenter": 50.724,  "yCenter":  95.835, "FAflag":False, "CCDNUM":43}
        infoDict["N13"] = {"xCenter": 50.724,  "yCenter": 159.725, "FAflag":False, "CCDNUM":44}
        infoDict["N14"] = {"xCenter": 84.540,  "yCenter":-159.725, "FAflag":False, "CCDNUM":45}
        infoDict["N15"] = {"xCenter": 84.540,  "yCenter": -95.835, "FAflag":False, "CCDNUM":46}
        infoDict["N16"] = {"xCenter": 84.540,  "yCenter": -31.945, "FAflag":False, "CCDNUM":47}
        infoDict["N17"] = {"xCenter": 84.540,  "yCenter":  31.945, "FAflag":False, "CCDNUM":48}
        infoDict["N18"] = {"xCenter": 84.540,  "yCenter":  95.835, "FAflag":False, "CCDNUM":49}
        infoDict["N19"] = {"xCenter": 84.540,  "yCenter": 159.725, "FAflag":False, "CCDNUM":50}
        infoDict["N20"] = {"xCenter": 118.356, "yCenter":-127.780, "FAflag":False, "CCDNUM":51}
        infoDict["N21"] = {"xCenter": 118.356, "yCenter": -63.890, "FAflag":False, "CCDNUM":52}
        infoDict["N22"] = {"xCenter": 118.356, "yCenter":   0.000, "FAflag":False, "CCDNUM":53}
        infoDict["N23"] = {"xCenter": 118.356, "yCenter":  63.890, "FAflag":False, "CCDNUM":54}
        infoDict["N24"] = {"xCenter": 118.356, "yCenter": 127.780, "FAflag":False, "CCDNUM":55}
        infoDict["N25"] = {"xCenter": 152.172, "yCenter": -95.835, "FAflag":False, "CCDNUM":56}
        infoDict["N26"] = {"xCenter": 152.172, "yCenter": -31.945, "FAflag":False, "CCDNUM":57}
        infoDict["N27"] = {"xCenter": 152.172, "yCenter":  31.945, "FAflag":False, "CCDNUM":58}
        infoDict["N28"] = {"xCenter": 152.172, "yCenter":  95.835, "FAflag":False, "CCDNUM":59}
        infoDict["N29"] = {"xCenter": 185.988, "yCenter": -63.890, "FAflag":False, "CCDNUM":60}
        infoDict["N30"] = {"xCenter": 185.988, "yCenter":   0.000, "FAflag":False, "CCDNUM":61}
        infoDict["N31"] = {"xCenter": 185.988, "yCenter":  63.890, "FAflag":False, "CCDNUM":62}
        infoDict["FS1"] = {"xCenter": -152.172,"yCenter": 143.7525,"FAflag":True , "CCDNUM":66}
        infoDict["FS2"] = {"xCenter": -185.988,"yCenter": 111.8075,"FAflag":True , "CCDNUM":65}
        infoDict["FS3"] = {"xCenter": -219.804,"yCenter":  15.9725,"FAflag":True , "CCDNUM":63}
        infoDict["FS4"] = {"xCenter": -219.804,"yCenter": -15.9725,"FAflag":True , "CCDNUM":64}
        infoDict["FN1"] = {"xCenter": 152.172, "yCenter": 143.7525,"FAflag":True , "CCDNUM":67}
        infoDict["FN2"] = {"xCenter": 185.988, "yCenter": 111.8075,"FAflag":True , "CCDNUM":68}
        infoDict["FN3"] = {"xCenter": 219.804, "yCenter":  15.9725,"FAflag":True , "CCDNUM":69}
        infoDict["FN4"] = {"xCenter": 219.804, "yCenter": -15.9725,"FAflag":True , "CCDNUM":70}

        return infoDict


    def __init__(self,**inputDict):

        self.infoDict = self.info()
        self.mmperpixel = 0.015

        # ccddict returns the chip name when given a chip number
        self.ccddict = {}
        for keyi in self.infoDict.keys():
            self.ccddict.update(
                {self.infoDict[keyi]['CCDNUM']: keyi}
                )

    def getPosition(self,extname,ix,iy):
        # return the x,y position in [mm] for a given CCD and pixel number
        # note that the ix,iy are Image pixels - overscans removed - and start at zero

        ccdinfo = self.infoDict[extname]

        # CCD size in pixels
        if ccdinfo["FAflag"]:
            xpixHalfSize = 1024.
            ypixHalfSize = 1024.
        else:
            xpixHalfSize = 1024.
            ypixHalfSize = 2048.

        # calculate positions
        xPos = ccdinfo["xCenter"] + (ix-xpixHalfSize+0.5)*self.mmperpixel
        yPos = ccdinfo["yCenter"] + (iy-ypixHalfSize+0.5)*self.mmperpixel

        return xPos,yPos

    def getPixel(self,extname,xPos,yPos):
        # given a coordinate in [mm], return pixel number

        ccdinfo = self.infoDict[extname]

        # CCD size in pixels
        if ccdinfo["FAflag"]:
            xpixHalfSize = 1024.
            ypixHalfSize = 1024.
        else:
            xpixHalfSize = 1024.
            ypixHalfSize = 2048.

        # calculate positions
        ix = (xPos - ccdinfo["xCenter"]) / self.mmperpixel + xpixHalfSize - 0.5
        iy = (yPos - ccdinfo["yCenter"]) / self.mmperpixel + ypixHalfSize - 0.5

        return ix,iy

    def getPixel_no_extname(self, xPos, yPos):
        # get pixel coordinate without specifying extname

        coord_mm = [xPos, yPos]

        # determine extname
        for extname in self.infoDict:
            bounds = self.getBounds(extname, boxdiv=0)
            # bounds are [[xmin, xmax], [ymin, ymax]]
            inside = np.multiply(*[(coord_mm[i] > bounds[i][0]) *
                                   (coord_mm[i] < bounds[i][1])
                                   for i in range(2)])
            if inside:
                # we have found our extname!
                break
        ix, iy = self.getPixel(extname, xPos, yPos)
        return ix, iy

    def getEdges(self, boxdiv=0):
        '''
        Get the pixel edges across entire focal plane
        '''
        # calculate center to center chip distances
        x_step = 33.816
        y_step = 63.89
        x_min = -236.712
        x_max = -x_min + x_step
        y_min = -223.615
        y_max = -y_min + y_step

        if boxdiv == 0:
            x_edges = np.arange(x_min, x_max, x_step)
            y_edges = np.arange(y_min, y_max, y_step)
        else:
            x_edges = np.arange(x_min, x_max, x_step / 2 ** (boxdiv - 1))
            y_edges = np.arange(y_min, y_max, y_step / 2 ** boxdiv)

        return [x_edges, y_edges]


    def getBounds(self, extname, boxdiv=0):
        '''
        Give the coordinates of two opposite corners in mm
        '''
        ccdinfo = self.infoDict[extname]

        # CCD size in pixels
        if ccdinfo["FAflag"]:
            xpixHalfSize = 1024.
            ypixHalfSize = 1024.
        else:
            xpixHalfSize = 1024.
            ypixHalfSize = 2048.

        xmin = ccdinfo["xCenter"] - xpixHalfSize * self.mmperpixel
        xmax = ccdinfo["xCenter"] + xpixHalfSize * self.mmperpixel
        ymin = ccdinfo["yCenter"] - ypixHalfSize * self.mmperpixel
        ymax = ccdinfo["yCenter"] + ypixHalfSize * self.mmperpixel
        boundi = [[xmin, xmax], [ymin, ymax]]

        if boxdiv > 0:
            # now you need to blow these up
            # first off, the y ones need an extra division to make square boxes
            boundi[1].insert(1, (boundi[1][0] + boundi[1][1])/2)

            for div in xrange(1, boxdiv):
                '''
                put in extra cuts
                '''
                for k in xrange(2):
                    # make the location of each cut
                    cuts = [(boundi[k][j+1] + boundi[k][j])/2.
                            for j in xrange(len(boundi[k])-1)]

                    # append the cut
                    for j in cuts:
                        boundi[k].append(j)
                    # now sort the cuts
                    boundi[k].sort()

        return boundi

    def getBounds_pixel(self, extname='S9', boxdiv=0):
        '''
        Give the coordinates of two opposite corners in pixel
        '''
        ccdinfo = self.infoDict[extname]

        # CCD size in pixels
        if ccdinfo["FAflag"]:
            xpixHalfSize = 1024.
            ypixHalfSize = 1024.
        else:
            xpixHalfSize = 1024.
            ypixHalfSize = 2048.

        xmin = 0
        xmax = 2 * xpixHalfSize
        ymin = 0
        ymax = 2 * ypixHalfSize
        boundi = [[xmin, xmax], [ymin, ymax]]

        if boxdiv > 0:
            # now you need to blow these up
            # first off, the y ones need an extra division to make square boxes
            boundi[1].insert(1, (boundi[1][0] + boundi[1][1])/2)

            for div in xrange(1, boxdiv):
                '''
                put in extra cuts
                '''
                for k in xrange(2):
                    # make the location of each cut
                    cuts = [(boundi[k][j+1] + boundi[k][j])/2.
                            for j in xrange(len(boundi[k])-1)]

                    # append the cut
                    for j in cuts:
                        boundi[k].append(j)
                    # now sort the cuts
                    boundi[k].sort()

        return boundi

    def average_boxdiv(self, X, Y, P, average, boxdiv=1, rejectsize=1,
            Ntrue=False, members=False, boxes=False):
        '''
        give average X, Y, P in boxes as well as P2

        with boxdiv < 2, the chips are divided into 2 squares each
        of size ~ (30.72 mm) ** 2
        each box div after that breaks them by another half
        '''

        bounds = []
        for i in range(1, 63):
            if i == 61:
                #n30 sucks
                continue
            extname = self.ccddict[i]
            boundi = self.getBounds(extname, boxdiv)
            bounds.append(boundi)


        # do this for loop and then numpythonic ?
        Pave = []
        var_Pave = []
        N = []
        members_list = []
        boxes_list = []

        for box in bounds:
            for x in xrange(len(box[0]) - 1):
                for y in xrange(len(box[1]) - 1):
                    choose = (
                             (X > box[0][x]) *
                             (X < box[0][x + 1]) *
                             (Y > box[1][y]) *
                             (Y < box[1][y + 1]))
                    if np.sum(choose) < rejectsize:
                        # make sure we have enough in the box
                        continue
                    N.append(np.sum(choose))
                    Pave.append(average(P[choose]))
                    var_Pave.append(
                        average(np.square(P[choose] - average(P[choose]))))
                    if members:
                        members_list.append(np.where(choose)[0])
                    if boxes:
                        boxes_list.append([0.5 * (box[0][x] + box[0][x + 1]),
                                           0.5 * (box[1][y] + box[1][y + 1])])
        Pave = np.array(Pave)
        var_Pave = np.array(var_Pave)
        N = np.array(N)
        members_list = np.array(members_list)
        boxes_list = np.array(boxes_list)
        if boxes:
            return boxes_list
        if members:
            return bounds, members_list
        if Ntrue:
            return Pave, var_Pave, N, bounds
        else:
            return Pave, var_Pave

class mosaicinfo(object):
    """ mosaicinfo is a class used to contain Mosaic geometry information and various utility routines
    """

    def info(self):
        # info returns a dictionary chock full of info on the DECam geometry
        # keyed by the CCD name

        infoDict = {}

        # store a dictionary for each CCD, keyed by the CCD name
        infoDict["0"] =  {"xCenter":  -(2048+1024)*self.mmperpixel,"yCenter":-2048*self.mmperpixel,"FAflag":False}
        infoDict["1"] =  {"xCenter":  -(1024)*self.mmperpixel,     "yCenter":-2048*self.mmperpixel,"FAflag":False}
        infoDict["2"] =  {"xCenter":   (1024)*self.mmperpixel,     "yCenter":-2048*self.mmperpixel,"FAflag":False}
        infoDict["3"] =  {"xCenter":   (2048+1024)*self.mmperpixel,"yCenter":-2048*self.mmperpixel,"FAflag":False}
        infoDict["4"] =  {"xCenter":  -(2048+1024)*self.mmperpixel,"yCenter": 2048*self.mmperpixel,"FAflag":False}
        infoDict["5"] =  {"xCenter":  -(1024)*self.mmperpixel,     "yCenter": 2048*self.mmperpixel,"FAflag":False}
        infoDict["6"] =  {"xCenter":   (1024)*self.mmperpixel,     "yCenter": 2048*self.mmperpixel,"FAflag":False}
        infoDict["7"] =  {"xCenter":   (2048+1024)*self.mmperpixel,"yCenter": 2048*self.mmperpixel,"FAflag":False}


        return infoDict


    def __init__(self,**inputDict):

        self.mmperpixel = 0.015
        self.infoDict = self.info()


    def getPosition(self,extname,ix,iy):
        # return the x,y position in [mm] for a given CCD and pixel number
        # note that the ix,iy are Image pixels - overscans removed - and start at zero

        ccdinfo = self.infoDict[extname]

        # CCD size in pixels
        xpixHalfSize = 1024.
        ypixHalfSize = 2048.

        # calculate positions
        xPos = ccdinfo["xCenter"] + (float(ix)-xpixHalfSize+0.5)*self.mmperpixel
        yPos = ccdinfo["yCenter"] + (float(iy)-ypixHalfSize+0.5)*self.mmperpixel

        return xPos,yPos


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    x = decaminfo().infoDict

    plt.figure()
    plt.xlabel('$X$ [mm] (East)')
    plt.ylabel('$Y$ [mm] (South)')
    plt.xlim(-250,250)
    plt.ylim(-250,250)
    for xi in x.keys():
        plt.text(x[xi]['xCenter'], x[xi]['yCenter'], xi)


    # create ellipticity example plot
    e1 = np.arange(-0.9, 0.9, 18)
    e2 = np.arange(-0.9, 0.9, 18)

    plt.show()

    ccddict = {}
    for keyi in decaminfo().infoDict.keys():
        ccddict.update({decaminfo().infoDict[keyi]['CCDNUM']: keyi})


    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    for boxdiv in range(4):
        bounds = [decaminfo().getBounds(i, boxdiv) for i in decaminfo().infoDict.keys()]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('$X$ [mm] (East)')
        ax.set_ylabel('$Y$ [mm] (South)')
        ax.set_xlim(-250,250)
        ax.set_ylim(-250,250)
        patches = []
        for box in bounds:
            for x in range(len(box[0]) - 1):
                for y in range(len(box[1]) - 1):
                    height = box[1][y+1] - box[1][y]
                    width = box[0][x+1] - box[0][x]
                    art = Rectangle([box[0][x], box[1][y]], width, height)
                    patches.append(art)
        collection = PatchCollection(patches, alpha=0.4)
        ax.add_collection(collection)

    plt.show()

