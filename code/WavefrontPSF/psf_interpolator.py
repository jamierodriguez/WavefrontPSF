#!/usr/bin/env python
"""
File: psf_interpolator.py
Author: Chris Davis
Description: Module that takes focal plane coordinates and other relevant data
             and returns some kind of representation of the basis.

"""

from pandas import DataFrame, read_csv
from numpy import vstack, float64
from WavefrontPSF.decamutil import decaminfo

class PSF_Interpolator(object):
    """Class that returns some sort of PSF representation.

    Attributes
    ----------

    x_keys : list of strings
        Keys that an interpolator takes to create output
    y_keys : list of strings
        Names of output keys

    Methods
    -------
    interpolate
        Returns the psf at some location. By default your basis is just
        whatever you put in.

    """

    def __init__(self, y_keys=[], x_keys=[], **kwargs):
        self.y_keys = y_keys
        self.x_keys = x_keys

    def check_data_for_keys(self, data):
        # very simple: just try to access
        can_do_x = True
        for key in self.x_keys:
            try:
                #print(key, data[key])
                data[key]
            except E:
                print('Problems with', key, E)
                can_do_x = False
        can_do_y = True
        for key in self.y_keys:
            try:
                #print(key, data[key])
                data[key]
            except E:
                print('Problems with', key, E)
                can_do_y = False
        return can_do_x, can_do_y

    def interpolate(self, X, **kwargs):
        interpolated = {}
        for key in self.x_keys:
            interpolated[key] = X[key]
        # for key in self.y_keys:
        #     interpolated[key] = func[key](X[self.x_keys])
        interpolated = DataFrame(interpolated)

        return interpolated

    def __call__(self, X, **kwargs):
        return self.interpolate(X, **kwargs)

class kNN_Interpolator(PSF_Interpolator):
    """Impliment my own version of Aaron's base inverse distance weighted
    interpolation. Using a Digestor to create your data, then interpolate using
    k nearest neighbors.

    """

    def __init__(self, data,
                 y_keys=['z{0}'.format(i) for i in range(4, 12)],
                 x_keys=['x', 'y'], **kwargs):
        """
        x_keys, y_keys : lists of strings
            The keys which we want to use as input (x) and output (y).
        """
        super(kNN_Interpolator, self).__init__(y_keys=y_keys, x_keys=x_keys, **kwargs)

        from sklearn.neighbors import KNeighborsRegressor

        knn_args = {'n_neighbors': 45,
                    'weights': 'uniform',
                    'p': 1,
                    }

        knn_args.update(kwargs)

        self.data = data

        # train the interpolant for each key
        # since training is just copying the data... this should be quick
        self.knn = {}
        for key in self.y_keys:
            self.knn[key] = KNeighborsRegressor(**knn_args)
            self.knn[key].fit(self.data[x_keys], self.data[key])

    def interpolate(self, X, force_interpolation=True, **kwargs):
        # force_interpolation: if false, if ALL interpolant variables already
        # present in X, then do not actually create new interpolation
        do_interpolation = force_interpolation
        for key in self.y_keys:
            # if a y_key is not present, force the interpolation
            if key not in X:
                do_interpolation = True

        if do_interpolation:
            interpolated = {}
            # for key in X.keys():#self.x_keys:
            #     interpolated[key] = X[key]
            for key in self.y_keys:
                interpolated[key] = self.knn[key].predict(X[self.x_keys])
            interpolated = DataFrame(interpolated)
            # want to overwrite any preexisting x
            X = interpolated.combine_first(X)
        else:
            # do nothing
            pass

        return X

    def __call__(self, X, **kwargs):
        return self.interpolate(X, **kwargs)

class Mesh_Interpolator(kNN_Interpolator):

    def __init__(self, mesh_name, directory, **kwargs):
        self.dec = decaminfo()
        y_keys = ['z{0}'.format(i) for i in range(4, 12)]
        x_keys = ['x', 'y']

        # ingest the data here
        data = self.digest_mesh(directory=directory, mesh_name=mesh_name)

        super(Mesh_Interpolator, self).__init__(data=data, y_keys=y_keys, x_keys=x_keys, **kwargs)

    def digest_mesh(self, mesh_name, directory):
        for z in range(4, 12):
            fileTitle = 'z{0}'.format(z) + 'Mesh_' + mesh_name
            fileName = directory + '/' + fileTitle + '.dat'
            zkey = 'z{0}'.format(z)
            wkey = 'w{0}'.format(z)

            dataPoints = read_csv(fileName, delim_whitespace=True,
                    header=None,
                    dtype={'Sensor': '|S3', 'x': float64, 'y': float64,
                           zkey: float64, wkey: float64},
                    names=['Sensor', 'x', 'y', zkey, wkey])

            if z == 4:
                ccdnum = [self.dec.infoDict[sensor_i]['CCDNUM']
                          for sensor_i in dataPoints['Sensor']]
                # data['x'] = dataPoints['x']
                # data['y'] = dataPoints['y']
                data = dataPoints
                data['ccdnum'] = ccdnum

            data[zkey] = dataPoints[zkey]
            data[wkey] = dataPoints[wkey]
        # take z4 and divide by 172 to put it in waves as it should be
        data['z4'] /= 172.
        return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from WavefrontPSF.wavefront import Wavefront
    from WavefrontPSF.psf_evaluator import Moment_Evaluator

    PSF_Evaluator = Moment_Evaluator()

    directory = '/Users/cpd/Projects/WavefrontPSF/meshes/Science-20140212s2-v1i2'
    mesh_name = 'Science-20140212s2-v1i2_All'
    PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=directory)
    WF = Wavefront(PSF_Interpolator=PSF_Interpolator,
                   PSF_Evaluator=PSF_Evaluator,
                   model=PSF_Interpolator.data,
                   num_bins=3)

    """ compare timings of my digestion vs donutana
    %timeit WF.PSF_Interpolator.digest_mesh(mesh_name, directory)

    from donutlib.donutana import donutana
    sensorSet = "ScienceOnly"
    method = "idw"
    nInterpGrid = 32
    methodVal = (4, 1.0)

    in_dict = {"sensorSet": sensorSet,
              "doTrefoil": True,
              "doSpherical": True,
              "doQuadrefoil": False,
              "unVignettedOnly": False,
              "interpMethod": method,
              "methodVal": methodVal,
              "nInterpGrid": nInterpGrid,
              "histFlag": False,  # True,
              "debugFlag": True,  # True,
              "donutCutString": ""}

    for zi in range(4, 12):
        try:
            in_dict.update({'z{0}PointsFile'.format(zi):
                           directory + '/z{0}Mesh_'.format(zi) +
                           mesh_name + '.dat'})
        except IOError:
            continue

    da = donutana(**in_dict)

    %timeit da = donutana(**in_dict)

    """

    directory = '/Users/cpd/Projects/WavefrontPSF/meshes/Science-20130325s1-v1i2'
    mesh_name = 'Science-20130325s1-v1i2_All'
    PSF_Interpolator = Mesh_Interpolator(mesh_name=mesh_name, directory=directory)
    WF_old = Wavefront(PSF_Interpolator=PSF_Interpolator,
                   PSF_Evaluator=PSF_Evaluator,
                   model=PSF_Interpolator.data,
                   num_bins=3)

    for z in range(4, 12):
        WF.field['z{0}_diff'.format(z)] = WF['z{0}'.format(z)] - WF_old['z{0}'.format(z)]
        fig, ax = WF.plot_field('z{0}_diff'.format(z))
        ax.set_title('z{0}_diff'.format(z))
        fig, ax = WF.plot_field('z{0}'.format(z))
        ax.set_title('z{0}'.format(z))
    plt.show()

    import ipdb; ipdb.set_trace()
