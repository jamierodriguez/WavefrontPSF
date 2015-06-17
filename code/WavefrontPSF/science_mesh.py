"""
File: science_mesh.py
Author: Chris Davis
Description: Module that takes directory of aaron's meshes and returns a knn representation.
"""

import numpy as np
import pandas as pd

from WavefrontPSF.digestor import Digestor
from WavefrontPSF.psf_interpolator import kNN_Interpolator


class Mesh_Interpolator(kNN_Interpolator):

    def __init__(self, files, **kwargs):
        y_keys = ['z{0}'.format(i) for i in range(4, 12)]
        x_keys = ['x', 'y']

        # ingest the data here

        super(Mesh_Interpolator, self).__init__(data=data, y_keys=y_keys, x_keys=x_keys, **kwargs)


