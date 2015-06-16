from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define some functions
from WavefrontPSF.donutengine import Zernike_to_Pixel_Interpolator, generate_random_coordinates
from WavefrontPSF.psf_evaluator import Moment_Evaluator

# Create PSF Data
PSF_Drawer = Zernike_to_Pixel_Interpolator()
PSF_Evaluator = Moment_Evaluator()
def evaluate_psf(data):
    stamps, data = PSF_Drawer(data)
    print('stamps made')
    evaluated_psfs = PSF_Evaluator(stamps)
    # this is sick:
    combined_df = evaluated_psfs.combine_first(data)

    return combined_df



out_dir = '/Volumes/Seagate/DES/donut_correspondance/'
out_dir = '/nfs/slac/g/ki/ki18/des/cpd/donuts/'

# y goes from -221 to 221
# x goes fro -200 to 200
# rzero goes from 0.1 to 0.25


Ndim = 8  # z4 through 11
Nsample = 1000000
zmax = 1.0

zernikes = np.random.random(size=(Ndim, Nsample)) * (zmax + zmax) - zmax
x, y, ext = generate_random_coordinates(Nsample)

rzero = np.random.random(size=Nsample) * (0.25 - 0.1) + 0.1



data = {'rzero': rzero,
        'x': x,
        'y': y}

x_keys = []
for zi, zernike in enumerate(zernikes):
    zkey = 'z{0}'.format(zi + 4)
    data[zkey] = zernike.flatten()
    x_keys.append(zkey)
df = pd.DataFrame(data)
df = evaluate_psf(df)

df.to_csv(out_dir + 'donuts_all_params.csv')
