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
    evaluated_psfs = PSF_Evaluator(stamps)
    # this is sick:
    combined_df = evaluated_psfs.combine_first(data)

    return combined_df



out_dir = '/Volumes/Seagate/DES/donut_correspondance/'

rzero = 0.14
x = 10
y = 0

# y goes from -221 to 221
# x goes fro -200 to 200
# rzero goes from 0.1 to 0.25


Ndim = 8  # z4 through 11
Nsample = 100000
zmax = 1.0

zernikes = np.random.random(size=(Ndim, Nsample)) * (zmax + zmax) - zmax
x, y, ext = generate_random_coordinates(Nsample)

for rzero_i, rzero in enumerate([0.10, 0.12, 0.14, 0.16, 0.18, 0.20]):

    # do batch with constant coordinates
    data = {'rzero': np.ones(Nsample) * rzero,
            'x': np.ones(Nsample) * 10,
            'y': np.ones(Nsample) * 0}

    x_keys = []
    for zi, zernike in enumerate(zernikes):
        zkey = 'z{0}'.format(zi + 4)
        data[zkey] = zernike.flatten()
        x_keys.append(zkey)
    df = pd.DataFrame(data)
    df = evaluate_psf(df)

    df.to_csv(out_dir + 'donuts_{0}_constcoords.csv'.format(rzero_i))


    data = {'rzero': np.ones(Nsample) * rzero,
            'x': x,
            'y': y}

    x_keys = []
    for zi, zernike in enumerate(zernikes):
        zkey = 'z{0}'.format(zi + 4)
        data[zkey] = zernike.flatten()
        x_keys.append(zkey)
    df = pd.DataFrame(data)
    df = evaluate_psf(df)

    df.to_csv(out_dir + 'donuts_{0}_varcoords.csv'.format(rzero_i))

# combine all the donuts together
from glob import glob
out_dir = '/Volumes/Seagate/DES/donut_correspondance/'
csvs = glob(out_dir + '*.csv')
df = pd.read_csv(csvs[0], index_col=0)
for csv in csvs[1:]:
    df = df.append(pd.read_csv(csv, index_col=0), ignore_index=True)
df.to_csv(out_dir + 'donuts.csv')
