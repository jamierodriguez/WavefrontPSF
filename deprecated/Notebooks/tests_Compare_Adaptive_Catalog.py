# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Last done February 5 ; January 22

# <codecell>

from scipy.stats import norm
import matplotlib.mlab as mlab

# make an image
from focal_plane import FocalPlane
from decamutil_cpd import decaminfo
import pyfits
from focal_plane_routines import MAD
from decam_csv_routines import generate_hdu_lists, generate_hdu_lists_sex
from focal_plane import FocalPlane
from focal_plane_routines import minuit_dictionary, mean_trim
from decam_csv_routines import generate_hdu_lists
from routines_plot import data_focal_plot, data_hist_plot
# create an FP object
path_base = '/Users/cpd/Desktop/Images/'

expid = 232698
path_mesh = '/Users/cpd/Desktop/Meshes/'
list_catalogs, list_fits_extension, list_chip = \
        generate_hdu_lists(expid, path_base='/Users/cpd/Desktop/Catalogs/')
    
FP = FocalPlane(list_catalogs=list_catalogs,
                list_fits_extension=list_fits_extension,
                list_chip=list_chip,
                boxdiv=0,
                max_samples_box=5000,
                conds='default',
                average=mean_trim,
                path_mesh=path_mesh,
                nPixels=32,
                )

recdatas = FP.recdata

# <codecell>

from adaptive_moments import adaptive_moments
from routines_plot import plot_star

for recdata in recdatas[:10]:

    _ = plot_star(recdata, weighted=False)

# <codecell>

mad_keys = ['X2WIN_IMAGE', 'X2WIN_IMAGE_SEX']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = plt.hist2d(data[mad_keys[0]], data[mad_keys[1]], bins=50)
colorbar()

# fit a line
z = np.polyfit(data[mad_keys[0]], data[mad_keys[1]], 1)
print(z)
x = np.linspace(data[mad_keys[0]].min(), data[mad_keys[0]].max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (data[mad_keys[1]] - np.poly1d(z)(data[mad_keys[0]])) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

mad_keys = ['Y2WIN_IMAGE', 'Y2WIN_IMAGE_SEX']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = plt.hist2d(data[mad_keys[0]], data[mad_keys[1]], bins=50)
colorbar()

# fit a line
z = np.polyfit(data[mad_keys[0]], data[mad_keys[1]], 1)
print(z)
x = np.linspace(data[mad_keys[0]].min(), data[mad_keys[0]].max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (data[mad_keys[1]] - np.poly1d(z)(data[mad_keys[0]])) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

mad_keys = ['X2WIN_IMAGE', 'X2WIN_IMAGE_SEX', 'Y2WIN_IMAGE', 'Y2WIN_IMAGE_SEX']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]

e1 = (data[mad_keys[0]] - data[mad_keys[2]]) / 2  # why this /2?
e1_sex = data[mad_keys[1]] - data[mad_keys[3]]

figure()
title('e1')
outs = plt.hist2d(e1, e1_sex, bins=50)
colorbar()

# fit a line
z = np.polyfit(e1, e1_sex, 1)
print(z)
x = np.linspace(e1.min(), e1.max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (e1_sex - np.poly1d(z)(e1)) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

mad_keys = ['X2WIN_IMAGE', 'X2WIN_IMAGE_SEX', 'Y2WIN_IMAGE', 'Y2WIN_IMAGE_SEX']

data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]

e1 = data[mad_keys[0]] + data[mad_keys[2]]
e1_sex = data[mad_keys[1]] + data[mad_keys[3]]

figure()
title('e0')
outs = plt.hist2d(e1, e1_sex, bins=50)
colorbar()

# fit a line
z = np.polyfit(e1, e1_sex, 1)
print(z)
x = np.linspace(e1.min(), e1.max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (e1_sex - np.poly1d(z)(e1)) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

# also doubles as e2...

mad_keys = ['XYWIN_IMAGE', 'XYWIN_IMAGE_SEX']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = plt.hist2d(data[mad_keys[0]], data[mad_keys[1]], bins=50)
colorbar()

# fit a line
z = np.polyfit(data[mad_keys[0]], data[mad_keys[1]], 1)
print(z)
x = np.linspace(data[mad_keys[0]].min(), data[mad_keys[0]].max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (data[mad_keys[1]] - np.poly1d(z)(data[mad_keys[0]])) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

mad_keys = ['X2WIN_IMAGE', 'X2WIN_IMAGE_SEX', 'Y2WIN_IMAGE', 'Y2WIN_IMAGE_SEX']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]

e1 = (data[mad_keys[0]] - data[mad_keys[2]]) / 2  # why this /2?
e1_sex = data[mad_keys[1]] - data[mad_keys[3]]

e0 = (data[mad_keys[0]] + data[mad_keys[2]])
e0_sex = data[mad_keys[1]] + data[mad_keys[3]]

e1 = e1 / e0
e1_sex = e1_sex / e0_sex

figure()
title('e1/e0')
outs = plt.hist2d(e1, e1_sex, bins=50)
colorbar()

# fit a line
z = np.polyfit(e1, e1_sex, 1)
print(z)
x = np.linspace(e1.min(), e1.max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (e1_sex - np.poly1d(z)(e1)) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

mad_keys = ['X2WIN_IMAGE', 'X2WIN_IMAGE_SEX', 'Y2WIN_IMAGE', 'Y2WIN_IMAGE_SEX', 'XYWIN_IMAGE', 'XYWIN_IMAGE_SEX']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]

e1 = (data[mad_keys[4]]) / 2  # why this /2?
e1_sex = data[mad_keys[5]]

e0 = (data[mad_keys[0]] + data[mad_keys[2]])
e0_sex = data[mad_keys[1]] + data[mad_keys[3]]

e1 = e1 / e0
e1_sex = e1_sex / e0_sex

figure()
title('e2/e0')
outs = plt.hist2d(e1, e1_sex, bins=50)
colorbar()

# fit a line
z = np.polyfit(e1, e1_sex, 1)
print(z)
x = np.linspace(e1.min(), e1.max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (e1_sex - np.poly1d(z)(e1)) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

mad_keys = ['X2WIN_IMAGE', 'Y2WIN_IMAGE', 'FWHM_WORLD']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]

e0 = np.sqrt(data[mad_keys[0]] + data[mad_keys[1]])
fwhm = data[mad_keys[2]] * 3600 / 0.27

figure()
title('e0 vs fwhm')
outs = plt.hist2d(e0, fwhm, bins=50)
colorbar()

# fit a line
z = np.polyfit(e0, fwhm, 1)
print(z)
x = np.linspace(e0.min(), e0.max(), 50)
plt.plot(x, np.poly1d(z)(x), 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (fwhm - np.poly1d(z)(e0)) / error
n, bins, patches = hist(pull, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <markdowncell>

# The very linear relationship makes sense given how the fwhm is made... and how usually Mxy << Myy, Mxx

# <codecell>

mad_keys = ['FWHM_WORLD', 'FWHM_WORLD_SEX']

data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = plt.hist2d(data[mad_keys[0]] * 3600, data[mad_keys[1]] * 3600, bins=50)
colorbar()

# fit a line
z = np.polyfit(data[mad_keys[0]], data[mad_keys[1]], 1)
print(z)
x = np.linspace(data[mad_keys[0]].min(), data[mad_keys[0]].max(), 50)
plt.plot(x * 3600, np.poly1d(z)(x) * 3600, 'b-')

figure()
error = 1  # TODO: Figure this out!
pull = (data[mad_keys[1]] - np.poly1d(z)(data[mad_keys[0]])) / error
n, bins, patches = hist(pull * 3600, normed=1, bins=20)
# best fit of data
(mu, sigma) = norm.fit(pull * 3600)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
title('{0:.2e}, {1:.2e}'.format(mu, sigma))

# <codecell>

mad_keys = ['FLUX_ADAPTIVE', 'MAG_AUTO']


data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = plt.hist2d(data[mad_keys[0]], 10**(-1 / 2.5 * data[mad_keys[1]]), bins=20)
colorbar()

# <codecell>

mad_keys = ['A4_ADAPTIVE']
data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = hist(data[mad_keys[0]], 25)

# <codecell>

mad_keys = ['WHISKER']
data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = hist(data[mad_keys[0]], 25)

# <codecell>

mad_keys = ['SPREAD_MODEL']
data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = hist(data[mad_keys[0]], 25)

# <codecell>

mad_keys = ['CLASS_STAR']
data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title(mad_keys[0])
outs = hist(data[mad_keys[0]], 25)

# <codecell>

mad_keys = ['X2WIN_IMAGE', 'X2WIN_IMAGE_SEX']
data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title('{0} - {1}'.format(mad_keys[0], mad_keys[1]))
outs = hist(data[mad_keys[0]] - data[mad_keys[1]], 25)

# <codecell>

mad_keys = ['XYWIN_IMAGE', 'XYWIN_IMAGE_SEX']
data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title('{0} - {1}'.format(mad_keys[0], mad_keys[1]))
outs = hist(data[mad_keys[0]] - data[mad_keys[1]], 25)

# <codecell>

mad_keys = ['Y2WIN_IMAGE', 'Y2WIN_IMAGE_SEX']
data = recdatas.copy()

conds = FP.filter(data, conds='default')
for mad_key in mad_keys:
    a = data[mad_key]
    d = np.median(a)
    c = 0.6745  # constant to convert from MAD to std
    mad = np.median(np.fabs(a - d) / c)
    conds_mad = (a < d + 3 * mad) * (a > d - 3 * mad)
    conds *= conds_mad
data = data[conds]
figure()
title('{0} - {1}'.format(mad_keys[0], mad_keys[1]))
outs = hist(data[mad_keys[0]] - data[mad_keys[1]], 25)

# <markdowncell>

# double check that galsim gives same results

# <codecell>

from routines_plot import process_image
from focal_plane_routines import convert_moments

i = 32
stamp = FP.recdata['STAMP'][i].reshape((FP.input_dict['nPixels'],FP.input_dict['nPixels']))
background = FP.recdata['BACKGROUND'][i]
threshold = FP.recdata['THRESHOLD'][i]

data = process_image(FP.recdata[i], weighted=False)
plot_star(FP.recdata[i])

# stamp = FP.stamp(zernike=[0.0,
#   -0.12858833770791267,
#   0.3283656145136757,
#   -0.55051450665826,
#   -0.10895992147064192,
#   0.12247219994194478,
#   0.167624831250874,
#   0.21903857808265365,
#   0.11867602330016813,
#   -0.17762883519138856,
#   0.25646224854963],
#                  rzero=0.14,
#                  coord=[0,0])
# background = 0
# threshold = 0
# imshow(stamp)

# analyze via galsim
import galsim
scale = 1
image = galsim.Image(array=stamp, scale=scale)  # scale is units / pixel, e.g. 0.27
shape = image.FindAdaptiveMom(image)
E1 = shape.observed_shape.getE1()
E2 = shape.observed_shape.getE2()
amp = shape.moments_amp
sigma = shape.moments_sigma
rho4 = shape.moments_rho4
centroid = shape.moments_centroid
m_x = centroid.x
m_y = centroid.y
e0 = np.square(sigma) / np.sqrt(1 - E1 ** 2 - E2 ** 2) * 0.27 ** 2  # into arcsec2
e1 = E1 * e0
e2 = E2 * e0
m_xy = e2 / 0.27 ** 2  # back into pixel2
m_xx = (e0 + e1) / 0.27 ** 2
m_yy = (e0 - e1) / 0.27 ** 2
properties_galsim = dict(flux=amp / 2, fwhm=sigma, a4=0.5 * rho4 - 1,
                         E1 = E1, E2 = E2,
                         e0 = e0, e1 = e1, e2 = e2,
                         x = m_x, y = m_y,
                         Mxx = m_xx, Mxy = m_xy, Myy = m_yy,
                         x2 = m_xx, xy = m_xy, y2 = m_yy)
moment_dict = convert_moments(FP.moments(stamp,
                             background=background,
                             threshold=threshold,
                             ))
moment_dict.update({'E1': moment_dict['e1'] / moment_dict['e0'],
                    'E2': moment_dict['e2'] / moment_dict['e0'],
                    })

keys = [key for key in properties_galsim if key in moment_dict]
for key in sort(keys):
    print(key, properties_galsim[key], moment_dict[key], properties_galsim[key] / moment_dict[key])

# <markdowncell>

# passes!

# <codecell>


