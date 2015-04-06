# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# make FocalPlaneShell object
from focal_plane import FocalPlane
from focal_plane_routines import minuit_dictionary, mean_trim
from decam_csv_routines import generate_hdu_lists
from routines_plot import data_focal_plot, data_hist_plot

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
data = FP.recdata
#data = np.concatenate((data[:7779], data[7780:8787], data[8788:]))

# <codecell>

def plot_star(recdata, star_figure=None, star_axis=None, nPixels=32,
              weighted=False):
    """Intelligently plot a star

    Parameters
    ----------
    recdata : recarray
        Contains a parameter 'STAMP' that needs to be resized, as well as
        X2WIN_IMAGE, XYWIN_IMAGE, Y2WIN_IMAGE (moment parameters) that then
        will be converted into the adaptive moment matrix.

    star_figure : matplotlib figure, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    star_axis : matplotlib axis, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    nPixels : integer
        Size of the stamp array dimensions

    weighted : bool
        If true, apply weighting exp(-rho2 / 2) as defined in Hirata et al 2004

    Returns
    -------
    star_figure : matplotlib figure, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    star_axis : matplotlib axis, optional
        The axis on which we manipulate. If not given, then the figure and axis
        are created.

    """

    # make the figure
    if (not star_figure) * (not star_axis):
        star_figure = plt.figure()
        star_axis = star_figure.add_subplot(111, aspect='equal')


    d = recdata.copy()
    stamp = d['STAMP']
    stamp.reshape(nPixels, nPixels)

    # filter via background and threshold
    stamp -= d['BACKGROUND']
    stamp = np.where(stamp > d['THRESHOLD'], stamp, np.nan)

    # do window
    max_nsig2 = 25
    y, x = np.indices(stamp.shape)
    Mx = nPixels / 2 + d['XWIN_IMAGE'] % int(d['XWIN_IMAGE']) - 1
    My = nPixels / 2 + d['YWIN_IMAGE'] % int(d['YWIN_IMAGE']) - 1

    r2 = (x - Mx) ** 2 + (y - My) ** 2

    Mxx = 2 * d['X2WIN_IMAGE']
    Myy = 2 * d['Y2WIN_IMAGE']
    Mxy = 2 * d['XYWIN_IMAGE']
    detM = Mxx * Myy - Mxy * Mxy
    Minv_xx = Myy / detM
    TwoMinv_xy = -Mxy / detM * 2.0
    Minv_yy = Mxx / detM
    Inv2Minv_xx = 0.5 / Minv_xx
    rho2 = Minv_xx * (x - Mx) ** 2 + TwoMinv_xy * (x - Mx) * (y - My) + Minv_yy * (y - My) ** 2
    stamp = np.where(rho2 < max_nsig2, stamp, np.nan)

    if weighted:
        stamp *= np.exp(-0.5 * rho2)

    im = star_axis.imshow(stamp)
    plt.colorbar(im)

    return star_figure, star_axis



def process_image(recdata, weighted=True, nPixels=32):
    d = recdata.copy()
    stamp = d['STAMP']
    stamp.reshape(nPixels, nPixels)

    # filter via background and threshold
    stamp -= d['BACKGROUND']
    stamp = np.where(stamp > d['THRESHOLD'], stamp, 0)

    # do window
    max_nsig2 = 25
    y, x = np.indices(stamp.shape)
    Mx = nPixels / 2 + d['XWIN_IMAGE'] % int(d['XWIN_IMAGE']) - 1
    My = nPixels / 2 + d['YWIN_IMAGE'] % int(d['YWIN_IMAGE']) - 1

    r2 = (x - Mx) ** 2 + (y - My) ** 2

    Mxx = 2 * d['X2WIN_IMAGE']
    Myy = 2 * d['Y2WIN_IMAGE']
    Mxy = 2 * d['XYWIN_IMAGE']
    detM = Mxx * Myy - Mxy * Mxy
    Minv_xx = Myy / detM
    TwoMinv_xy = -Mxy / detM * 2.0
    Minv_yy = Mxx / detM
    Inv2Minv_xx = 0.5 / Minv_xx
    rho2 = Minv_xx * (x - Mx) ** 2 + TwoMinv_xy * (x - Mx) * (y - My) + Minv_yy * (y - My) ** 2
    stamp = np.where(rho2 < max_nsig2, stamp, 0)

    if weighted:
        stamp *= np.exp(-0.5 * rho2)
        
    return stamp

def ims(data):
    d = data.copy()
    stamp = d
    stamp.resize(32,32)
    
    imshow(stamp)
    return

# <codecell>

stamps = [process_image(di) for di in data]
stamps = np.array(stamps)
ims(np.std(stamps, axis=0))
colorbar()
figure()
ims(np.mean(stamps, axis=0))
colorbar()

# <codecell>

args = data[np.argsort(data['SN_FLUX'])[:20]]
for arg in args:
    fig, ax = plot_star(arg, weighted=False)

# <codecell>

def cutouts(recdata):
    nPixels = 32
    stamp = process_image(recdata, weighted=False, nPixels=nPixels)

    # take central 20 x 20 pixels and 5 x 5
    center = nPixels / 2
    cutout = stamp[center - 10: center + 10, center - 10: center + 10]
    central_cutout = stamp[center - 5: center + 5, center - 5: center + 5]
    
    return cutout, central_cutout

# <codecell>

fail = [stamp_i for stamp_i in xrange(len(data['STAMP']))
        if np.max(cutouts(data[stamp_i])[0]) != np.max(cutouts(data[stamp_i])[1])]
for fail_i in fail:
    plot_star(data[fail_i])

# <codecell>

from sklearn.decomposition import PCA
from sklearn import preprocessing
data_scaled = preprocessing.scale(data)
# do PCA analysis
n_components = 20
pca = PCA(n_components)
pca.fit(data_scaled)
pca_score = pca.explained_variance_ratio_
V = pca.components_

print(pca_score, V)

# <codecell>

for v in range(n_components):
    figure()
    ims(V[v])
    colorbar()

# <codecell>


