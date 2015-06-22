""" just some default parameter locations"""

def param_default_cpd(expid=0):
    # params are configged for ki-ls. This gives some for my laptop
    """
from WavefrontPSF.fitter import param_default_cpd, drive_fit, do_fit, plot_results
expid = 174000
params = param_default_cpd(expid)
WF = drive_fit(expid, params=params, skip_fit=True)
minuit, chi2, plane = drive_fit(expid, params=params)

comp, fig, ax = plot_results(WF, plane, minuit, num_bins=2)

misalignment={'e0':0, 'e1':0, 'e2': 0, 'z04d':0, 'z05d':0, 'z06d':0, 'z07d':0, 'z08d':0}
    """
    expid_path = '{0:08d}/{1:08d}'.format(expid - expid % 1000, expid)

    params = {'mesh_directory': '/Users/cpd/Projects/WavefrontPSF/meshes/Science-20140212s2-v1i2',
              'data_directory': '/Users/cpd/Projects/WavefrontPSF/meshes/obsevations/' + expid_path,
              'data_name': '_selpsfcat.fits',
              'analytic_coeffs': '/Users/cpd/Projects/WavefrontPSF/meshes/Analytic_Coeffs/model.npy'}
    return params

def param_default_kils(expid=0):
    expid_path = '{0:08d}/{1:08d}'.format(expid - expid % 1000, expid)

    params = {'mesh_directory': '/nfs/slac/g/ki/ki18/cpd/Projects/WavefrontPSF/meshes/Science-20140212s2-v1i2',
              'mesh_name': 'Science-20140212s2-v1i2_All',
              'data_directory': '/nfs/slac/g/ki/ki18/des/cpd/psfex_catalogs/SVA1_FINALCUT/psfcat/' + expid_path,
              'analytic_coeffs':'/nfs/slac/g/ki/ki18/cpd/Projects/WavefrontPSF/meshes/Analytic_Coeffs/model.npy'}

    return params

