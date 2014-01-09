from __future__ import print_function, division
from array import array
import numpy as np
import time

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

'''
minuit tolerance
'''
#TODO: PEP8 compliant; comments


def grad_func(in_dict, fit_func, error_dict, h_base=1e-1,
              stencil=[[-1. / 2, -1], [1. / 2, 1]]):
    """numerical gradient of a function

    Parameters
    ----------

    in_dict : dictionary
        A dictionary of the current values for the fit_func

    fit_func : function
        The function we are concerned with. Must be able to call
        fit_func(**in_dict)

    error_dict : dictionary
        Dictionary giving the error for each parameter, assumed to be in the
        format 'error_{parameter_name}'

    h_base : float, optional
        How much smaller than the errors are we stepping for our gradient?

    stencil : list, optional
        What stencil are we using. In format [[multiplicative_factor, number of
        h's]]. Default is centered two point stencil.

    Returns
    -------

    derivatives : dictionary
        A dictionary of derivatives where each key is the first derivative on
        that parameter

    """

    #stencil=[[-1. / 12, 2], [2. / 3, 1], [-2. / 3, -1], [1. / 12, -2]]):
    #print(in_dict)
    derivatives = {}
    for key in in_dict:
        h = h_base * error_dict['error_{0}'.format(key)]
        # calculate 5 point stencil
        fprime = 0
        for h_mod in stencil:
            prime_dict = in_dict.copy()
            prime_dict[key] += h_mod[1] * h
            fprime += h_mod[0] * fit_func(**prime_dict) / h
        derivatives.update({key: fprime})
    return derivatives


class Minuit_Fit(object):
    """ generic minuit fitting function
        GradFunc takes an input dictionary of values

    Attributes
    ----------
    gFitFunc : function
        The function we shall fit.

    gGradFunc : function, optional
        The function for the gradient

    verbosity : int

    force_derivatives : int

    max_iterations : int

    minuit_dict : dictionary

    par_names : list
        list of the parameter names

    npar : int
        len(par_names)

    strategy : int

    tolerance : float

    grad_dict : dictionary

    gMinuit : MINUIT object

    Methods
    -------
    setupFit

    chisq -- private; cut out

    doFit

    outFit

    """

    def __init__(self, FitFunc, minuit_dict, par_names,
                 GradFunc=grad_func,
                 grad_dict=dict(h_base=1e-3,
                                stencil=[[-1. / 12, 2], [2. / 3, 1],
                                         [-2. / 3, -1], [1. / 12, -2]]),
                 force_derivatives=0,
                 verbosity=0, max_iterations=1000,
                 tolerance=0.3,
                 strategy=1):
        # init contains all initializations which are done only once for all
        # fits

        self.gFitFunc = FitFunc
        self.gGradFunc = GradFunc
        self.verbosity = verbosity
        self.force_derivatives = force_derivatives
        self.max_iterations = max_iterations
        self.minuit_dict = minuit_dict
        self.npar = len(par_names)
        self.par_names = par_names
        self.strategy = strategy
        self.tolerance = tolerance
        self.grad_dict = grad_dict
        if 'error_dict' not in self.grad_dict.keys():
            error_dict = {}
            for key in par_names:
                error_dict.update({'error_{0}'.format(key):
                                   minuit_dict['error_{0}'.format(key)]})
            self.grad_dict.update({'error_dict': error_dict})

        # setup MINUIT
        self.gMinuit = ROOT.TMinuit(self.npar)
        self.gMinuit.SetFCN(self.chisq)

        # arglist is for the parameters in Minuit commands
        arglist = array('d', 10 * [0.])
        ierflg = ROOT.Long(1982)  # or should it be ROOT.Long(0) ?

        # set the definition of 1sigma
        arglist[0] = 1.0
        self.gMinuit.mnexcm("SET ERR", arglist, 1, ierflg)

        # turn off Warnings
        arglist[0] = 0
        self.gMinuit.mnexcm("SET NOWARNINGS", arglist, 0, ierflg)

        # set printlevel
        arglist[0] = self.verbosity
        self.gMinuit.mnexcm("SET PRINTOUT", arglist, 1, ierflg)

        # do initial setup of Minuit parameters
        self.setupFit()

    def setupFit(self):
        """ set up and initialize parameters for the fit

        Attributes born
        ---------------
        iflag
        old_dict
        nCalls
        nCallsDerivative

        should these guys be born in the init?
        """

        keys = self.minuit_dict.keys()
        self.iflag = 0  # for noting changes in iflag
        self.old_dict = {}  # for comparing across steps
        # Set starting values and step sizes for parameters (note that one can
        # redefine the parameters, so this method can be called multiple times)
        for ipar in range(self.npar):
            parName = self.par_names[ipar]
            if parName in keys:
                startingParam = self.minuit_dict[parName]
                self.old_dict.update({parName: self.minuit_dict[parName]})
            else:
                startingParam = 0
            key = 'error_{0}'.format(parName)
            if key in keys:
                errorParam = self.minuit_dict[key]
            else:
                errorParam = 1
            key = 'limit_{0}'.format(parName)
            if key in keys:
                loParam = self.minuit_dict[key][0]
                hiParam = self.minuit_dict[key][1]
            else:
                loParam, hiParam = 0, 0

            self.gMinuit.DefineParameter(ipar, parName, startingParam,
                                         errorParam, loParam, hiParam)

            # fix it if the self.minuit_dict says so
            key = 'fix_{0}'.format(parName)
            if key in keys:
                if self.minuit_dict[key]:
                    self.gMinuit.FixParameter(ipar)
                else:
                    # likewise release it
                    self.gMinuit.Release(ipar)

        self.nCalls = 0
        self.nCallsDerivative = 0

    def chisq(self, npar, gin, f, par, iflag):
        """chisquare fit

        Parameters
        ----------
        npar

        gin

        f

        par

        iflag

        Notes
        -----
        Called from the doFit routine; updates gMinuit by giving
        the chisq and calculating the derivatives and such

        """

        # convert params to the input dictionary
        in_dict = {}
        for ipar in range(self.npar):
            in_dict.update({self.par_names[ipar]: par[ipar]})

        chisquared = self.gFitFunc(**in_dict)
        self.nCalls += 1

        # printout
        if self.verbosity >= 2:
            print('minuit_fit: \tChi2 = ', '{0: .4e}'.format(chisquared),
                  '\n\t\tkey\tvalue\t\tdelta\t\terror')
            for ipar in range(len(in_dict.keys())):
                key = self.par_names[ipar]
                aVal = ROOT.Double(0.23)
                errVal = ROOT.Double(0.24)
                self.gMinuit.GetParameter(ipar, aVal, errVal)
                print(
                    '\t\t', key, '\t', '{0: .4e}'.format(in_dict[key]),
                    '\t', '{0: .4e}'.format(in_dict[key] - self.old_dict[key]),
                    '\t', '{0: .4e}'.format(errVal))
            self.old_dict = in_dict.copy()
            if iflag != self.iflag:
                print('iflag has changed from', self.iflag, 'to', iflag)
                self.iflag = iflag

        # return result
        f[0] = chisquared

        if (iflag == 2):
            if (self.gGradFunc):

                # not currently called in calcAll
                dChi2dpar = self.gGradFunc(in_dict, self.gFitFunc,
                                           **self.grad_dict)
                if self.verbosity >= 2:
                    print('minuit_fit: \tdChi2dpar',
                          '\n\t\tkey\tderivative\tvalue\t\tstep')
                    for ipar in range(len(in_dict.keys())):
                        key = self.par_names[ipar]
                        print('\t\t', key,
                              '\t', '{0: .4e}'.format(dChi2dpar[key]),
                              '\t', '{0: .4e}'.format(in_dict[key]),
                              '\t', '{0: .4e}'.format(
                                  self.grad_dict['h_base'] *
                                  self.grad_dict['error_dict']
                                  ['error_{0}'.format(key)]))
                gin.SetSize(self.npar)  # need to handle root bug
                self.nCallsDerivative += 1
                #
                # fill gin with Derivatives
                #
                for ipar in range(self.npar):
                    # dChi2dpar is a dictionary!
                    gin[ipar] = dChi2dpar[self.par_names[ipar]]

    def doFit(self):

        """ do the fit

        Attributes
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Make sure you run setupFit first!

        """

        # arglist is for the parameters in Minuit commands
        arglist = array('d', 10 * [0.])
        ierflg = ROOT.Long(1982)

        arglist[0] = self.force_derivatives
        # =0 means to check gradient each time; 1 means never
        self.gMinuit.mnexcm("SET GRADIENT", arglist, 1, ierflg)

        # tell Minuit to use strategy for fastest fits
        arglist[0] = self.strategy  # was 1
        self.gMinuit.mnexcm("SET STRATEGY", arglist, 1, ierflg)

        # start timer
        self.startingtime = time.clock()

        # Now ready for minimization step
        self.gMinuit.SetMaxIterations(self.max_iterations)
        #self.gMinuit.Migrad()
        arglist[0] = self.max_iterations
        # Number of calls to FCN before giving up.
        arglist[1] = self.tolerance  # Tolerance;  0.001 * tolerance * UP (1.0)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)

        self.migrad_ierflg = np.int(ierflg)

        # done, check elapsed time
        firsttime = time.clock()
        self.deltatime = firsttime - self.startingtime
        if self.verbosity >= 1:
            print('minuit_fit: Elapsed time fit = ', self.deltatime)

        # number of calls
        if self.verbosity >= 1:
            print('minuit_fit: Number of CalcAll calls = ', self.nCalls)
            print('minuit_fit: Number of CalcDerivative calls = ',
                  self.nCallsDerivative)

    def outFit(self):
        """ return the results of the fit

        Parameters
        ----------
        None

        Returns
        -------
        output_dict : dictionary
            Dictionary with several Minuit results

        """

        output_dict = {}

        # generic statuses
        status = self.gMinuit.GetStatus()
        output_dict.update({'status': {
            'GetStatus': status,
            'migrad_ierflg': self.migrad_ierflg,
            'par_names': self.par_names,
            'deltatime': self.deltatime,
            'force_derivatives': self.force_derivatives,
            'strategy': self.strategy,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'startingtime': self.startingtime,
            'verbosity': self.verbosity,
            'grad_dict': self.grad_dict,
            'npar': self.npar,
            'nCalls': float(self.nCalls),
            'nCallsDerivative': float(self.nCallsDerivative)
            }})

        # get more fit details from MINUIT
        amin, edm, errdef = ROOT.Double(0.18), ROOT.Double(0.19), ROOT.Double(0.20)
        nvpar, nparx, icstat = ROOT.Long(1983), ROOT.Long(1984), ROOT.Long(1985)
        self.gMinuit.mnstat(amin, edm, errdef, nvpar, nparx, icstat)
        if self.verbosity >= 1:
            mytxt = "amin = %.3f, edm = %.3f,   effdef = %.3f,   nvpar = %.3f,  nparx = %.3f, icstat = %.3f " % (amin,edm,errdef,nvpar,nparx,icstat)
            print('minuit_fit: ', mytxt)

        output_dict.update({'mnstat': {
            'amin': float(amin),
            'edm': float(edm),
            'errdef': float(errdef),
            'nvpar': float(nvpar),
            'nparx': float(nparx),
            'icstat': float(icstat),
            'nCalls': float(self.nCalls),
            'nCallsDerivative': float(self.nCallsDerivative)}})

        # get fit values and errors
        aVal = ROOT.Double(0.21)
        errVal = ROOT.Double(0.22)

        # also write the error matrix to the fits file, as a 2nd image
        covmat = ROOT.TMatrixDSym(self.npar)
        self.gMinuit.mnemat(covmat.GetMatrixArray(), self.npar)
        covariance = np.zeros((self.npar, self.npar))
        corr = np.zeros((self.npar, self.npar))

        output_dict.update({'minuit': {}, 'args': {}, 'errors': {}})
        for ipar in range(self.npar):
            self.gMinuit.GetParameter(ipar, aVal, errVal)

            par_name = self.par_names[ipar]
            output_dict['minuit'].update({par_name: np.float64(aVal)})
            output_dict['args'].update({par_name: np.float64(aVal)})

            if errVal < 1e9:
                output_dict['minuit'].update({'error_{0}'.format(par_name):
                                              np.float64(errVal)})
                output_dict['errors'].update({'{0}'.format(par_name):
                                              np.float64(errVal)})

            else:
                output_dict['minuit'].update({'error_{0}'.format(par_name): 0})
                output_dict['errors'].update({'{0}'.format(par_name): 0})

            # error matrix part
            for jpar in range(self.npar):
                covariance[jpar, ipar] = covmat[jpar][ipar]
                corr[jpar, ipar] = covmat[jpar][ipar] / \
                    np.sqrt(covmat[jpar][jpar] * covmat[ipar][ipar])

        output_dict.update({'covariance': covariance, 'correlation': corr})

        # printout parameters in a convenient format
        if self.verbosity >= 1:
            self.gMinuit.mnprin(3, amin)

        return output_dict
