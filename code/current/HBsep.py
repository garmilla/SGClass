# Author - Ross Fadely
#
import numpy as np
np.seterr(all='warn')
#np.seterr(divide='ignore')
import ctypes as ct
import pyfits as pf

from scipy.optimize import fmin_l_bfgs_b
from scipy.integrate import romberg, simps, quad
import matplotlib.pyplot as plt
from utils import *
from lum_funcs import *

import pdb
import pickle

def zero_to_floor(array, floor = 1.e-100):
    """
    Change the zero values in an array to floor
    """
    change = np.where(array.ravel() == 0.0)[0]
    for i in change:
        array.ravel()[i] = floor

def floor_to_zero(array, floor = 1.e-100):
    """
    Change the values in an array that are less than floor
    to zero
    """
    change = np.where(array.ravel() < floor)[0]
    for i in change:
        array.ravel()[i] = 0.0

class HBsep(object):
    """
    Hierarchical Bayesian classification of astronomical
    objects, using photometry.
    """
    def __init__(self, ra, dec, maglims, filts, class_labels, Nzs, z_maxs=None, z_min=0., method=1, zrefs=None):

	self.l, self.b = eq2gal(ra,dec)
	self.maglims = maglims
	self.filts = filts
        self.Nzs = Nzs
        self.z_min = z_min
	self.method = method
	self.num = 101

        if z_maxs is None:
            self.z_maxs = np.zeros(len(Nzs))
        else:
            self.z_maxs = z_maxs

        self.class_labels = class_labels
        self.Nclasses = len(class_labels)

        self.model_mags = {}
        self.model_fluxes = {}
	if self.method == 2:
	    # Initialize cosmological parameters
	    self.O0 = 0.272
	    self.Ol = 0.728
	    self.H0 = 69.32 # In km/s/Mpc
            self.h = 0.6932 # From WMAP
	    # Speed of light in km/s
            self.c = 3.0e5 # in km/s
	    self.zrefs = {}
	    for (i, c) in enumerate(class_labels):
	        self.zrefs[c] = zrefs[i] # Reference redshifts, typically z_min
	    pc = 3.08567758e18 # Parsec in cm
	    self.rsun = 8.0*1.0e3*pc #Distance from the sun to the galactic center

    def get_filter_norm(self, filter_list_path):
        """
        Return the normalizing flux in AB for given
        filter list.
        """
        f = open(filter_list_path)
        self.Nfilter = len(f.readlines())
        f.close
        self._make_models(None, filter_list_path, 1, 0., 0.,
                          np.zeros((2, 2)).astype(np.float64), True)

    def _make_models(self, sed_list_path, filter_list_path, Nz, zmin, zmax,
                     models, filter_only=False):
        """
        Prepare for and call model maker.
        """

        #pdb.set_trace()
        # load funtion
        model_maker = ct.CDLL('./_model_maker.so')

        # initialize filter normalizations
        self.filter_norms = np.zeros(self.Nfilter).astype(np.float64)

        # pointers for filters
        filt_norm_p = self.filter_norms.ctypes.data_as(ct.POINTER(ct.c_double))
        filt_list_p = ct.c_char_p(filter_list_path)

        # ctypes foo
        Nz = ct.c_long(np.int64(Nz))
        zmin = ct.c_double(np.float64(zmin))
        zmax = ct.c_double(np.float64(zmax))
        models_p = ctype_2D_double_pointer(models)

        # filters only, or models too?
        if filter_only:
            sed_list_p = filt_list_p
            filter_only = ct.c_long(np.int64(1))
        else:
            sed_list_p = ct.c_char_p(sed_list_path)
            filter_only = ct.c_long(np.int64(0))

        model_maker.model_maker(filt_list_p, sed_list_p, filter_only, Nz, zmin,
                                zmax, filt_norm_p, models_p)

    def create_models(self, filter_list_path, list_of_sed_list_paths,
                      normalize_models=True):
        """
        For each class, produce models over redshifts
        """

        # specify Nfilter
        f = open(filter_list_path)
        self.Nfilter = len(f.readlines())
        f.close

        for i in range(self.Nclasses):
            key = self.class_labels[i]
            sed_list_path = list_of_sed_list_paths[i]

	    if key == 'star' and self.method == 2:
	        self.kms = get_star_kms(sed_list_path)

            # get number of seds in class
            f = open(sed_list_path)
            Nseds = len(f.readlines())
            f.close

            # init
            self.model_fluxes[key] = \
                np.zeros((self.Nzs[i] * Nseds,
                          self.Nfilter)).astype(np.float64)

            # make the models
            self._make_models(sed_list_path, filter_list_path, self.Nzs[i],
                              self.z_min, self.z_maxs[i],
                              self.model_fluxes[key])

            # remove models with zero fluxes
            #hasZero = np.any(self.model_fluxes[key] == 0.0, axis=1)
	    #remove = np.where(hasZero)[0]
	    #if len(remove) > 0:
	    #    self.model_fluxes[key] = np.delete(self.model_fluxes[key], remove, axis=0)

	    # change zeros for floor to avoid problems
	    zero_to_floor(self.model_fluxes[key])

            # normalize models
            if normalize_models:
                self.model_fluxes[key] /= np.mean(self.model_fluxes[key],
                                                  axis=1)[:, None]

            # compute magnitudes
            self.model_mags[key] = -2.5 * np.log10(self.model_fluxes[key] /
                                                   self.filter_norms[None, :])

            print '\nCreated '+key+' models'

    def read_and_process_data(self, data,
                              missing_mags=None,
                              limiting_mags=None,
                              limiting_sigmas=None,
                              normalize_flux=True,
                              inflation_factor=1e6):
        """
        Read in data location, process.
        """
        self.missing_mags = missing_mags
        self.limiting_mags = limiting_mags
        self.limiting_sigmas = limiting_sigmas

        # read and process data
        data = self.read_data(data)
        self.Ndata = data.shape[0]
        self.mags = data[:, :self.Nfilter]
        self.mag_errors = data[:, self.Nfilter:]
        self.process_data(inflation_factor, normalize_flux)

    def read_data(self, data):
        """
        Read in data file or array, do some moderate
        error checking.
        """
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, str):
            try:
                # hdu number might be an issue
                f = pf.open(data)
                tdata = f[1].data
                f.close()
                data = np.zeros((len(tdata.field(0)), len(tdata[0])))

                for m in range(data.shape[1]):
                    data[:, m] = tdata.field(m)
            except:
                try:
                    data = np.loadtxt(data)
                except:
                    print '\nData file not read.'
                    assert False

        self.shape_check(data, 'Data Error:')
        return data

    def shape_check(self, data, item):
        """
        Simple shape checks
        """
        assert len(data.shape) == 2, '\n\n' + item + \
            ' Must be a 2D numpy array'
        assert data.shape[0] > data.shape[1], '\n\n' + item + \
            ' Must have more rows than columns'
        assert np.mod(data.shape[1], 2) == 0, '\n\n' + item + \
            ' Ncolumn is not even, what gives?'

    def process_data(self, inflation_factor, normalize_flux):
        """
        Turn data into flux, flux err
        """
        # must calculate normalization first
        try:
            print '\nFilter AB normalizations: ', self.filter_norms
            print
        except:
            assert False, 'Must run make_model or get_filter_norm'

        # normal flux calculation
        self.fluxes = 10.0 ** (-0.4 * self.mags) * self.filter_norms[None, :]
        self.flux_errors = 10.0 ** (-0.4 * self.mags) * \
            self.filter_norms[None, :] * np.log(10.0) * 0.4 * self.mag_errors

        # account for missing data
        if self.missing_mags is not None:
            Nind, find = np.where(self.mags == self.missing_mags)
            ind = np.where(self.fluxes > 0)[0]
            self.fluxes[Nind,find] = np.median(self.fluxes[ind])
            self.flux_errors[Nind,find] = self.fluxes[Nind,find] * \
                inflation_factor

        # account for anything fainter than limiting magnitudes
        if self.limiting_mags is not None:
            assert self.limiting_sigmas is not None, \
                'Must specify Nsigma for limiting mags'
            for i in range(self.Nfilter):
                ind = np.where(self.mags[:,i] >= self.limiting_mags[i])[0]
                self.fluxes[ind, i] = 0.0
                self.flux_errors[ind, i] = 10.0**(-0.4*self.limiting_mags[i]) \
                    * self.limiting_sigmas[i]

        # normalize if desired
        if normalize_flux:
            ind = np.where(self.fluxes != 0.0)[0]
            self.flux_norm = np.mean(self.fluxes[ind])
            self.fluxes /= self.flux_norm
            self.flux_errors /= self.flux_norm

    def fit_models(self):
        """
        Produce chi2 and coefficients of model fits to data.
        """
        fitter = ct.CDLL('./_fit_models.so')

        self.chi2s = {}
        self.coeffs = {}
        self.coefferrs = {}
        for i in range(self.Nclasses):

            key = self.class_labels[i]
            Nmodel = self.model_fluxes[key].shape[0]
            self.chi2s[key] = np.zeros((self.Ndata, Nmodel))
            self.coeffs[key] = np.zeros((self.Ndata, Nmodel))
            self.coefferrs[key] = np.zeros((self.Ndata, Nmodel))

            # ctypes prep
            Ndata = ct.c_long(self.Ndata)
            Nmodel = ct.c_long(Nmodel)
            Nfilter = ct.c_long(self.Nfilter)
            datap = ctype_2D_double_pointer(self.fluxes)
            dataerrp = ctype_2D_double_pointer(self.flux_errors)
            modelsp = ctype_2D_double_pointer(self.model_fluxes[key])
            chi2sp = ctype_2D_double_pointer(self.chi2s[key])
            coeffsp = ctype_2D_double_pointer(self.coeffs[key])
            coefferrsp = ctype_2D_double_pointer(self.coefferrs[key])

            fitter.fit_models(Ndata, Nmodel, Nfilter, modelsp, datap, dataerrp,
                              coeffsp, coefferrsp, chi2sp)

    def coefficient_marginalization(self, Nstep_factor=2, Nsigma=5,
                                    delta_chi2_cut=32., floor=1e-100):
        """
        Marginalize over the fit coefficients.
        """
        Ndata = ct.c_long(self.Ndata)
        Nstep = ct.c_long((Nsigma * Nstep_factor) * 2 + 1)
        Nsigma = ct.c_double(Nsigma)
        det_flux_errors = np.prod(self.flux_errors, axis=1)
        det_flux_errorsp = det_flux_errors.ctypes.data_as(
            ct.POINTER(ct.c_double))
        delta_chi2_cut = ct.c_double(delta_chi2_cut)

        marg = ct.CDLL('./_coeff_marginalization.so')
        self.calc_coeff_priors()

        self.ignored = np.zeros(self.Ndata)
        self.bad_fit_flags = {}
        self.coeff_marg_like = {}
        for i in range(self.Nclasses):

            key = self.class_labels[i]
            Nmodel = self.model_fluxes[key].shape[0]
            minchi2 = np.min(self.chi2s[key], axis=1)
            self.coeff_marg_like[key] = np.zeros((self.Ndata, Nmodel))

            # ctypes prep
            Nmodel = ct.c_long(Nmodel)
            minchi2p = minchi2.ctypes.data_as(ct.POINTER(ct.c_double))
            prior_varsp = self.coeff_prior_vars[key].ctypes.data_as(
                ct.POINTER(ct.c_double))
            prior_meansp = self.coeff_prior_means[key].ctypes.data_as(
                ct.POINTER(ct.c_double))
            chi2sp = ctype_2D_double_pointer(self.chi2s[key])
            coeffsp = ctype_2D_double_pointer(self.coeffs[key])
            coefferrsp = ctype_2D_double_pointer(self.coefferrs[key])
            marglikep = ctype_2D_double_pointer(self.coeff_marg_like[key])

            marg.coeff_marginalization(Nstep, Ndata, Nmodel, Nsigma, minchi2p,
                                       delta_chi2_cut, coeffsp, coefferrsp,
                                       prior_meansp, prior_varsp,
                                       det_flux_errorsp, chi2sp, marglikep)

	    floor_to_zero(self.coeff_marg_like[key])

            # Flag bad fits
            self.bad_fit_flags[key] = np.zeros(self.Ndata)
            ind = np.where(self.coeff_marg_like[key].sum(axis=1) < floor)[0]
            self.bad_fit_flags[key][ind] = 1
            self.ignored[ind] += 1

        # construct index for likelihood estimation
        ind = np.where(self.ignored < self.Nclasses)[0]
        self.use = ind
        ind = np.where(self.ignored == self.Nclasses)[0]
        self.ignore = ind

    def calc_coeff_priors(self):
        """
        Compute prior parms for coefficient fits.
        """
        self.coeff_prior_vars = {}
        self.coeff_prior_means = {}
        for i in range(self.Nclasses):

            key = self.class_labels[i]
            coeffs = self.coeffs[key]
            weights = 1./self.coefferrs[key]

            mean = np.sum(weights * np.log(coeffs), axis=0) / \
                np.sum(weights, axis=0)
            var = np.var(coeffs, axis=0)
            self.coeff_prior_means[key] = mean
            self.coeff_prior_vars[key] = var

    def func(self, z):
        return np.power(np.power(1+z, 2)*(self.O0*z+1)-self.Ol*z*(z+2), -0.5)

    def get_D(self, z):
        """
        Get comoving distance at redshift z
        """
        r = self.c/self.H0*romberg(self.func, 0, z)
        D = r
        return D

    def get_dVc(self, z, D = None):
        """
        Get comoving volume element at redshift z
        """
        if D == None:
            D = get_D(z)
        dVc = self.c/self.H0*D**2*\
	      np.power(np.power(1+z, 2)*(self.O0*z+1)-self.Ol*z*(z+2), -0.5)
        return dVc

    def init_D_dVc_grid(self):
        """
        Initialize grid of comoving distances
        """
        self.D = {}
        self.dVc = {}
        for i in range(self.Nclasses):
            Nz = self.Nzs[i]
            if Nz == 1:
                continue
            key = self.class_labels[i]
            zgrid = np.linspace(1.e-4, self.z_maxs[i], Nz)
	    self.D[key] = np.zeros((Nz,))
	    self.dVc[key] = np.zeros((Nz,))
	    for i in range(Nz):
	        self.D[key][i] = self.get_D(zgrid[i])
	        self.dVc[key][i] = self.get_dVc(zgrid[i], self.D[key][i])

    def Cstar_prior(self, C, key, m):
        Prior = np.power(C, -5.0/2)/\
	        np.sqrt(self.rsun**2+self.kms[m]/C-\
		        2*self.rsun*np.sqrt(self.kms[m]/C)*\
			np.cos(self.b)*np.cos(self.l))
        return Prior

    def Cz_prior(self, C, key, idx, zidx):
        ref = (np.abs(self.zgrid[key]-self.zrefs[key])).argmin()
        M = C*np.power(self.D[key][zidx]*(1+self.zgrid[key][zidx])/1.0e-5, 2)\
	   *self.model_fluxes[key][idx+ref]/self.filter_norms
	M = -2.5*np.log10(M)
	if key == 'galaxy':
	    Phi = Phi_gal(M[2], band='r') # Use r-band
	elif key == 'qso':
	    Phi = Phi_qso(M[3], self.zgrid[key][zidx], band='i') # Use i-band
	return Phi/C*self.dVc[key][zidx]

    def Chisqrd(self, n, m, C, key):
        temp = (self.fluxes[n]-C*self.model_fluxes[key][m])/self.flux_errors[n]
	temp = np.power(temp, 2)
	chisqrd = np.sum(temp)
	return chisqrd

    def data_lkhood(self, n, m, C, key):
        log_lkhood = - np.sum(np.log(self.flux_errors[n]*2*np.sqrt(np.pi)))\
	             - 0.5*self.Chisqrd(n, m, C, key)
	return np.exp(log_lkhood)

    def star_margin_func(self, C, key, n, m):
        margin_func = self.data_lkhood(n,m,C,key)*\
                      self.Cstar_prior(C,key,m)/\
                      self.c_normal[key][m]
        return margin_func

    def z_margin_func(self, C, key, n, idx, zidx):
        m = idx+zidx
        margin_func = self.data_lkhood(n,m,C,key)*\
                      self.Cz_prior(C,key,idx,zidx)/\
                      self.c_normal[key][m]
        return margin_func

    def marginalize(self, func, a, b, args=()):
        x = np.linspace(a, b, num=self.num)
	y = np.zeros(x.shape)
	n_args = len(args)
	for i in range(len(x)):
	    if n_args == 2:
	        y[i] = func(x[i], args[0], args[1])
	    elif n_args == 3:
	        y[i] = func(x[i], args[0], args[1], args[2])
	    elif n_args == 4:
	        y[i] = func(x[i], args[0], args[1], args[2], args[3])
	#plt.figure()
	#plt.semilogx(x, y)
	#plt.show()
	#integral = simps(y, x)
	dx = x[1]-x[0]
	integral = np.sum(y*dx)
	return integral

    def init_Cstarpriors(self, key, Ntemplate):
        """
        Find the limits for C from the magnitude limits of the sample
        """
        self.Clims[key] = np.zeros((Ntemplate, 2))
        self.c_normal[key] = np.zeros((Ntemplate,))
	Clims = np.zeros((len(self.filts),2))
	for i in range(Ntemplate):
	    for j in range(len(self.filts)):
	        Clims[j] = np.power(10.0, -5.0/2*self.maglims)*\
	                   self.filter_norms[self.filts[j]]/\
	    	           self.model_fluxes[key][i][j]
	    self.Clims[key][i][0] = np.amin(Clims[:,0])
	    self.Clims[key][i][1] = np.amax(Clims[:,1])
	    self.c_normal[key][i] = quad(self.Cstar_prior,\
	                            self.Clims[key][i][0],\
	                            self.Clims[key][i][1],\
	     			    args=(key, i))[0]

    def init_Czpriors(self, key, Ntemplate, Nz):
        """
        Find the limits for C from the magnitude limits of the sample
        """
        self.Clims[key] = np.zeros((Ntemplate*Nz, 2))
        self.c_normal[key] = np.zeros((Ntemplate*Nz,))
	Clims = np.zeros((len(self.filts),2))
	for i in range(Ntemplate):
	    for j in range(Nz):
	        for k in range(len(self.filts)):
	            Clims[k] = np.power(10.0, -5.0/2*self.maglims)*\
	                       self.filter_norms[self.filts[k]]/\
	    	               self.model_fluxes[key][i*Nz+j][k]
	        self.Clims[key][i*Nz+j][0] = np.amin(Clims[:,0])
		self.Clims[key][i*Nz+j][1] = np.amax(Clims[:,1])
	        self.c_normal[key][i*Nz+j] = quad(self.Cz_prior,\
		                             self.Clims[key][i*Nz+j][0],\
		                             self.Clims[key][i*Nz+j][1],\
		 			     args=(key, i*Nz,j))[0]

    def fixed_c_marginalization(self):
        """
        Marginalize over fixed priors for C's
        """
        self.abs_mags = {}
	self.Clims = {}
	self.c_normal = {}
	self.zgrid = {}
        self.coeff_marg_like = {}
        for i in range(self.Nclasses):
            Nz = self.Nzs[i]
            key = self.class_labels[i]
	    print "\nMarginalizing for class " + key
            self.abs_mags[key] = np.zeros(self.model_mags[key].shape)
            Nmodel = self.model_fluxes[key].shape[0]
            self.coeff_marg_like[key] = np.zeros((self.Ndata, Nmodel))
	    Ntemplate = Nmodel/Nz
            if Nz == 1:
	        assert key == 'star'
		print "\nInitializing star C priors"
	        self.init_Cstarpriors(key, Ntemplate)
		print "\nComputing C's marginalized likelihoods for stars"
       	        for j in range(Ntemplate):
		    print "Template {0}".format(j)
		    for n in range(self.Ndata):
		        self.coeff_marg_like[key][n,j] =\
			                  self.marginalize(self.star_margin_func,\
		                          self.Clims[key][j][0],\
		                          self.Clims[key][j][1],\
		 	         	  args=(key,n,j))
	    else:
                self.zgrid[key] = np.linspace(1.e-4, self.z_maxs[i], Nz)
		print "\nInitializing {0} C priors".format(key)
	        self.init_Czpriors(key, Ntemplate, Nz)
		print "\nComputing C's marginalized likelihoods for {0}".format(key)
       	        for j in range(Ntemplate):
	            for k in range(Nz):
		        for n in range(self.Ndata):
		            self.coeff_marg_like[key][n,j*Nz+k] =\
			                         self.marginalize(self.z_margin_func,\
		                                 self.Clims[key][j*Nz+k][0],\
		                                 self.Clims[key][j*Nz+k][1],\
		 			         args=(key,n,j*Nz,k))

    def apply_and_marg_redshift_prior(self):
        """
        Apply redshift prior and margninalize over redshift
        """
        self.zc_marg_like = {}
        for i in range(self.Nclasses):
            key = self.class_labels[i]

            Nz = self.Nzs[i]
            Nmodel = self.model_fluxes[key].shape[0]
            Ntemplate = Nmodel/Nz

            if Nz == 1 or self.method == 2:
                self.zc_marg_like[key] = self.coeff_marg_like[key]
                continue
            zgrid = np.linspace(1.e-4, self.z_maxs[i], Nz)

            self.zc_marg_like[key] = np.zeros((self.Ndata, Ntemplate))
            if self.method == 1:
                # shape = Nmodel,Nz
                z_medians = np.array([np.ones(Nz) * zmed
                                      for zmed in self.z_medians[key]])
		if np.any(z_medians == 0.0):
		    print "z_medians is zero!"
		#with open('z_medians.pkl', 'w') as f:
		#    pickle.dump(z_medians, f)
		#with open('z_pow.pkl', 'w') as f:
		#    pickle.dump(self.z_pow[key], f)
		#with open('zgrid.pkl', 'w') as f:
		#    pickle.dump(zgrid, f)
                z_prior = zgrid**self.z_pow[key] * \
                    np.exp(-(zgrid/z_medians)**self.z_pow[key])
		if np.any(z_prior.sum(axis=1)[:, None] == 0.0):
		    print "z_prior.sum(axis=1) is zero!"
		    print "z_grid", zgrid
	        floor_to_zero(z_prior)
                z_prior /= z_prior.sum(axis=1)[:, None]
                prior_weighted_like = self.coeff_marg_like[key] * \
                    z_prior.ravel()[None, :]
                for j in range(Ntemplate):
                    self.zc_marg_like[key][:, j] = \
                        np.sum(prior_weighted_like[:, j*Nz:j*Nz+Nz], axis=1)

    def assign_hyperparms(self, hyperparms):
        """
        Assign hyperparameters from a flattened list (from optimizer)
        """
        count = 0

        # assign template weights
        self.template_weights = {}
        for i in range(self.Nclasses):
            key = self.class_labels[i]
            Ntemplate = np.int(self.model_fluxes[key].shape[0] / self.Nzs[i])
            self.template_weights[key] = \
                np.exp(np.array(hyperparms[count:count+Ntemplate]))
	    if self.template_weights[key].sum() == 0.0:
	        print "self.template_weights[{0}].sum() is zero!".format(key)
            self.template_weights[key] /= self.template_weights[key].sum()
            count += Ntemplate

        # prior parms
        if self.method == 1:
            self.z_medians = {}
            for i in range(self.Nclasses):
                if self.Nzs[i] == 1 or self.method != 1:
                    continue
                key = self.class_labels[i]
                Ntemplate = np.int(self.model_fluxes[key].shape[0] / self.Nzs[i])
                self.z_medians[key] = \
                    np.array(hyperparms[count:count+Ntemplate])
                count += Ntemplate
            self.z_pow = {}
            for i in range(self.Nclasses):
                if self.Nzs[i] == 1 or self.method != 1:
                    continue
                key = self.class_labels[i]
                self.z_pow[key] = hyperparms[count:count+1]
                count += 1

        self.class_weights = np.exp(np.array(hyperparms[-self.Nclasses:]))
	if self.class_weights.sum() == 0.0:
	    print "self.class_weights.sum() is zero!"
        self.class_weights /= self.class_weights.sum()

    def calc_neg_lnlike(self, floor=1e-100):
        """
        Calculate marginalized likelihoods.
        """
        self.tzc_marg_like = {}
        self.marg_like = np.zeros(self.Ndata)
        ind = self.use
        for i in range(self.Nclasses):
            key = self.class_labels[i]
            self.tzc_marg_like[key] = np.zeros(self.Ndata)
            self.tzc_marg_like[key][ind] = \
                np.sum(self.zc_marg_like[key][ind] *
                       self.template_weights[key][None, :], axis=1)

            self.marg_like += np.maximum(self.tzc_marg_like[key] *
                                         self.class_weights[i], floor)

        self.neg_log_likelihood = -1.0 * np.sum(np.log(self.marg_like[ind]))

    def call_neg_lnlike(self, hyperparms):
        """
        Give this to optimizer to call.
        """
        weights = np.exp(np.array(hyperparms[-self.Nclasses:]))
        if np.Inf in weights:
            return np.Inf
        self.assign_hyperparms(hyperparms)
	if self.method == 1:
            self.apply_and_marg_redshift_prior()
        self.calc_neg_lnlike()
	if np.isnan(self.neg_log_likelihood):
	    self.neg_log_likelihood = np.Inf
	#print self.neg_log_likelihood
        return self.neg_log_likelihood

    def init_hyperparms(self, z_median, z_pow):
        """
        Initialize flattened list of hyperparameters.
        """
        # initialize parameters, barf
        p0 = np.array([])
        for i in range(self.Nclasses):
            key = self.class_labels[i]
            Ntemplate = self.model_fluxes[key].shape[0] / self.Nzs[i]
            p0 = np.append(p0, np.ones(Ntemplate) * np.log(1./Ntemplate))
        for i in range(self.Nclasses):
            if self.Nzs[i] != 1 and self.method == 1:
                key = self.class_labels[i]
                Ntemplate = self.model_fluxes[key].shape[0] / self.Nzs[i]
                p0 = np.append(p0, np.ones(Ntemplate) * z_median[i])
        for i in range(self.Nclasses):
            if self.Nzs[i] != 1 and self.method == 1:
                p0 = np.append(p0, z_pow[i])
        for i in range(self.Nclasses):
            p0 = np.append(p0, np.log(1./self.Nclasses))

        return p0

    def init_hyperparm_bounds(self):
        """
        Make bounds for fmin_l_bfgs_b
        """
        bounds = []
        for i in range(self.Nclasses):
            key = self.class_labels[i]
            Ntemplate = self.model_fluxes[key].shape[0] / self.Nzs[i]
            bounds.extend([(-1.*np.Inf, np.Inf) for j in range(Ntemplate)])
        for i in range(self.Nclasses):
            if self.Nzs[i] != 1 and self.method == 1:
                key = self.class_labels[i]
                Ntemplate = self.model_fluxes[key].shape[0] / self.Nzs[i]
                bounds.extend([(0.1, self.z_maxs[i]) for j in
                               range(Ntemplate)])
        for i in range(self.Nclasses):
            if self.Nzs[i] != 1 and self.method == 1:
                bounds.extend([(0., 2.)])
        for i in range(self.Nclasses):
            bounds.extend([(-1.*np.Inf, np.Inf)])

        return bounds

    def optimize(self, z_median=None, z_pow=None, init_p0=None,
                 eps=1.e-1, factr=1.e4, maxfun=15000):
        """
        Optimize using scipy's fmin_l_bfgs_b
        """
        if init_p0 is not None:
            p0 = init_p0
        else:
            p0 = self.init_hyperparms(z_median, z_pow)

        if self.method == 2:
	    print "Initializing redshift and comoving measures grid"
	    self.init_D_dVc_grid()
	    print "Marginalizing over C's"
	    self.fixed_c_marginalization()
	    self.apply_and_marg_redshift_prior()

        bounds = self.init_hyperparm_bounds()

        self.init_nll = self.call_neg_lnlike(p0)
        result = fmin_l_bfgs_b(self.call_neg_lnlike, p0, approx_grad=1,
                               bounds=bounds, epsilon=eps, factr=factr,
                               maxfun=maxfun, iprint=2)

        return self.call_neg_lnlike(result[0]), result[0]

    def get_relative_likelihoods(self):
        """
        Return array of relative likelihoods
        """
        relative_likelihoods = np.zeros((self.Ndata, self.Nclasses))
        for i in range(self.Nclasses):
            key = self.class_labels[i]
            relative_likelihoods[:, i] = self.tzc_marg_like[key] * \
                self.class_weights[i]

        relative_likelihoods /= relative_likelihoods.sum(axis=1)[:, None]
        return relative_likelihoods

    def write_fits_table(self, filename, data, labels):
        """
        Write a FITS data table to given `filename` usings `labels` for
        column names.
        """
        Nlabels = len(labels)
        if Nlabels > 1:
            cols = pf.ColDefs([pf.Column(name=labels[i], format='E',
                                         array=data[:, i])
                               for i in range(Nlabels)])
        else:
            cols = pf.ColDefs([pf.Column(name=labels[0], format='E',
                                         array=data[0,:])])

        tbhdu = pf.new_table(cols)
        hdu = pf.PrimaryHDU(np.arange(10))

        tblist = pf.HDUList([hdu, tbhdu])
        tblist.writeto(filename, clobber=True)

    def write_relative_likelihoods(self, filename):
        """
        Write relative marg. likelihoods to FITS file.
        """
        likes = self.get_relative_likelihoods()
        flags = np.ones(likes.shape[0])

        ind = np.in1d(self.ignore, np.arange(likes.shape[0]))
        flags[ind] = 0.0

        out = np.zeros((likes.shape[0], likes.shape[1]+1))
        out[:, :-1] = likes
        out[:, -1] = flags
        labels = [self.class_labels[i] for i in range(self.Nclasses)]
        labels.append('used')
        self.write_fits_table(filename, out, labels)

    def write_array(self, filename, array, label):
        """
        Write a 1D array to FITS table.
        """
        self.write_fits_table(filename, np.atleast_2d(array), label)

    def write_minchi2(self, filename):
        """
        Write the min. chi^2 for each class to FITS table.
        """
        labels = []
        minchi2s = np.zeros((self.Ndata, self.Nclasses))
        for i in range(self.Nclasses):
            key = self.class_labels[i]
            labels.append(key+' minchi2')
            minchi2s[:, i] = self.chi2s[key].min(axis=1)

        self.write_fits_table(filename, minchi2s, labels)
