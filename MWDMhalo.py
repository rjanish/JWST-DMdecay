
import numpy as np 
import scipy.integrate as integ
import scipy.interpolate as interp
import astropy.coordinates as coord 

import matplotlib.pyplot as plt


def MWDecayFlux_old(lam, lam0, decay_rate, D, sigma_lam):
	""" 
	Flux due to DM decays along a line-of-sight with
	the given D-factor, assuming a Guassian (in wavelength)
	spectral response with the given width. 
	"""
	arg = (lam - lam0).T/sigma_lam
	norm = sigma_lam*np.sqrt(2*np.pi)
	sepectral_response = (lam**2).T*np.exp(-0.5*arg**2)/norm
	return (sepectral_response*D*decay_rate/(4*np.pi)).T

class MWDecayFlux(object):
	""" 
	Flux due to DM decays along a line-of-sight with
	the given D-factor, assuming a Guassian (in wavelength)
	spectral response with the given width. 
	"""
	def __init__(self, sigma_doppler, vesc, lam0, sigma_inst, D):
		self.cs = ConvolvedSpectrum(sigma_doppler, vesc, lam0, sigma_inst)
		self.D = D

	def __call__(self, lam, rate):
		return self.cs(lam)*self.D*rate/(2)  # 2 instead of 4pi -> convert dE to dnu

class MWchisq_nobackground(object):
	"""
	The chisq with no attempt to model background
	(see Cirelli et al 2021, eqn 21)
	"""
	def __init__(self, data, model, selector=None):
		self.data = data 
		self.model = model 
		self.Nrows = len(data)
		self.selector = selector

	def __call__(self, decay_rate, lam0, shift=0.0, mode=None):
		by_spectra = np.zeros(self.Nrows)
		for index, row in enumerate(self.data):
			if self.selector == "any":
				valid = (row["sky"] > 0) & (row["error"] > 0)
			if self.selector == "strictly":
				if np.any(row["sky"] < 0):
					continue
				valid = (row["sky"] > 0) & (row["error"] > 0)
			model = self.model(row["lam"][valid], lam0, decay_rate, 
				               row["D"], row["max_res"])
			diff = model - row["sky"][valid]
			diff[diff < 0] = 0.0
			chisq_i = (diff/row["error"][valid])**2
			by_spectra[index] = np.sum(chisq_i)
		if mode == "total":
			return np.sum(by_spectra) - shift
		elif mode == "itemized":
			return by_spectra - shift

class MWchisq_powerlaw(object):
	"""
	The chisq with no attempt to model background
	(see Cirelli et al 2021, eqn 21)
	"""
	def __init__(self, data, model):
		self.data = data 
		self.model = model 
		self.Nrows = len(data)

	def __call__(self, params, lam0, shift=0.0):
		decay_rate = params[0]
		A = params[1:1 + self.Nrows]
		p = params[1 + self.Nrows:]
		total = 0.0 
		for index, row in enumerate(self.data):
			valid = row["error"] > 0
			dm = self.model(row["lam"][valid], lam0, decay_rate, 
			                row["D"], row["max_res"])
			lam_scaled = row["lam"][valid]
			background = A[index]*(lam_scaled**p[index])
			model = background + dm 
			diff = model - row["sky"][valid]
			diff[diff < 0] = 0.0
			chisq_i = (diff/row["error"][valid])**2
			total += np.sum(chisq_i)
		return total - shift

class ShiftedGaussian_lambda(object):
	""" 
	Model for the spectrum due to two-photon decays 
	for a gas of non-relativistic identical particles 
	with a truncated Maxwellian velocity distrbution.
	This is df/dE as a functino of lambda 
	"""
	def __init__(self, sigma_v, vesc, lam0):
		"""
		sigma: Maxwellian velocity dispersion 
		vesc: maximum speed 
		m: mass of parent 
		"""
		self.w = sigma_v*lam0
		self.lam_esc = vesc*lam0
		self.lam0 = lam0
		self.y_esc = vesc/(np.sqrt(2)*sigma_v)
		# set normalization factor
		integral = integ.quad(self.scaled_speed_distribution, 
			                  0, self.y_esc)
		self.Nv, self.Nv_error = np.asarray(integral)*4/np.sqrt(np.pi)
		# distribution factors
		self.arg_max = 0.5*(self.lam_esc**2)/(self.w**2)
		self.prefactor = (self.lam0**2)/(self.Nv*self.w*(2*np.pi)**1.5)

	def scaled_speed_distribution(self, y):
		return (y**2)*np.exp(-y**2)

	def __call__(self, lam):
		lam = np.asarray(lam)
		out = np.zeros(lam.shape)
		in_range = np.abs(lam - self.lam0) < self.lam_esc
		arg = 0.5*(lam[in_range] - self.lam0)**2/self.w**2
		out[in_range] = self.prefactor*(np.exp(-arg) - np.exp(-self.arg_max))
		return out

class ConvolvedSpectrum(object):
	""" 
	Model for the spectrum due to two-photon decays 
	for a gas of non-relativistic identical particles 
	with a truncated Maxwellian velocity distrbution, 
	convolved with a Gaussian instrumental response. 
	"""
	def __init__(self, sigma_doppler, vesc, lam0, sigma_inst):
		"""
		sigma: Maxwellian velocity dispersion 
		vesc: maximum speed 
		m: mass of parent 
		"""
		self.dfdE_doppler = ShiftedGaussian_lambda(sigma_doppler, vesc, lam0)
		self.lam_min = lam0*(1 - vesc)
		self.lam_max = lam0*(1 + vesc)
		self.sigma_inst = sigma_inst
		self.gauss_factor = 1.0/(sigma_inst*np.sqrt(2*np.pi))

	def gaussian(self, lam):
		return self.gauss_factor*np.exp(-lam**2/(2*self.sigma_inst**2))

	def convolution_integrand(self, lam_prime, lam):
		return self.dfdE_doppler(lam_prime)*self.gaussian(lam - lam_prime)

	def __call__(self, lam):
		out = np.zeros(lam.size)
		for index, lam_i in enumerate(lam):
			integral = integ.quad(self.convolution_integrand, 
				                  self.lam_min, self.lam_max, args=(lam_i))
			out[index] = integral[0]
		return out

def compute_halo_Dfactor(b, l, profile, d_s):
	""" 
	compute normalized D-factor for given line-of-sight,
	assuming profile is a 1d radial mass distribution 
	""" 
	rho_los = lambda s: profile(rsq_from_galcenter(s, b, l, d_s))	
	D, D_error = integ.quad(rho_los, 0, np.inf)
	return D

def NFWprofile(x):
	"""
	NFW profile normalzied to rho_s = 1 and r_s = 1
	"""
	return 1.0/(x*(1.0 + x))

def rsq_from_galcenter(s, b, l, d_s):
	""" 
	Given a point p at galactic coords (b, l) in degrees 
	and a line-of-sight distance s from the earth, this
	returns the square of the distance from 
	p to the galactic center, where d_s is the distance 
	from the sun to the galactic center. 
	"""
	b_rad, l_rad = np.array([b, l])*(np.pi/180.0)
	return s**2 + d_s**2 - 2*d_s*s*np.cos(b)*np.cos(l)