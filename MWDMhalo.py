
import numpy as np 
import scipy.integrate as integ
import scipy.interpolate as interp
import astropy.coordinates as coord 

import matplotlib.pyplot as plt


def MWDecayFlux(lam, lam0, decay_rate, D, sigma_lam):
	""" 
	Flux due to DM decays along a line-of-sight with
	the given D-factor, assuming a Guassian (in wavelength)
	spectral response with the given width. 
	"""
	arg = (lam - lam0).T/sigma_lam
	norm = sigma_lam*np.sqrt(2*np.pi)
	sepectral_response = (lam**2).T*np.exp(-0.5*arg**2)/norm
	return (sepectral_response*D*decay_rate/(4*np.pi)).T

class MWchisq_nobackground(object):
	"""
	The chisq with no attempt to model background
	(see Cirelli et al 2021, eqn 21)
	"""
	def __init__(self, data, model, reg_factor):
		self.data = data 
		self.model = model 
		self.Nrows = len(data)
		self.regularize_data(reg_factor)

	def regularize_data(self, reg_factor):
		lam_min = np.zeros(self.Nrows)
		lam_max = np.zeros(self.Nrows)
		dlam_min = np.zeros(self.Nrows)
		for index, row in enumerate(self.data):
			lam_min[index] = row["lam"][0]
			lam_max[index] = row["lam"][1]
			dlam_min[index] = np.min(row["lam"][1:] - row["lam"][:-1])
		self.lam_start = np.min(lam_min)
		self.lam_end = np.max(lam_max)
		self.dlam = np.min(dlam_min)/reg_factor
		self.lam = np.arange(self.lam_start, self.lam_end, self.dlam)
		self.lam2D = np.repeat([self.lam], self.Nrows, axis=0)
		self.reg_sky = np.nan*np.ones((self.Nrows, self.lam.size))
		self.reg_error = np.nan*np.ones((self.Nrows, self.lam.size))
		self.reg_D = np.nan*np.ones(self.Nrows) 
		self.reg_res = np.nan*np.ones(self.Nrows)
		for index, row in enumerate(self.data):
			sky_func = interp.interp1d(row["lam"], row["sky"],
				                       bounds_error=False, 
				                       fill_value=np.nan)
			self.reg_sky[index, :] = sky_func(self.lam)
			error_func = interp.interp1d(row["lam"], row["error"],
				                       bounds_error=False, 
				                       fill_value=np.nan)  
			  # change this to properly propagate the errors 
			self.reg_error[index, :] = error_func(self.lam)
			self.reg_D[index] = row["D"]
			self.reg_res[index] = row["max_res"]

	def __call__(self, decay_rate, lam0, shift=0.0, method=None):
		if method == "loop":
			total = 0.0 
			for row in self.data:
				if np.any(row["sky"] < 0):
					continue 
					# I don't understand why of the observations 
					# have negative sky spectra, but they do. These
					# always contribute to chisq as defined here, since 
					# the model gives a strictly postive flux, and as a 
					# result they produce a chisq >> 1 for any DM decay 
					# rate. We need to understand what these negative
					# sky spectra mean. For now just skip them and only
					# use the stricly postive spectra.  
				valid = row["error"] > 0
				model = self.model(row["lam"][valid], lam0, decay_rate, 
					               row["D"], row["max_res"])
				diff = model - row["sky"][valid]
				diff[diff < 0] = 0.0
				chisq_i = (diff/row["error"][valid])**2
				total += np.sum(chisq_i)
		elif method == "interp":
			valid = (self.reg_sky > 0) & (self.reg_error > 0)
			model = self.model(self.lam2D, lam0, decay_rate, 
					           self.reg_D, self.reg_res)
			diff = model[valid] - self.reg_sky[valid]
			diff[diff < 0] = 0.0
			chisq_i = (diff/self.reg_error[valid])**2
			total = np.sum(chisq_i)
		return total - shift

class MWchisq_powerlaw(object):
	"""
	The chisq with no attempt to model background
	(see Cirelli et al 2021, eqn 21)
	"""
	def __init__(self, data, model):
		self.data = data 
		self.model = model 
		self.Nsets = len(data)

	def __call__(self, params, lam0, shift=0.0):
		decay_rate = params[0]
		A = params[1:1 + self.Nsets]
		p = params[1 + self.Nsets:]
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

class ShiftedGaussian(object):
	""" 
	Model for the spectrum due to two-photon decays 
	for a gas of non-relativistic identical particles 
	with a truncated Maxwellian velocity distrbution. 
	"""
	def __init__(self, sigma, vesc, m):
		"""
		sigma: Maxwellian velocity dispersion 
		vesc: maximum speed 
		m: mass of parent 
		"""
		self.sigma = sigma
		self.vesc = vesc
		self.m = m
		self.w = 0.5*m*sigma
		self.yesc = vesc/(sigma*np.sqrt(2)) 
		# set normalization factor
		integral = integ.quad(self.evaluate_scaled, 0, self.yesc)
		self.N, self.N_error = np.asarray(integral)*2/np.sqrt(np.pi)

	def evaluate_scaled(self, y):
		return np.exp(-y**2) - np.exp(-self.yesc**2)

	def __call__(self, E):
		E = np.asarray(E)
		y = np.abs(E - 0.5*self.m)/(self.w*np.sqrt(2)) 
		out = np.zeros(E.shape)
		prefactor = self.N*self.w*np.sqrt(2*np.pi)
		out[y < self.yesc] = self.evaluate_scaled(y[y < self.yesc])/prefactor
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
	Given a point p at galactic coords (b, l) and a 
	line-of-sight distance s from the earth, this
	returns the square of the distance from 
	p to the galactic center, where d_s is the distance 
	from the sun to the galactic center. 
	"""
	return s**2 + d_s**2 - 2*d_s*s*np.cos(b)*np.cos(l)