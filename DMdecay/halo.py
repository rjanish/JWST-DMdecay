
import numpy as np 
import scipy.integrate as integ


def sigma_from_fwhm(fwhm, lam0, sigma_v):
    sigma_inst = fwhm/(2*np.sqrt(2*np.log(2)))
    return np.sqrt(sigma_inst**2 + sigma_v**2)

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

class ShiftedGaussian_lambda(object):
	""" 
	Model for the spectrum due to two-photon decays 
	for a gas of non-relativistic identical particles 
	with a truncated Maxwellian velocity distribution.
	This is df/dE as a function of lambda 
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
	with a truncated Maxwellian velocity distribution, 
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
	rho_los = lambda s: profile(r_from_galcenter(s, b, l, d_s))	
	D, D_error = integ.quad(rho_los, 0, np.inf)
	return D

def NFWprofile(x):
	"""
	NFW profile normalized to rho_s = 1 and r_s = 1
	"""
	return 1.0/(x*(1.0 + x)**2)

# test test test 

def r_from_galcenter(s, b, l, d_s):
	""" 
	Given a point p at galactic coords (b, l) in degrees 
	and a line-of-sight distance s from the earth, this
	returns the distance from p to the galactic center, 
	where d_s is the distance 
	from the sun to the galactic center. 
	"""
	b_rad, l_rad = np.array([b, l])*(np.pi/180.0)
	return np.sqrt(s**2 + d_s**2 - 2*d_s*s*np.cos(b)*np.cos(l))