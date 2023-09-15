
import numpy as np 
import scipy.integrate as integ
import astropy.coordinates as coord 


class MWDecayFlux(object):
	""" 
	Generate model for the fluxes do to DM decays 
	in the MW halo along several line-of-sight
	"""
	def __init__(self, name, D, sigma_lambda):
		""" """
		self.name = name
		self.D = D 
		self.sigma_lambda = sigma_lambda

	def __call__(self, lam, lam0, decay_rate):
		arg = (lam - lam0)/self.sigma_lambda
		norm = self.sigma_lambda*np.sqrt(2*np.pi)
		sepectral_response = np.exp(-0.5*arg**2)/norm
		return sepectral_response*self.D*decay_rate/(4*np.pi)

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