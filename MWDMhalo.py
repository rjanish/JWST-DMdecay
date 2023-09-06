
import numpy as np 
import scipy.integrate as integ


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


class NFWprofile(object):
	def __init__(self, rhos, rs):
		self.rhos = rhos
		self.rs = rs 

	def __call__(self, r):
		r = np.asarray(r)
		f1 = r/self.rs 
		f2 = 1 + f1
		return self.rhos/(f1*f2)
