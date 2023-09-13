
import numpy as np 
import scipy.integrate as integ
import astropy.coordinates as coord 


class MWDecayFlux(object):
	""" 
	Generate model for the fluxes do to DM decays 
	in the MW halo along several line-of-sight
	"""
	def __init__(ra, dec):
		""" """
		self.coords = SkyCoord(ra=np.asarray(ra)*u.degree, 
			                   dec=np.asarray(dec)*u.degree)




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


def compute_Dfactors(ra, dec, profile):
	""" compute normalized D-factors for each line-of-sight """ 
	D = np.zeros(ra.shape)
	targets = coords.SkyCoord(ra=ra, dec=dec)
	for index in range(D.size):
		rho_los = lambda s: profile(s, targets[index].b, targets[index].l)	
		Di, Di_error = np.quad(rho_los, 0, np.inf)
		D[index] = Di
	return D


def NFWprofile_galcoords(s, b, l, d_s):
	""" 
	NFW profiled normalized to rho_s = 1 and r_s = 1,
	given as a function of the line-of-sight distance s
	from the earth, galactic coordinates b, l, and the 
	distance d_s from the sun to the galactic center. 
	"""
	x_sq = s**2 + d_s**2 - 2*d_s*s*np.cos(b)*np.cos(l)
	return 1.0/(np.sqrt(x_sq) + x_sq)