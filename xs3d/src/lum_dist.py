#! /usr/bin/python3
import numpy as np
from scipy.integrate import quad
import astropy.units as u
from astropy.coordinates import SkyCoord

from .start_messenge import Print

class Angdist:

	def __init__(self, H0 = 70, Omega_m = 0.30, Omega_l = 0.70, print_res = 0):
		self.H0 = H0 #71 #km/s/Mpc
		self.Omega_m = Omega_m
		self.Omega_l = Omega_l
		self.c = 299792.458
		self.print = print_res

		P=Print()
		self.P=P

	# Commoving radial distance X of a source at redshift self.z
	def X(self,a):
		t1=(self.c/self.H0)/np.sqrt(a*self.Omega_m+a**2*(1-self.Omega_m-self.Omega_l)+a**4*self.Omega_l)
		return t1


	def comv_distance(self, z):
		a0=1./(1+z)
		a1=1
		X = quad(self.X,a0,a1)
		dA=X[0]/(1+z)
		dL=X[0]*(1+z)
		scale=1*dA*1e6/206265.

		if self.print:

			self.P.out('Commov. dist [Mpc]', round(X[0],3))
			self.P.out('Lum. dist. [Mpc]', round(dL,3))
			self.P.out('scale [pc/arcs]', round(scale,3))

			#print ("Commoving_radial_distance = %0.1f"%X[0])
			#print ("Angular_distance [Mpc] = %0.1f"%dA)
			#print ("Luminosity_distance [Mpc] = %0.1f"%dL)
			#print ("Scale [pc/arcsec] = %0.3f"%scale)

		return dL, scale


	# Velocity transform.
	# V-Helio is the velocity due to the rotation of the Earth and the motion of the Earth in its orbit around the Sun.
	# V-LSRK is the velocity referenced to the local standard of rest, in addition to the motions of the Earth.
	# V-GAL is the velocity referenced to the center of the galaxy.
	# Measurements within our galaxy are usually referred to the LSR
	# Measurements of extragalactic objects are usually referred to the Sun (heliocentric or barycentric),
	# or sometimes to the galactic center (V_GAL).

	# NED's Velocity Conversion Calculator.
	# See also https://www.gb.nrao.edu/~fghigo/gbtdoc/doppler.html
	# l and b are the object's longitude and latitude, V is its unconverted velocity,
	# and the apices (with Galactic coordinates) of the various motions are given as:
	def vcor(self,corr_vel,header,frame='Helio2CMB'):

		vcor_tmp = 0

		# No velocity correction is applied by default
		if not corr_vel:
			return vcor_tmp

		transform = {'Helio2Gal'	:{'lapex': 87.8,	'bapex': 1.7,	'vapex':232.3},
					 'Helio2LG'		:{'lapex': 93.0,	'bapex': -4.0,	'vapex':316.0},
					 'Helio2CMB'	:{'lapex': 264.14,	'bapex': 48.26,	'vapex':371.0},
					 'Helio2LSRK'	:{'lapex':56.16,	'bapex': 22.76,	'vapex':20.0}
					}

		if np.all( [ k in header for k in ['CRVAL1','CRVAL2']] ):

			# target coordinates
			ra	= header['CRVAL1']
			dec	= header['CRVAL2']
			try:
				ra_  = eval(ra)  if type(ra) == str  else float(ra)
				dec_ = eval(dec) if type(dec) == str else float(dec)
			except(TypeError):
				return vcor_tmp

			# Define RA/Dec in degrees (defaulting to ICRS)
			c = SkyCoord(ra=ra_ * u.deg, dec=dec_ * u.deg, frame='icrs')

			# Convert to Galactic coordinates
			gal_coords = c.galactic

			l_tar = np.radians(gal_coords.l.deg)
			b_tar = np.radians(gal_coords.b.deg)

			t = transform[frame]
			bapex, lapex, v_apex = np.radians(t['bapex']), np.radians(t['lapex']),t['vapex']

			# NED's Velocity Conversion Calculator
			# NED converts velocities from one reference frame to another using the standard equation.

			# Compute the angular separation (theta) between your target and the Solar Apex using spherical trigonometry
			# cos theta = sin(btar)*sin(bapex) + cos(btar)*cos(bapex)*cos(ltar-lapex)
			vcor_tmp = v_apex * ( np.sin(b_tar)*np.sin(bapex) + np.cos(b_tar)*np.cos(bapex)*np.cos(l_tar-lapex) )

			# then Vcorrected = Vsys + Vapex*cos theta
		else:
			print('No CRVAL1/CRVAL2 was found in the cube Header. No reference frame change was applied.')

		return vcor_tmp




	
