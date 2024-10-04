#! /usr/bin/python3
import numpy as np
import sys
from scipy.integrate import quad


class Angdist:

	def __init__(self, redshift, H0 = 70, Omega_m = 0.30, Omega_l = 0.70, print_res = False):
		self.H0 = H0 #71 #km/s/Mpc
		self.Omega_m = Omega_m
		self.Omega_l = Omega_l
		self.z = redshift
		self.c = 299792.458
		self.print = print_res


	# THE ASTROPHYSICAL JOURNAL SUPPLEMENT SERIES, 120:49-50, 1999 January
	# Aproximacion analitica con constante cosmologicaS
	def eta(self,a,Omega_0):
		s=((1-Omega_0)/Omega_0)**(1/3.)
		d=2*np.sqrt(s**3+1)*(1./a**4-0.1540*s/a**3+0.4340*(s/a)**2+0.19097*s**3/a+0.066941*s**4)**(-1/8.)



		Omega_0=1
		dL=(self.c/self.H0)*(1+self.z)*(eta(1,Omega_0)-eta(1/(1+self.z),Omega_0))
		d_A=dL/(1+self.z)**2

		print ("dL =",dL )
		print ("dA =",d_A) 

		return dL, d_A

	##############
	# Commoving radial distance X of a source at redshift self.z


	def X(self,a):
		t1=(self.c/self.H0)/np.sqrt(a*self.Omega_m+a**2*(1-self.Omega_m-self.Omega_l)+a**4*self.Omega_l)
		return t1


	def comv_distance(self):

		a0=1./(1+self.z)
		a1=1
		X = quad(self.X,a0,a1)
		dA=X[0]/(1+self.z)
		dL=X[0]*(1+self.z)
		scale=1*dA*1e6/206265.

		if self.print:
			print ("####")
			print ("Making the integral")
			print ("####")

			print ("Commoving_radial_distance = %0.1f"%X[0])
			print ("Angular_distance [Mpc] = %0.1f"%dA)
			print ("Luminosity_distance [Mpc] = %0.1f"%dL)
			print ("Scale [pc/arcsec] = %0.3f"%scale)
		
		return dL, scale



