import numpy as np
import matplotlib.pylab as plt
import numpy.polynomial.legendre as leg
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from .pixel_params import eps_2_inc
from scipy.optimize import curve_fit
from .axes_params import axes_ambient as axs

# Parameterization from Weijmans 2008
# https://ui.adsabs.harvard.edu/abs/2008MNRAS.383.1343W/abstract
def sigma_func(x, s0, s1, r1):
	return s0 + s1*np.exp(-x/r1)


def sb_func1(x, S0, S1, r0, r1):
	return S0*np.exp(-x/r0)+ S1*np.exp(-x/r1)


# Parameterization from Nikki 2025
# https://ui.adsabs.harvard.edu/abs/2025A%26A...697A..87G/abstract
def sb_func2(x, S0, r0, c1, c2, c3):
	return S0*(np.exp(-x/r0)) * (1+c1*x+c2*x**2+c3*x**3)



# Parameterization from Bureau 2007
# https://ui.adsabs.harvard.edu/abs/2002AJ....123.1316B/abstract
# The Astronomical Journal 123(3):1316

def sigma2_func(x, s0, s1, r1):
	return s0 + s1*np.exp(-x**2/r1**2)


def sb_sigma2func(x, S0, r0, beta):
	return S0*(r0+1) / (r0+np.exp(beta*x)) 

# Parameterization from Se-Heon 2015
#https://ui.adsabs.harvard.edu/abs/2015AJ....149..180O/abstract

def sb_sigma2func_seheon(x, I0, r0, alpha):
	return I0*(r0+1)/(r0+np.exp(alpha*x))




class AD_cor:
	def __init__(self, galaxy,vmode, R, Sigma, Vrot, mom01d, eps, plot = True, hz = None, regularize = False, thermal = None, SB = 'Buereau', out = './'):
	

		# inclination
		b_a = 1 - eps

		self.Disp = Sigma 
		self.Vt = Vrot
		# correct SB by inclination
		sb_cor = mom01d * b_a			
		self.SB = mom01d * b_a
		
		# regularize dispersion and SB profiles to decrease noise ?
		# use Legendre polynomial
		degree = 3 
		if regularize :
			# here are the coefficients
			coeffs_vrot = leg.legfit(R,Vrot,degree)
			coeffs_sigma = leg.legfit(R,Sigma,degree)	
			coeffs_sb = leg.legfit(R,sb_cor,degree)		

			# regularized functions
			self.Vrot_L = leg.legval(R,coeffs_vrot)
			self.Disp_L = leg.legval(R,coeffs_sigma)	
			self.SB_L = leg.legval(R,coeffs_sb)
		else:
			self.Vrot_L = self.Vt
			self.Disp_L = self.Disp	
			self.SB_L = self.SB 
	
	
		self.Disp2_L = self.Disp_L*self.Disp_L
		self.Disp2_SB_L = self.Disp2_L * self.SB_L 
		
		# Least square fitting on Dispersion**2 
		try:
			popt, pcov = curve_fit(sigma2_func, R, self.Disp2_L)
			[s0, s1, r1] = popt
			self.Disp2_sm = sigma2_func(R, *popt)
		except(ValueError, RuntimeError):
			self.Disp2_sm = self.Disp_L*self.Disp_L	
			pass



		# Least square fitting on SB

		try:
			if SB == 'Nikki':		 						
				popt, pcov = curve_fit(sb_func2, R, self.SB_L)
				self.SB_sm = sb_func2(R, *popt)
				
			if SB == 'Weijmans':
				popt, pcov = curve_fit(sb_func1, R, self.SB_L)
				self.SB_sm = sb_func1(R, *popt)
				
			if SB == 'Buereau':
				popt, pcov = curve_fit(sb_sigma2func, R, self.Disp2_SB_L)
				self.Disp_SB2_sm = sb_sigma2func(R,*popt)
				self.SB_sm = self.SB_L
		#	if SB == 'SeHeon':			
		#		popt, pcov = curve_fit(sb_sigma2func, R, self.Disp_SB2_L)
							
												
		except(ValueError, RuntimeError):
			self.SB_sm = self.SB_L
			self.Disp_SB2_sm = self.Disp_SB2_sm
			pass
				

		self.Disp2 = Sigma*Sigma
	
		
		'''
		Asymmetric drift correction
		
		Vasymm^2 = -R*(disp**2)  * d/dR (ln [SB*(disp**2)] )
				 = -R*(disp**2)  * [  d/dR (ln SB) + d/dR (ln disp**2) ]
				 
		'''
		
		


		# compute derivatives
		if SB == 'Buereau':	
			self.D_all = np.gradient(np.log( self.Disp_SB2_sm), R )
		#elif SB == 'SeHeon':
		 
		else:
			self.D1 = np.gradient(np.log(self.SB_sm), R)
			self.D2 = np.gradient(np.log(self.Disp2_sm, R))
			self.D_all = self.D1 + self.D2	
		

		
		# sum of L	
		self.AD2 = -R*self.Disp2_sm*(self.D_all)
		self.AD = np.sqrt(self.AD2) 

		if plot :
			fig, ax = plt.subplots(figsize=(6, 12), dpi = 300)
			gs = gridspec.GridSpec(4, 1)
			ax0 = plt.subplot(gs[0,0])
			ax1 = plt.subplot(gs[1,0])
			ax2 = plt.subplot(gs[2,0])
			ax3 = plt.subplot(gs[3,0])			


			#ax0_twin = ax0.twinx()
			vmin = abs(np.nanmin(sb_cor))
			vmax = abs(np.nanmax(sb_cor))
		
			norm_log = True if (vmin>0) & (np.log10(vmax/vmin)>1) else False
			if 	norm_log:
				ax0.semilogy(R, sb_cor, 'ok', )
				ax0.semilogy(R, self.SB_L, '-k', label = 'regularized')	
				ax0.semilogy(R, self.SB_sm, 'g-', label = 'fit')
				ax0.set_yscale('log') 	
			else:
				ax0.plot(R, sb_cor, 'ok', )
				ax0.plot(R, self.SB_L, '-k', label = 'regularized')	
				ax0.plot(R, self.SB_sm, 'g-', label = 'fit')
			#ax0_twin.plot(R, self.SB_sm, 'k--')



			ax1.plot(R,self.Disp, 'ok')
			
			ax2.plot(R,self.Disp2,'ok')
			ax2.plot(R,self.Disp2_sm,'-k')		
			
			
			ax3.plot(R, self.Disp2_SB_L, 'ok')
			ax3.plot(R, self.Disp_SB2_sm, '-k')		
			
			
			#ax0.set_xlabel('r (arcsec)', fontsize = 18)
			#ax1.set_xlabel('r (arcsec)', fontsize = 18)
			#ax2.set_xlabel('r (arcsec)', fontsize = 18)				
			ax3.set_xlabel('r (arcsec)', fontsize = 18)						
			
			ax0.set_ylabel('$\mathrm{\Sigma}$ (flux km/s)', fontsize = 18)
			ax1.set_ylabel('$\sigma$ (km/s)', fontsize = 18)
			ax2.set_ylabel('$\sigma^2$ (km$^2$/s$^2$)', fontsize = 18)				
			ax3.set_ylabel('$\mathrm{\Sigma} \sigma^2$ (flux km$^2$/s$^{2}$)', fontsize = 18)
			
			

			axs(ax0,rotation='horizontal', fontsize_ticklabels=18);axs(ax1,rotation='horizontal', fontsize_ticklabels=18);axs(ax2,rotation='horizontal', fontsize_ticklabels=18);axs(ax3,rotation='horizontal', fontsize_ticklabels=18)
						
			ax0.set_ylim(0.01,None)
			ax0.set_yscale('log') 						
			ax3.set_yscale('log') 									
		
		if hz is not None:
			# Self-gravitating exponential disk with constant
			# velocity dispersion
			vasymm2 = -3.36*(np.mean(self.Disp))**2*(R/re)
			
		fig.tight_layout()		
		plt.savefig("%sfigures/adrift_%s_model_%s.png"%(out,vmode,galaxy))
		plt.close()

		

	def __call__(self):	
		return self.AD
	
	


