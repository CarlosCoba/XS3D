import numpy as np

from .cloud_tilted_rings import  Ring
from .cloud_fit_engine import (
    build_params, fit_rings, make_weight_map,
    residual_cube, rotation_curve,
    set_bounds, _print_params_summary,
)

class Set_params:
	
	def __init__(self, vmode, psf_lsf, rings, rwidth, vary, hdr, guess_common, m_hrm=0):
		self.R = rings
		R_nc = self.R['R_nc']		
		self.nrings = len(self.R['R_pos'])
		bmaj = psf_lsf.bmaj
		# width is the radial extent of a single anchor ring in the galaxy disk plane.
		self.width = rwidth
		# When ring_spacing < PSF_FWHM :
		if rwidth < bmaj:
			# This means each fine ring is thinner than one beam:	
			self.width 			= bmaj # (never wider than the spacing)
			psf_lsf.radial_step = bmaj # (fine grid matches the spacing)
		
		self.common = guess_common
		self.nx 	= hdr.nx
		self.ny 	= hdr.ny
		self.m_hrm 	= m_hrm
		self.vmode 	= vmode
		self.dx 	= hdr.dx
		self.vel_axis=hdr.wave_kms
		
		fit_disp = psf_lsf.vary_disp
		replacements = {0: 'fixed', 1: 'tied', 2: 'free'}
		params = ('pa','inc','x_center','y_center','v_sys','phi_b')
		self.vary = {params[k]:replacements.get(x, x) for k,x in enumerate(vary)}
		#self.vary_nc = ['free' if R_nc[k] else 0 for k in range(self.nrings) ]
		self.vary_nc = [replacements.get(x, x) for k,x in enumerate(R_nc)  ]
				
		self.vary_disp =  replacements[fit_disp]
		nclouds = guess_common['n_clouds']
		self.inc = guess_common['inc']
		
		nclouds_per_pix = nclouds/self.dx**2 
		self.nclouds = [ nclouds_per_pix for k in range(self.nrings)]		
			
	def circular(self,vels):
		disp_tab,vrot_tab,vrad_tab,vtan_tab = vels
		guess_rings = [
					Ring(radius= self.R['R_pos'][k], nclouds=self.nclouds[k], width=self.width, v_rot=vrot_tab[k], v_disp=disp_tab[k],\
					v_rad=vrad_tab[k], v_2r=vrad_tab[k], v_2t=vtan_tab[k], **self.common) for k in range(self.nrings)
				]				
		return guess_rings

				
	def harm(self,vels):
		m_hrm = self.m_hrm
		[disp_tab,c_tab,s_tab]=vels
		
		disp = np.array(disp_tab)[np.newaxis, :]
		s_k = np.array(s_tab)
		c_k = np.array(c_tab)
		vrot=c_k[0,:]
		vdisp=disp[0,:]		

		guess_rings=[]
		for k in range(self.nrings):
				ring_k = Ring( radius= self.R['R_pos'][k], width=self.width, v_rot=vrot[k],v_disp=vdisp[k], harmonics={(m+1): (c_k[m,:][k], s_k[m,:][k]) for m in range(m_hrm)}, **self.common )
				guess_rings.append(ring_k)
		return guess_rings


	def prms(self,vmode):
		spec = {
				'pa'      : self.vary['pa'],
				'inc'     : self.vary['inc'],
				'v_sys'   : self.vary['v_sys'],
				'x_center': self.vary['x_center'],
				'y_center': self.vary['y_center'],
				'v_disp'  : self.vary_disp,												
			}
		if vmode!='hrm':
			spec['v_rot'] 	= 'free'					
		if vmode=='radial':
			spec['v_rad']	= self.vary_nc				
		if vmode=='bisymmetric':
			spec['v_2r']	= self.vary_nc				
			spec['v_2t'] 	= self.vary_nc							
			spec['phi_bar']	= self.vary['phi_b']

		if 'hrm' in vmode:
			for m in range(self.m_hrm):
				k = int(m+1)
				spec[f'c_m{k}']  =  self.vary_nc
				spec[f's_m{k}']  =  self.vary_nc
			spec['c_m1'] = 'free'
													
		return spec				
				
	def lmfit_bounds(self, params):
		# Build lmfit Parameters and set bounds
		#params = build_params(guess_rings, spec)
		n	= self.nrings			

		set_bounds(params, 'v_disp', n,   0,  500.0)
		
		if self.vary['pa'] == 'free':
			set_bounds(params, 'pa', n,   -360,  360.0)

		min_inc = 8 if self.inc < 80 else 70
		if self.vary['inc'] == 'free':
			set_bounds(params, 'inc', n,   min_inc,  90.0)
									
		params['pa_r0'].min  =   -360.0
		params['pa_r0'].max  = 360.0
		params['inc_r0'].min =  min_inc
		params['inc_r0'].max =  180.0

		params['v_sys_r0'].min =  np.min(self.vel_axis)
		params['v_sys_r0'].max =  np.max(self.vel_axis)
		
				
		params['x_center_r0'].min =  0
		params['x_center_r0'].max =  self.nx			
		params['y_center_r0'].min =  0
		params['y_center_r0'].max =  self.ny
			
		if 'hrm' not in self.vmode:
			set_bounds(params, 'v_rot',  n,  0.0, 500.0)
									
		if self.vmode == 'radial':
			set_bounds(params, 'v_rad',  n,  -500.0, 500.0)		

		if self.vmode == 'bisymmetric':
			set_bounds(params, 'v_2t',  n,  -500, 500.0)					
			set_bounds(params, 'v_2r',  n,  -500, 500.0)
			set_bounds(params, 'phi_bar', n,   0,  180.0)									
			#params['phi_bar_r0'].min = 0.0
			#params['phi_bar_r0'].max = 180.0   # only 0-180 needed due to 2*phi_bar symmetry

		if 'hrm' in self.vmode:
			for m in range(self.m_hrm):
				hn=int(m+1)
				set_bounds(params, f'c_m{hn}',   self.nrings,  -500, 500)
				set_bounds(params, f's_m{hn}',   self.nrings,  -500, 500)
