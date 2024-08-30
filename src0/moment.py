import io
import sys
import warnings
import itertools
import numpy as np
from opy import deepcopy as copy
from s.path import basename, isfile
from cipy.interpolate import interp1d
from cipy.ndimage import median_filter
from stropy.io.fits.verify import VerifyWarning



from onstants import __c__, __sigma_to_FWHM__, __indices__, _INDICES_POS
from tats import hyperbolic_fit_par, std_m, pdl_stats, _STATS_POS


def momana_spec_wave(gas_flux__w, wave, vel, sigma, crval, cdelt,
					 n_MC=10, egas_flux__w=None, flux_ssp__w=None, gas_flux__mdl=None):
	"""
	It will proceed the moment analysis of emission line centered at `wave`.
	The parameters `crval` and `cdelt` are required in order to generate
	wavelength interval to proceed the analysis.

	::

		NW = len(gas_flux__w)
		wavelengths = crval + cdelt*[0, 1, ..., NW],


	*TODO*: create wavelength intervals withoud crval and cdelt, i.e., set intervals using `wave` and a defined interval in Angstroms.

	Parameters
	----------
	gas_flux__w : array like
		Gas spectrum.

	wave : float
		Emission line central wavelength to be analyzed.

	vel : float
		Ha velocity

	sigma : float
		Ha sigma.

	crval : float
		The wavelength attributed to the flux at position 0.

	cdelt : float
		The step which the wavelengths varies.

	n_MC : int, optional
		Number of Monte-Carlo iterations. Defaults to 10.

	egas_flux__w : array like or None
		Error in `gas_flux__w` used to randomize the gas flux value at each MC
		iteration. If None the gas flux value is not randomized. Defaults to None.

	flux_ssp__w : array like or None
		Stellar population spectrum used to the EW computation. If None, the EW
		will be 0. Defaults to None.

	Returns
	-------
	array like :
		The result of the moment analysis of emission line centered in `wave`.
		::

			[
				integrated flux [10^-16 erg/s/cm^2],
				velocity [km/s],
				dispersion [km/s],
				equivalent width [Angstroms],
				sigma(integrated flux),
				sigma(velocity),
				sigma(dispersion),
				sigma(equivalent width),
			]
	"""
	gas_flux__w_copy=np.copy(gas_flux__w)
	flux_ssp__w = np.zeros_like(gas_flux__w) if flux_ssp__w is None else flux_ssp__w
	egas_flux__w = np.zeros_like(gas_flux__w) if egas_flux__w is None else egas_flux__w
	gas_flux__w = gas_flux__w if gas_flux__mdl is None else gas_flux__mdl

	nw = gas_flux__w.size
	a_I0 = np.zeros(n_MC)
	a_I1 = np.zeros(n_MC)
	a_I2 = np.zeros(n_MC)
	a_I2_new = np.zeros(n_MC)
	a_vel_I1 = np.zeros(n_MC)
	s_I0 = 0
	s_I1 = 0
	s_I2 = 0
	s_vel_I1 = 0
	for i_mc in np.arange(0, n_MC):
		I0 = 0
		I1 = 0
		I2 = 0
		I2_new = 0
		vel_I1 = 0
		f = 1 + vel/__c__
		start_w = wave*f - 1.5*__sigma_to_FWHM__*sigma
		end_w = wave*f + 1.5*__sigma_to_FWHM__*sigma
		start_i = int((start_w - crval) / cdelt)
		end_i = int((end_w - crval) / cdelt)
		d_w = (end_w - start_w) / 4
		if start_i < 0:
			start_i = 0
		if start_i >= nw:
			start_i = nw - 1
		if end_i < 0:
			end_i = 0
		if end_i >= nw:
			end_i = nw - 1
		if i_mc == 0:
			rnd_a = np.zeros(end_i - start_i + 1)
		else:
			rnd_a = 0.5 - np.random.uniform(size=(end_i - start_i + 1))
		n_I0 = 0
		s_WE = 0
		sum_val = 0
		sum_val_abs = 0
		for iz in np.arange(start_i, end_i):
			val = gas_flux__w_copy[iz]
			e_val = egas_flux__w[iz]
			val = val + rnd_a[n_I0]*e_val
			val = val*np.abs(sigma)*np.sqrt(2*np.pi)
			val_I1=gas_flux__w[iz]
			val_I1 = val_I1*np.abs(sigma)*np.sqrt(2*np.pi)
			w = crval + iz*cdelt
			WE = np.exp(-0.5*((w - wave*f)/sigma)**2)
			I0 = I0 + WE*val*np.exp(0.5*((w - wave*f)/sigma)**2)
			I1 = I1 + np.abs(val_I1)*w
			s_WE = s_WE + WE
			sum_val_abs = sum_val_abs + np.abs(val)
			sum_val = sum_val + val
			n_I0 += 1
		if n_I0 != 0:
			s_WE /= n_I0
			I0 = I0/n_I0/s_WE
			if sum_val_abs != 0:
				I1 = I1/sum_val_abs
				if (I1 != 0) & (I0 != 0):
					vel_I1 = (I1/wave - 1)*__c__
				s_WE = 0
				for iz in np.arange(start_i, end_i):
					val = gas_flux__w[iz]
					val = val*np.abs(sigma)*np.sqrt(2*np.pi)
					f1 = (1 + vel_I1/__c__)
					if (I0 > 0) & (val > 0) & (I0 > val):
						S_now = (w - wave*f1)/(np.sqrt(2*(np.log(I0) - np.log(val))))
						if S_now > 0:
							WE = val/I0
							I2 = I2 + S_now*WE
							s_WE = s_WE + WE
							n_I0 += 1
				if s_WE > 0:
					I2 = I2/s_WE/__sigma_to_FWHM__
				else:
					I2 = sigma
				n_I0 = 0
				I2_new = 0
				sum_val_abs = 0
				for iz in np.arange(start_i, end_i):
					w = crval + iz*cdelt
					val = gas_flux__w[iz]
					val = val + rnd_a[n_I0]*e_val*0.5
					WE1 = np.exp(-0.5*((w - wave*f1)/sigma)**2)
					if (I0 > 0) & (val > 0):
						I2_new += np.abs(val)*(w - (wave * f1))**2*WE1
						sum_val_abs += np.abs(val)*WE1
					n_I0 += 1
				if sum_val_abs > 0:
					I2_new = I2_new/sum_val_abs
				else:
					I2_new = sigma
		a_I0[i_mc] = I0
		a_I1[i_mc] = I1
		a_I2[i_mc] = I2
		a_I2_new[i_mc] = I2_new
		a_vel_I1[i_mc] = vel_I1
	if n_MC > 1:
		I0 = np.mean(a_I0)
		I1 = np.mean(a_I1)
		I2 = np.sqrt(2)*__sigma_to_FWHM__*np.sqrt(np.abs(np.median(a_I2_new)))
		vel_I1 = np.mean(a_vel_I1)
		s_I0 = __sigma_to_FWHM__*np.std(a_I0, ddof=1)
		s_I1 = __sigma_to_FWHM__*np.std(a_I1, ddof=1)
		s_I2 = np.sqrt(std_m(a_I2_new))
		s_vel_I1 = np.std(a_vel_I1, ddof=1)
	else:
		I0 = a_I0[0]
		I1 = a_I1[0]
		I2 = a_I2[0]
		vel_I1 = a_vel_I1[0]
		s_I0 = 0
		s_I1 = 0
		s_I2 = 0
		s_vel_I1 = 0;
	s_I0 = np.abs(s_I0)
	s_I1 = np.abs(s_I1)
	s_I2 = np.abs(s_I2)
	s_vel_I1 = np.abs(s_vel_I1)
	#
	# Continuum error
	#
	start_i_0 = int((start_w -60 - crval) / cdelt)
	end_i_0 = int((start_w -30 - crval) / cdelt)
	start_i_1 = int((end_w + 30 - crval)/ cdelt)
	end_i_1 = int((end_w + 60 - crval) / cdelt)

	if start_i_0 < 0:
		start_i_0 = 0
	if start_i_1 < 0:
		start_i_1 = 0
	if end_i_0 >= nw:
		end_i_0 = nw - 4
	if end_i_1 >= nw:
		end_i_1 = nw - 1
	cont_0 = gas_flux__w[start_i_0:end_i_0]
	cont_1 = gas_flux__w[start_i_1:end_i_1]
	mean_0 = np.mean(cont_0)
	mean_1 = np.mean(cont_1)
	std_0 = np.std(cont_0, ddof=1)
	std_1 = np.std(cont_1, ddof=1)
	val_cont = 0.5*(mean_0 + mean_1)
	e_val_cont = np.sqrt((std_0**2 + std_1**2)/2)
	s_I0 = np.sqrt(s_I0**2 + (e_val_cont*I2)**2)
	s_I2 = np.sqrt(s_I2**2 + (e_val_cont/I2) **2 + (0.1*I2)**2)
	s_vel_I1 = np.sqrt(s_vel_I1**2 + (e_val_cont*(s_I2/5500)*__c__)**2 + ((0.02*I2/wave)*__c__)**2)
	#
	# EW
	#
	start_i_0 = int((start_w - 60 - crval)/cdelt)
	end_i_0 = int((start_w - 30 - crval)/cdelt)
	start_i_1 = int((end_w + 30 - crval)/cdelt)
	end_i_1 = int((end_w + 60 - crval)/cdelt)
	if start_i_0 < 0:
		start_i_0 = 0
	if start_i_1 < 0:
		start_i_1 = 0
	if end_i_0 > nw - 1:
		end_i_0 = nw - 1
	if end_i_1 > nw - 1:
		end_i_1 = nw - 1
	cont_0 = flux_ssp__w[start_i_0:end_i_0]
	cont_1 = flux_ssp__w[start_i_1:end_i_1]
	mean_0 = np.mean(cont_0)
	mean_1 = np.mean(cont_1)
	std_0 = np.std(cont_0, ddof=1)
	std_1 = np.std(cont_1, ddof=1)
	val_cont = 0.5*(mean_0 + mean_1)
	e_val_cont = np.sqrt((std_0**2 + std_1**2)/2)
	if val_cont != 0:
		EW = -1*I0/np.abs(val_cont)
		e_EW = np.abs(s_I0)/np.abs(val_cont) + (I0*np.abs(e_val_cont))/(val_cont**2)
		if np.abs(EW) < e_EW:
			EW = 0
		if EW > 0:
			EW = 0
	else:
		EW = np.nan
		e_EW = np.nan
	return I0, vel_I1, I2, EW, s_I0, s_vel_I1, s_I2, e_EW
	
	
	
	
	
def vel_eline(flux, wave, nsearch, imin, wave_ref, set_first_peak=True):
	y_min = 1e12
	y_max = -1e12
	flux[flux==0]=np.nan # clc
	mask_fin = np.isfinite(flux)
	mask_y_max = flux[mask_fin] > y_max
	if mask_y_max.sum() > 0:
		y_max = np.nanmax(flux[mask_fin][mask_y_max])
	mask_y_min = flux[mask_fin] < y_min
	if mask_y_min.sum() > 0:
		y_min = np.nanmin(flux[mask_fin][mask_y_min])
	# print("y_max=", y_max, "\t y_min=", y_min)
	crval = wave[0]
	cdelt = wave[1] - wave[0]
	peak_y_max = np.array([])
	peak_y_pixel = np.array([])
	i = 0
	vel = 0
	dmin = 0
	npeaks = 0
	mask_out = 0
	med = np.nanmean(flux)
	sig = np.nanstd(flux, ddof=1)
	for j in np.arange(nsearch, len(flux)-nsearch):
		peak = True
		if set_first_peak:
			if flux[j - i] > (med + 2 * sig):
				peak = True
			else:
				peak = False
		for i in np.arange(nsearch):
			if flux[j-i] < flux[j-i-1]:
				peak = False
			if flux[j+i] < flux[j+i+1]:
				peak = False
		if peak:
			if flux[j] < imin * y_max:
				peak = False
		if peak:
			if npeaks > 0:
				delta = j - peak_y_pixel[int(npeaks - 1)]
				if delta < dmin:
					peak = False
		if peak:
			peak_y_pixel = np.append(peak_y_pixel, j)
			a, b, c = j - 1, j, j + 1
			x = [a, b, c]
			y = [-flux[a], -flux[b], -flux[c]]
			peak_y, _ = hyperbolic_fit_par(x, y)
			peak_y_max = np.append(peak_y_max, peak_y)
			npeaks += 1
		if (y_max > y_min) and (y_min != 0):
			wave_peak = np.array([crval + cdelt * peak for peak in peak_y_max])
			if npeaks == 1:
				vel = (wave_peak[0]/wave_ref - 1) * __c__
				mask_out = 1
			if npeaks == 2:
				vel = (wave_peak[0] / wave_ref - 1) * __c__
				mask_out = 2
			if npeaks == 3:
				vel = (wave_peak[1] / wave_ref - 1) * __c__
				mask_out = 2
	return vel, mask_out, npeaks


