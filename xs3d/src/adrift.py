"""
asymmetric_drift.py
===================
Compute the asymmetric drift correction to recover the true circular
velocity v_c(r) from the observed sub-circular rotation velocity v_rot(r).

Physical background
-------------------
Gas in a disk does not rotate at the circular velocity v_c — it rotates
slightly slower because the radial pressure gradient provides partial
support against gravity.  From the radial Jeans equation for a gaseous
disk with isotropic velocity dispersion (σ_R = σ_φ = σ_v):

	v_c²(r) = v_rot²(r) − σ_v²(r) × d ln(Σ σ_v²) / d ln(r)

Expanding the logarithmic derivative:

	v_c²(r) = v_rot²(r)
			  − σ_v²(r) × [r/Σ × dΣ/dr]	  ← surface density gradient
			  − 2 σ_v(r) × r × dσ_v/dr		← dispersion gradient

Inputs required
---------------
1. v_rot(r)   — observed rotation velocity from tilted-ring fit  [km/s]
2. sigma_v(r) — velocity dispersion profile from fit			 [km/s]
3. I0(r)   — surface density profile from M0 map			  [arb. units]
				For moderately inclined galaxies: azimuthally-averaged
				M0 ring profile.
				For edge-on: requires Abel deprojection first.

Usage
-----
	from asymmetric_drift import asymmetric_drift_correction

	v_c, v_c_err = asymmetric_drift_correction(
		r	   = ring_radii,
		v_rot   = best_vrot,
		sigma_v = best_vdisp,
		I0   = m0_profile,
		e_v_rot   = vrot_errors,	 # optional
		e_sigma_v = vdisp_errors,	# optional
	)
"""

import numpy as np
from scipy.interpolate import UnivariateSpline


def _smooth_and_deriv(r, y, smooth_factor=None, k=3):
	"""
	Fit a smoothing spline to y(r) and return (y_smooth, dy_dr).

	Parameters
	----------
	r			 : array  radii (must be strictly increasing)
	y			 : array  profile values
	smooth_factor : float or None
		Spline smoothing factor.  None = auto (scipy default).
		Larger = smoother, smaller = closer to data.
	k			 : int  spline degree (default 3 = cubic)

	Returns
	-------
	y_smooth : array  smoothed profile at r
	dy_dr	: array  derivative dy/dr at r
	"""
	spl	 = UnivariateSpline(r, y, k=k, s=smooth_factor, ext='extrapolate')
	dspl	= spl.derivative()
	return spl(r), dspl(r)


#def asymmetric_drift_correction(r, v_rot, sigma_v, I0,
#								e_v_rot=None, e_sigma_v=None,
#								smooth_I0=None, smooth_sigma=None,
#								spline_k=3):
								
def asymmetric_drift_correction(mom0, rings,pixel,
								e_v_rot=None, e_sigma_v=None,
								smooth_I0=None, smooth_sigma=None,
								spline_k=3):
																
	"""
	Compute the asymmetric drift correction and return the circular velocity.

	v_c²(r) = v_rot²(r) − σ_v² × r/Σ × dΣ/dr − 2 σ_v × r × dσ_v/dr

	Parameters
	----------
	r			: (N,) array   ring radii [arcsec or kpc]
	v_rot		: (N,) array   observed rotation velocity [km/s]
	sigma_v		: (N,) array   velocity dispersion [km/s]
	I0			: (N,) array   surface density proxy (M0 ring profile)
							   [arbitrary units — only its gradient matters]
	e_v_rot		: (N,) array or None   1-sigma error on v_rot [km/s]
	e_sigma_v   : (N,) array or None   1-sigma error on sigma_v [km/s]
	smooth_I0	: float or None  smoothing for Σ(r) spline (None=auto)
	smooth_sigma: float or None  smoothing for σ_v(r) spline (None=auto)
	spline_k	: int   spline degree for derivative estimation (default 3)

	Returns
	-------
	v_c		: (N,) array   circular velocity [km/s]
	v_c_err	: (N,) array or None   propagated 1-sigma error [km/s]
			   None if e_v_rot and e_sigma_v are both None.
	info	: dict   intermediate quantities for diagnostics:
			   'correction'  : Δv² = v_c² - v_rot²  [km²/s²]
			   'term_I0'  : σ_v² × r/Σ × dΣ/dr   (surface density term)
			   'term_sigma'  : 2 σ_v × r × dσ_v/dr  (dispersion gradient term)
			   'I0_smooth': smoothed Σ(r)
			   'sigma_smooth': smoothed σ_v(r)
			   'dI0_dr'   : dΣ/dr
			   'dsigma_dr'   : dσ_v/dr

	Notes
	-----
	The correction is always positive (v_c > v_rot) for physical disks
	where Σ and σ_v decrease outward.  If the correction is negative
	at some radii (e.g. due to noise in dΣ/dr), v_c is clamped to v_rot.

	The gradient dΣ/dr is computed by fitting a smoothing spline to Σ(r)
	and differentiating analytically — this avoids noise amplification
	from numerical finite differences.  Use smooth_I0 to control
	how much the spline smooths over radial profile noise.

	For EDGE-ON galaxies: I0 should be the Abel-deprojected surface
	density, not the raw M0 major-axis profile.  The raw profile is the
	Abel transform of Σ(r) and using it directly would give wrong dΣ/dr.
	"""


	mom0_map = np.where(mom0>0,mom0,0)

	
	attr = {
	'radius'	: ('RADIUS',	'arcsec',  'E', 'Mean ring radius'),
	'width'		: ('WIDTH',	 'arcsec',  'E', 'Radial width of ring'),
	'v_rot'		: ('VROT',	  'km/s',	'E', 'Circular rotation velocity'),
	'v_disp'	: ('VDISP',	 'km/s',	'E', 'Velocity dispersion (1-D sigma)')}
	
	r 		= np.array([ getattr(R, 'radius') for R in rings ],dtype=np.float32 )			
	v_rot	= np.array([ getattr(R, 'v_rot') for R in rings ],dtype=np.float32)
	sigma_v = np.array([ getattr(R, 'v_disp') for R in rings ],dtype=np.float32)				

	if spline_k >= len(r):
		spline_k = int(len(r) -1)
	
	# compute surface brightness profile
	r_profile, I0_profile, n_pixels = extract_m0_profile(mom0_map, rings, pixel)

	r		= np.asarray(r,	   dtype=float)
	v_rot	= np.asarray(v_rot,   dtype=float)
	sigma_v = np.asarray(sigma_v, dtype=float)
	I0	= np.asarray(I0_profile,   dtype=float)

	# Sort by radius (spline requires strictly increasing x)
	idx	  = np.argsort(r).astype(int)
	r		= r[idx]
	v_rot   = v_rot[idx]
	sigma_v  = sigma_v[idx]
	I0 = I0[idx]
	if e_v_rot  is not None: e_v_rot  = np.asarray(e_v_rot)[idx]
	if e_sigma_v is not None: e_sigma_v = np.asarray(e_sigma_v)[idx]

	# Smooth Σ(r) and σ_v(r), compute derivatives analytically
	I0_sm, dI0_dr = _smooth_and_deriv(r, I0,   smooth_I0, spline_k)
	sigma_sm, dsigma_dr = _smooth_and_deriv(r, sigma_v, smooth_sigma, spline_k)

	# Guard against near-zero Σ (avoid division by zero)
	I0_safe = np.where(np.abs(I0_sm) > 1e-30*I0_sm.max(),
						  I0_sm, 1e-30*I0_sm.max())

	# Two correction terms
	term_I0 = sigma_sm**2 * r / I0_safe * dI0_dr		# σ² r/Σ dΣ/dr
	term_sigma = 2. * sigma_sm * r * dsigma_dr			# 2σ r dσ/dr

	# Total correction Δv² = v_c² - v_rot²
	# Both terms are NEGATIVE for typical disks (Σ and σ decrease outward)
	# so v_c² > v_rot²  ✓
	correction_tmp = -(term_I0 + term_sigma)   # = v_c² - v_rot²

	correction	= np.where(v_rot**2>correction_tmp,correction_tmp,0)

	# Circular velocity
	v_c_sq = v_rot**2 + correction
	# Clamp to avoid sqrt of negative (can occur at noisy outer radii)
	v_c_sq = np.maximum(v_c_sq, v_rot**2)
	v_c	= np.sqrt(v_c_sq)

	# Error propagation (first-order)
	v_c_err = None
	if e_v_rot is not None or e_sigma_v is not None:
		ev  = e_v_rot   if e_v_rot   is not None else np.zeros_like(v_rot)
		es  = e_sigma_v if e_sigma_v is not None else np.zeros_like(sigma_v)

		# dv_c/dv_rot = v_rot / v_c
		dvc_dvrot = v_rot / np.maximum(v_c, 1e-3)

		# dv_c/dσ_v  (derivative of correction w.r.t. σ_v, holding gradients fixed)
		# correction ≈ -2σ_v × (σ_v r/Σ × dΣ/dr + r × dσ_v/dr)
		# d(correction)/dσ_v ≈ -2(σ_v r/Σ × dΣ/dr + r × dσ_v/dr) - 2σ_v r/σ_v × dσ_v/dr
		#					 ≈ -(2σ_v r/Σ × dΣ/dr + 4r × dσ_v/dr)  (approximate)
		dcorr_dsigma = -(2.*sigma_sm*r/I0_safe*dI0_dr + 4.*r*dsigma_dr)
		dvc_dsigma   = dcorr_dsigma / (2.*np.maximum(v_c, 1e-3))

		v_c_err = np.sqrt((dvc_dvrot * ev)**2 + (dvc_dsigma * es)**2)

	info = dict(
		correction	= correction,
		term_I0		= term_I0,
		term_sigma	= term_sigma,
		I0_smooth	= I0_sm,
		sigma_smooth	= sigma_sm,
		dI0_dr		= dI0_dr,
		dsigma_dr	= dsigma_dr,
	)
	return v_c, v_c_err, info


def extract_m0_profile(m0_map, rings, dx_arcsec):
	"""
	Extract the azimuthally-averaged M0 profile in elliptical annuli,
	using the per-ring PA and inc from the tilted-ring fit.

	Parameters
	----------
	m0_map	: (ny, nx) array   observed moment-0 map
	rings	: list of Ring	 fitted rings (provides PA, inc, radius, width)
	dx_arcsec : float			pixel scale [arcsec/pixel]

	Returns
	-------
	r_profile	 : (N,) array   ring radii [arcsec]
	I0_profile : (N,) array   mean M0 per ring annulus [map units]
	n_pixels	  : (N,) array   number of pixels per annulus
	"""

	ny, nx = m0_map.shape
	cx	 = np.mean([r.x_center for r in rings])
	cy	 = np.mean([r.y_center for r in rings])

	# Sky pixel coordinates relative to centre
	x_pix  = (np.arange(nx) - cx) * dx_arcsec   # arcsec
	y_pix  = (np.arange(ny) - cy) * dx_arcsec
	XX, YY = np.meshgrid(x_pix, y_pix)

	r_profile	 = np.array([ring.radius for ring in rings])
	I0_profile = np.zeros(len(rings))
	n_pixels	  = np.zeros(len(rings), dtype=int)

	for i, ring in enumerate(rings):
		pa_r  = np.radians(ring.pa)
		inc_r = np.radians(ring.inc)

		# Deproject sky coordinates to disk frame
		# Undo PA rotation
		x_rot = -XX*np.sin(pa_r) + YY*np.cos(pa_r)   # wait, inverse of sky->disk
		y_rot = -XX*np.cos(pa_r) - YY*np.sin(pa_r)
		# Undo inclination
		cos_inc = max(np.cos(inc_r), 1e-3)
		y_disk  = y_rot / cos_inc
		x_disk  = x_rot

		# Disk radius at each pixel
		r_disk = np.sqrt(x_disk**2 + y_disk**2)

		# Annular mask
		r_in  = max(ring.radius - ring.width/2., 0.)
		r_out = ring.radius + ring.width/2.
		mask  = (r_disk >= r_in) & (r_disk < r_out)

		n_pix = int(mask.sum())
		if n_pix > 0:
			I0_profile[i] = float(m0_map[mask].mean())
		n_pixels[i] = n_pix

	return r_profile, I0_profile, n_pixels
