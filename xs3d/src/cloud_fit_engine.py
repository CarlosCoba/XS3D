"""
=========================================================
Compares a model cube (built with cloud_tilted_rings.py's TiltedRingModel) against
an observed datacube and recovers the best-fit kinematic parameters by
minimising a weighted chi-squared cost function.

This version uses the lmfit library, which replaces the hand-rolled
_pack / _unpack / _build_bounds machinery with a clean Parameters
object.  Every parameter has value, min, max, vary, and expr attributes
that can be changed at any time with a single assignment — no custom
packing logic required.

Cost function
-------------
	chi2 = sum_{x,y,v}  W(x,y) * [F_obs(x,y,v) - F_mod_norm(x,y,v)]^2

	F_mod_norm is the model cube rescaled so that its moment-0 matches
	the observed moment-0 at every spatial pixel.  This decouples
	kinematics from surface brightness.

	W(x,y) = |cos(phi)|^alpha  where phi is the azimuthal angle in the
	deprojected disk plane.  alpha=2 (default) gives strong preference
	to pixels near the major axis where rotation is most visible.

lmfit parameter naming convention
----------------------------------
Standard Ring attributes:
	<attr>_r<i>	   e.g.  inc_r0, pa_r0, v_disp_r2

Harmonic coefficients  c_m<M>  and  s_m<M>:
	c_m<M>_r<i>	   e.g.  c_m1_r0, c_m1_r1, s_m2_r0

Tied parameters use lmfit's expr mechanism:
	params['pa_r1'].expr = 'pa_r0'   # r1 always equals r0

Optimisation methods (all via lmfit.minimize)
---------------------------------------------
	method='nelder'		  Nelder-Mead simplex  (default, gradient-free)
	method='leastsq'		 Levenberg-Marquardt  (requires residual vector)
	method='differential_evolution'  Global search
	method='emcee'		   Bayesian MCMC posterior sampling

Usage example
-------------
	See __main__ block at the bottom of this file.

Install
-------
	pip install lmfit
"""

import copy
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as plt
from scipy import stats
from .utils import circmean
from .conv_fftw2 import save_fftw_wisdom,load_fftw_wisdom

try:
	import lmfit
	from lmfit import Parameters, minimize as lm_minimize, fit_report
except ImportError:
	raise ImportError(
		"lmfit is required.  Install with:  pip install lmfit")

from .cloud_tilted_rings import TiltedRingModel, CubeConfig, Ring, RingBuilder


# ---------------------------------------------------------------------------
# Spatial weight map
# ---------------------------------------------------------------------------

def make_weight_map(mom0, psf_cfg, rings, alpha=(2.0,1), r_max_px=None, n_sigma_z=3, edge_on_threshold=90.0, vertical_weight=False):
	"""
    Boolean mask of the projected galaxy volume out to r_max_px.

    Accounts for three contributions to the sky-plane footprint:

    1. Radial extent: disk-plane radius ≤ r_max_px
       → ellipse with semi-major=r_max, semi-minor=r_max×cos(inc)

    2. Vertical extent: disk height z ∈ [−n_sigma_z×z_scale, +n_sigma_z×z_scale]
       projects onto the sky minor axis as:
           z_sky = z_disk × sin(inc)
       → adds n_sigma_z × z_scale_arcsec × sin(inc) / dx_arcsec pixels
         of half-width in the minor-axis direction beyond the ellipse

    3. Edge-on (inc ≥ 85°): rectangular strip, combining both effects:
           hw = max(r_max_px × cos(inc),
                    n_sigma_z × z_scale_arcsec × sin(inc) / dx,
                    1.0)
	Parameters
	----------
	psf_cfg : CubeConfig
	rings	: list of Ring
		Used for mean inc, pa, and kinematic centre.
	alpha	: float
		Cosine weighting exponent.
		0 = uniform, 1 = cosine, 2 = cosine-squared (default).
	r_max	: float or None
		Radius of the outermost ring [same units as ring radii / pixels].
		If None, no spatial mask is applied (all pixels included).
		In fit_rings() this is set automatically to the radius of the
		outermost input ring so that extrapolated regions are excluded
		from the minimisation.

	Returns
	-------
	W : np.ndarray, shape (ny, nx),  values in [0, 1]
	"""
	(rweight, zweight) 	= alpha
	vertical_weight		= zweight

	psf_pix 			= psf_cfg.fwhm_psf_pix
	z_scale_pix 		= psf_cfg.zscale_pix
	# Observed (PSF-convolved) vertical extent
	zscale_pix_obs 		= np.sqrt(z_scale_pix**2 + psf_pix**2)
	# this capture 95% of the vertical emission
	#n_sigma_z = 1

	inc_arr = [r.inc for r in rings]
	pa_arr	= [r.pa%360  for r in rings]
	inc_deg = float(circmean(inc_arr))
	inc = np.radians(circmean(inc_arr))
	pa  = np.radians(circmean(pa_arr))
	cx  = np.mean([r.x_center for r in rings])
	cy  = np.mean([r.y_center for r in rings])

	xs = np.arange(psf_cfg.nx) - cx
	ys = np.arange(psf_cfg.ny) - cy
	XX, YY = np.meshgrid(xs, ys)

 	# If the disk is warped do not appy radial weight
	#warp = len(set(inc_arr)) == 1 # This means that all values in inc_arr are the same.
	#if warp : rweight = 0

 # ── Step 2 (inverse): undo PA rotation ───────────────────────────
    # _disk_to_sky Step 2 rotates disk coords to sky coords by PA:
    #   x_sky = -x_inc × sin(PA) - y_inc × cos(PA)
    #   y_sky = +x_inc × cos(PA) - y_inc × sin(PA)
    # Here we go the other way: given a sky pixel (XX, YY) we rotate
    # back into the kinematic frame aligned with the disk major axis.
    # The inverse rotation (by -PA) gives:
    #   x_rot = -x_sky × sin(PA) + y_sky × cos(PA)  → major axis direction
    #   y_rot = -x_sky × cos(PA) - y_sky × sin(PA)  → minor axis direction
	x_rot =  -XX * np.sin(pa) + YY * np.cos(pa)   # along major axis
	y_rot =  -XX * np.cos(pa) - YY * np.sin(pa)   # along minor axis (sky)

    # Vertical projection: a point at height z_disk above the midplane
    # projects onto the sky minor axis as z_sky = z_disk × sin(inc)
    # (from _disk_to_sky Step 1: y_sky includes -z_disk × sin(inc) term)
    # n_sigma_z Gaussian scale heights span this many pixels on the sky:
	z_hw_px = n_sigma_z * zscale_pix_obs * np.sin(inc)  if z_scale_pix > 0 else 0.0

	sigma_z_sky  = zscale_pix_obs * np.sin(inc)  # pixels
	if z_scale_pix>0:
		W_z = np.exp(-y_rot**2 / (2.0 * sigma_z_sky**2))
	else:
		W_z = np.ones_like(y_rot)

	 # ----------------------------------------------------------------
	# CHANGE 1: branch on inclination regime
	# ----------------------------------------------------------------

	if inc_deg >= edge_on_threshold:
        # Edge-on: rectangular strip.
        # Minor-axis half-width hw combines two physical contributions:
        #   r_max_px × cos(inc) : radial extent projected onto minor axis
        #   z_hw_px             : vertical (z) extent projected onto minor axis
        # At inc=90°: cos(inc)=0 so only z_hw_px contributes (plus minimum=psf/2)
		if r_max_px is not None and z_scale_pix>=0:
			strip_halfwidth = max(r_max_px * np.cos(inc), z_hw_px, psf_pix/2)
			inside_strip = (
				(np.abs(x_rot) <= r_max_px) & (np.abs(y_rot) <= strip_halfwidth)
			)
			W = inside_strip.astype(float)
		else:
			# No r_max: uniform weight everywhere
			W = np.ones_like(x_rot)
	else:
        # ── Step 1 (inverse): undo inclination ───────────────────────
        # _disk_to_sky Step 1 compresses y by cos(inc):
        #   y_inc = y_disk × cos(inc)
        # Inverse: y_disk = y_rot / cos(inc)
        # This recovers the disk-plane y coordinate for each sky pixel,
        # allowing us to compute the true disk-plane radius.

		cos_inc = max(np.cos(inc), 1e-3)
		y_disk  = y_rot / cos_inc					  # deproject minor axis
		phi	 = np.arctan2(y_disk, x_rot)

        # Disk-plane radius at each sky pixel (midplane z=0)
		r_disk = np.sqrt(x_rot ** 2 + y_disk ** 2)

		# Component 1: kinematic major-axis weighting
		W = np.abs(np.cos(phi)) ** rweight
		# Component 2: spatial mask — EXACT projected cylinder footprint.
		# A sky pixel (x_rot, y_rot) receives emission from the model
		# cylinder  {r_disk ≤ r_max, |z| ≤ z_max}  if and only if
		# there exists a point (x_disk, y_disk, z_disk) inside the
		# cylinder that projects to it.  Since x_rot = x_disk exactly,
		# we need |x_rot| ≤ r_max and then:
		#
		# y_rot is achievable ↔ ∃ y_disk ∈ [-R, R], z_disk ∈ [-z_max, z_max]
		# with R = sqrt(r_max^2 - x_rot^2), such that
		# y_rot = -y_disk * cos(inc) + z_disk * sin(inc)
		#
		# The range of achievable y_rot for fixed x_rot is:
		#	|y_rot| ≤ R * cos(inc) + z_max * sin(inc)
		#			= sqrt(r_max^2 - x_rot^2) * cos(inc) + z_max * sin(inc)
		#
		# This boundary is the MINKOWSKI SUM of the midplane ellipse
		# (y_boundary = ±R*cos(inc)) and a uniform vertical expansion
		# of ±z_max*sin(inc) in the y_rot direction.
		# The resulting shape is a STADIUM (ellipse with flat sides),
		# NOT a rectangle and NOT a plain ellipse.

		if r_max_px is not None:
			#Radial condition: inside disk out to r_max
			in_disk = r_disk <= r_max_px

			if z_scale_pix >= 0:
				#Exact projected cylinder boundary:
				#	|y_rot| ≤ sqrt(r_max^2 - x_rot^) * cos(inc) + z_max_px * sin(inc)
				R_sq        = np.maximum(r_max_px**2 - x_rot**2, 0.0)
				z_max_px	= z_scale_pix * n_sigma_z
				y_boundary  = (np.sqrt(R_sq) * cos_inc + z_max_px * np.sin(inc))
				spatial_mask=((np.abs(x_rot) <= r_max_px) &(np.abs(y_rot) <= y_boundary))

				if zweight:
					return W_z * W * (spatial_mask).astype(float)
				else:
					return W * (spatial_mask).astype(float)
			else:
				return W * (in_disk).astype(float)

	return W


# ---------------------------------------------------------------------------
# Moment-0 normalisation
# ---------------------------------------------------------------------------

def normalize_to_moment0(mod_cube, obs_cube):
	"""
	Rescale each model spectrum so its integrated flux matches the
	observed moment-0 at every spatial pixel (x, y):

		F_mod_norm(x,y,v) = F_mod(x,y,v) * M0_obs(x,y) / M0_mod(x,y)

	Pixels where M0_mod = 0 are left at zero (masked out of the fit).

	Parameters
	----------
	mod_cube : np.ndarray, shape (nv, ny, nx)
	obs_cube : np.ndarray, shape (nv, ny, nx)

	Returns
	-------
	mod_norm : np.ndarray, shape (nv, ny, nx)
	"""
	m0_obs = np.sum(obs_cube, axis=0)
	m0_mod = np.sum(mod_cube, axis=0)
	with np.errstate(invalid="ignore", divide="ignore"):
		scale = np.where(m0_mod > 0, m0_obs / m0_mod, 0.0)
	return mod_cube * scale[np.newaxis, :, :]


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def chi2(obs_cube, mod_cube, weight_map):
	"""
	Weighted chi-squared between observed and model cubes.

		chi2 = sum_{x,y,v}  W(x,y) * [F_obs_n - F_mod_norm_n]^2

	Both cubes are normalised by the observed peak before differencing,
	making the value dimensionless and scale-independent.

	Parameters
	----------
	obs_cube   : np.ndarray, shape (nv, ny, nx)
	mod_cube   : np.ndarray, shape (nv, ny, nx)   raw model output
	weight_map : np.ndarray, shape (ny, nx)

	Returns
	-------
	float
	"""
	obs_peak = obs_cube.max()
	if obs_peak == 0:
		return 1e30

	mod_norm  = normalize_to_moment0(mod_cube, obs_cube)
	obs_n	 = obs_cube / obs_peak
	mod_n	 = mod_norm / obs_peak
	residuals = obs_n - mod_n
	W		 = weight_map[np.newaxis, :, :]
	return float(np.sum(W * residuals ** 2))


# ---------------------------------------------------------------------------
# Harmonic key helpers  (identical to previous version)
# ---------------------------------------------------------------------------

def _is_harmonic_key(p):
	"""True for keys of the form 'c_mN' or 's_mN' (N a positive integer)."""
	if len(p) < 4:
		return False
	if p[0] not in ('c', 's') or p[1:3] != '_m':
		return False
	return p[3:].isdigit()


def _parse_harmonic_key(p):
	"""Return (coeff, m) for a harmonic key, e.g. 'c_m2' -> ('c', 2)."""
	return p[0], int(p[3:])


def _get_ring_attr(ring, attr):
	"""Get attr from ring, supporting harmonic keys."""
	if _is_harmonic_key(attr):
		coeff, m = _parse_harmonic_key(attr)
		c_m, s_m = ring.harmonics.get(m, (0.0, 0.0))
		return c_m if coeff == 'c' else s_m
	return getattr(ring, attr)


def _set_ring_attr(ring, attr, val):
	"""Set attr on ring, supporting harmonic keys and keeping m=1 in sync."""
	if _is_harmonic_key(attr):
		coeff, m = _parse_harmonic_key(attr)
		c_m, s_m = ring.harmonics.get(m, (0.0, 0.0))
		if coeff == 'c':
			ring.harmonics[m] = (float(val), s_m)
		else:
			ring.harmonics[m] = (c_m, float(val))
		if m == 1:
			ring.v_rot =  ring.harmonics[1][0]
			ring.v_rad = -ring.harmonics[1][1]
	else:
		setattr(ring, attr, float(val))


# ---------------------------------------------------------------------------
# lmfit parameter name convention
# ---------------------------------------------------------------------------
#
#   Standard attr:   "<attr>_r<i>"		e.g.  "inc_r0", "pa_r2"
#   Harmonic attr:   "<harm>_r<i>"		e.g.  "c_m1_r0", "s_m2_r1"
#
# The ring index i runs 0 .. n_rings-1.
#
# Tied parameters are implemented via lmfit's expr mechanism:
#   params['pa_r1'].expr = 'pa_r0'
# This means the optimiser only has one free variable (pa_r0) and
# pa_r1, pa_r2, ... are computed algebraically — no pack/unpack needed.

def _lm_name(attr, ring_idx):
	"""Build the lmfit parameter name for a given attribute and ring index."""
	return f"{attr}_r{ring_idx}"


# ---------------------------------------------------------------------------
# Build lmfit Parameters from rings + param_spec
# ---------------------------------------------------------------------------

def build_params(rings, param_spec):
	"""
	Construct an lmfit Parameters object from the ring list and a
	parameter specification dict.

	Parameter specification (param_spec)
	--------------------------------------
	A dict where each key is a Ring attribute name (or harmonic key
	'c_mN' / 's_mN') and the value controls how that attribute is
	optimised.  Four forms are supported:

	'free'
		Each ring has an independent free parameter.
		Bounds default to (-inf, +inf); supply via bounds_dict if needed.

	'tied'
		All rings share one value.  Ring 0 is the free parameter;
		rings 1..N-1 get  expr='<attr>_r0'  so they track it exactly.

	'fixed'
		All rings hold their initial value.  vary=False is set on every
		ring.  The parameter still appears in the Parameters object so
		you can inspect or change it later.

	list  (per-ring control, one entry per ring)
		Each entry is one of:
		  'free'	independent free parameter for this ring
		  'tied'	shares the value of the first 'tied' entry for this attr
		  'fixed'   holds initial value, vary=False
		  <number>  fixed at exactly this numeric value, vary=False

	Parameters
	----------
	rings	  : list of Ring
	param_spec : dict   see above

	Returns
	-------
	params : lmfit.Parameters

	Examples
	--------
	# v_rot free per ring, single tied pa, v_rad free inner / fixed outer,
	# second harmonic free inner / fixed at 0 outer, v_sys fixed
	spec = {
		'c_m1'	: 'free',
		's_m1'	: ['free', 'free', 0.0, 0.0],
		'c_m2'	: ['free', 'free', 0.0, 0.0],
		's_m2'	: ['free', 'free', 0.0, 0.0],
		'v_disp'  : 'free',
		'pa'	  : 'tied',
		'inc'	 : 'tied',
		'v_sys'   : 'fixed',
		'x_center': 'fixed',
		'y_center': 'fixed',
	}
	params = build_params(rings, spec)

	# Toggle pa to free independently per ring:
	for i in range(len(rings)):
		params[f'pa_r{i}'].expr = ''	  # clear the tie
		params[f'pa_r{i}'].vary = True

	# Fix inc at a known value:
	for i in range(len(rings)):
		params[f'inc_r{i}'].vary  = False
		params[f'inc_r{i}'].value = 60.0
	"""
	n	  = len(rings)
	params = Parameters()

	for attr, mode in param_spec.items():
		# Expand uniform string modes to a per-ring list
		if isinstance(mode, str):
			entries = [mode] * n
		elif isinstance(mode, list):
			if len(mode) != n:
				raise ValueError(
					f"param_spec['{attr}'] has {len(mode)} entries "
					f"but there are {n} rings.")
			entries = mode
		else:
			raise TypeError(
				f"param_spec values must be str or list, got {type(mode)}")

		# First pass: find the 'tied' anchor ring index (if any)
		tied_anchor = None
		for i, entry in enumerate(entries):
			if entry == 'tied':
				tied_anchor = i
				break

		# Second pass: add one lmfit parameter per ring
		for i, entry in enumerate(entries):
			name = _lm_name(attr, i)
			val  = _get_ring_attr(rings[i], attr)

			if entry == 'free':
				params.add(name, value=val, vary=True)

			elif entry == 'tied':
				if i == tied_anchor:
					# The anchor is a normal free parameter
					params.add(name, value=val, vary=True)
				else:
					# All other tied rings point to the anchor via expr
					anchor_name = _lm_name(attr, tied_anchor)
					params.add(name, value=val, vary=False,
							   expr=anchor_name)

			elif entry == 'fixed':
				params.add(name, value=val, vary=False)

			elif isinstance(entry, (int, float)):
				# Fixed at the given numeric value
				params.add(name, value=float(entry), vary=False)

			else:
				raise ValueError(
					f"Unknown entry '{entry}' in param_spec['{attr}']")

	return params


# ---------------------------------------------------------------------------
# Apply lmfit Parameters back to Ring list
# ---------------------------------------------------------------------------

def params_to_rings(params, rings):
	"""
	Write the current parameter values back into a deep copy of rings.

	Parameters
	----------
	params : lmfit.Parameters   (e.g. from the objective callback)
	rings  : list of Ring	   template (not modified)

	Returns
	-------
	new_rings : list of Ring   deep-copied and updated
	"""
	new_rings = copy.deepcopy(rings)
	for name, par in params.items():
		# name format: "<attr>_r<i>"
		# Split on last '_r' to handle attrs like 'x_center', 'c_m1'
		idx = name.rfind('_r')
		if idx == -1:
			continue
		attr	 = name[:idx]
		ring_idx = int(name[idx + 2:])
		_set_ring_attr(new_rings[ring_idx], attr, par.value)
	return new_rings


# ---------------------------------------------------------------------------
# Objective function for lmfit
# ---------------------------------------------------------------------------

def _make_objective(obs_cube, obs_emap, moms_obs, rings, cube_cfg, psf_lsf, cube_oper, weight_alpha, seed,
					verbose_counter, model, verbose):
	"""
	Return a closure that lmfit.minimize can call.

	lmfit passes the current Parameters object; we convert it to rings,
	build the model cube, and return either a scalar (for Nelder-Mead /
	differential_evolution) or a flattened residual array (for leastsq /
	emcee).
	"""
	[mom0_obs,mom1_obs,mom2_obs]=moms_obs
	bmaj = psf_lsf.bmaj
	chi2_scale=np.var(obs_cube)
	def objective(params):
		new_rings = params_to_rings(params, rings)

		# Recompute weight map from current geometry — cheap (<1 ms)
		r_max_cur	= max(r.radius + bmaj for r in new_rings)
		r_max_cur_pix 	= r_max_cur/cube_cfg.pix_arcs
		W_cur			= make_weight_map(mom0_obs,psf_lsf,new_rings,alpha=weight_alpha,r_max_px=r_max_cur_pix)
		W_cur_sum		= np.sum(W_cur)
		[nz,ny,nx]		= obs_cube.shape

		# Reset the RNG to the fixed seed so that repeated calls with
		# the same parameters produce the same cube (deterministic chi2).
		# We reset only the RNG — the ConvolutionEngine is NOT recreated.
		model.rng  = np.random.default_rng(seed)
		model._rb  = RingBuilder(cube_cfg, model.rng)
		mod_cube   = model.build(new_rings, verbose=False)

		obs_peak = obs_cube.max()
		if obs_peak == 0:
			cost = np.ones(obs_cube.size) * 1e15
			return cost

		#eflux = obs_peak
		eflux = obs_emap

		mom0_mod_tmp	= cube_oper.obs_mommaps(mod_cube,mom_out=(0))
		mom0_msk		= (mom0_obs > 0) & (mom0_mod_tmp > 0)
		mod_cube_norm	= mod_cube*np.divide(mom0_obs,mom0_mod_tmp,where=mom0_msk,out=np.zeros_like(mom0_mod_tmp))
		mom0_mod, mom1_mod = cube_oper.obs_mommaps(mod_cube_norm,mom_out=(0,1))

		obs_n	= obs_cube / eflux
		mod_n 	= mod_cube_norm / eflux
		W 		= W_cur[np.newaxis, :, :]
		W		= W / W_cur_sum
		msk 	= (mom0_obs > 0) & (mom0_mod_tmp > 0) & (W_cur > 0)
		Ndata	= np.sum(msk)*nz

		# Residuals from moment 1 map
		lmbda		= 1
		dv_norm		= cube_cfg.dv
		res_moms	= ( np.sqrt(lmbda) * np.sqrt(W_cur/W_cur_sum) * (mom1_obs - mom1_mod) )[msk]
		res_moms	/= dv_norm

		# Residuals from 3D fitting
		residuals	= (obs_n - mod_n) * msk
		wresiduals	= np.sqrt(W) * residuals  # weighted residuals

		verbose_counter[0] += 1
		if verbose_counter[0] % 20 == 0 and verbose:
			cost = float(np.sum(residuals ** 2))
			free_vals = {k: f"{v.value:.2f}"
						 for k, v in params.items() if v.vary}
			Nvary = len(free_vals)
			dof	  = Ndata-Nvary
			chisqr= cost/dof
			print(f"  Iter {verbose_counter[0]:5d}  "
				  f"chi2r={chisqr:.6f}  "
				  + "  ".join(f"{k}={v}" for k, v in
							   list(free_vals.items())[:6]))

		# Return flat residual vector — lmfit sums squares internally.
		# Works for both scalar methods (nelder, differential_evolution)
		# and vector methods (leastsq, emcee).
		#return np.concatenate([residuals.ravel(), res_moms.ravel()])
		return np.concatenate([wresiduals.ravel()])

	return objective


# ---------------------------------------------------------------------------
# Public fitting functions
# ---------------------------------------------------------------------------

def fit_rings(obs_cube, obs_emap, moms_obs, rings, param_spec, lmfit_prms, cube_cfg, psf_lsf, cube_oper,
			  weight_alpha=(2.0,1),
			  method='nelder',
			  seed=42,
			  verbose=True,
			  fit_kws=None):
	"""
	Fit the tilted-ring model to an observed datacube using lmfit.

	This is the single entry point for all optimisation methods.  The
	method parameter selects the algorithm; everything else is handled
	by lmfit transparently.

	Parameters
	----------
	obs_cube	 : np.ndarray, shape (nv, ny, nx)
		Observed datacube.
	rings		: list of Ring
		Initial guess rings.  Fixed parameters are taken from here.
	param_spec   : dict
		Parameter specification — see build_params() for full details.
		Keys are Ring attribute names or harmonic keys ('c_mN', 's_mN').
		Values are 'free' | 'tied' | 'fixed' | per-ring list.
	cube_cfg	 : CubeConfig
	weight_alpha : float
		Exponent for the major-axis cosine weight map (default 2).
	method	   : str
		lmfit minimisation method.  Recommended choices:
		  'nelder'				 Nelder-Mead simplex (default)
								   Gradient-free, robust, no bounds needed.
		  'differential_evolution' Global stochastic search.
								   Requires min/max set on all free params.
		  'leastsq'				Levenberg-Marquardt.
								   Fast near the minimum; uses Jacobian.
		  'emcee'				  Bayesian MCMC (slow; for posteriors).
	seed		 : int
		Random seed passed to TiltedRingModel.
	verbose	  : bool
		Print chi2 and free-parameter values every 20 iterations.
	fit_kws	  : dict or None
		Extra keyword arguments forwarded to lmfit.minimize.
		Examples:
		  Nelder-Mead:  {'options': {'xatol': 1e-4, 'fatol': 1e-4}}
		  Diff. evol.:  {'max_nfev': 10000, 'popsize': 10}
		  emcee:		{'steps': 1000, 'nwalkers': 50, 'burn': 200}

	Returns
	-------
	best_rings : list of Ring
		Rings updated with best-fit parameter values.
	result	 : lmfit.MinimizerResult
		Full lmfit result object.  Key attributes:
		  result.params		best-fit Parameters
		  result.success	   convergence flag
		  result.nfev		  number of function evaluations
		  result.chisqr		final chi-squared
		  result.message	   solver message
		For leastsq / emcee, also:
		  result.covar		 covariance matrix
		  result.errorbars	 True if uncertainties were estimated

	Examples
	--------
	# --- Nelder-Mead (local, gradient-free) ---
	spec = {
		'c_m1'	: 'free',
		's_m1'	: ['free', 'free', 0.0, 0.0],
		'v_disp'  : 'free',
		'pa'	  : 'tied',
		'inc'	 : 'tied',
		'v_sys'   : 'fixed',
		'x_center': 'fixed',
		'y_center': 'fixed',
	}
	best, result = fit_rings(obs_cube, rings, spec, cfg)
	print(fit_report(result))

	# --- Differential evolution (global search, needs bounds) ---
	params = build_params(rings, spec)
	for i in range(len(rings)):
		params[f'c_m1_r{i}'].min = 50.0
		params[f'c_m1_r{i}'].max = 300.0
	params['pa_r0'].min = 0.0
	params['pa_r0'].max = 360.0
	best, result = fit_rings(obs_cube, rings, spec, cfg,
							 method='differential_evolution')

	# --- Levenberg-Marquardt (fast local, gives uncertainties) ---
	best, result = fit_rings(obs_cube, rings, spec, cfg, method='leastsq')
	print(fit_report(result))   # prints uncertainties on each parameter

	# --- Toggle a parameter after the fact ---
	result.params['pa_r0'].vary = False
	result.params['pa_r0'].value = 120.0
	best2, result2 = fit_rings(obs_cube, rings, spec, cfg,
								method='nelder')
	"""
	# r_max: outermost input ring radius — pixels beyond this are
	# extrapolated by _interpolate_rings and must not enter the chi2.
	r_max   = max(r.radius + 0.5 * r.width for r in rings)

	params  = build_params(rings, param_spec)
	bounds = lmfit_prms.lmfit_bounds(params)
	counter = [0]

	# before measuring the planner check if there is any available
	load_fftw_wisdom(cube_cfg)
	
	model   = TiltedRingModel(cube_cfg, psf_lsf, seed=seed,planner_effort='FFTW_MEASURE')

	# save the planner to reuse it in the future
	save_fftw_wisdom(cube_cfg)
	
	obj	 = _make_objective(obs_cube, obs_emap, moms_obs, rings, cube_cfg, psf_lsf, cube_oper, weight_alpha, seed, counter, model, verbose)
	
	if verbose:
		_print_params_summary(params, rings)

	kws = fit_kws or {}
	result = lm_minimize(obj, params, method=method, **kws)

	best_rings = params_to_rings(result.params, rings)

	if verbose:
		print(f"\nConverged : {result.success}")
		print(f"chi2	  : {result.chisqr:.6f}")
		print(f"nfev	  : {result.nfev}")
		if hasattr(result, 'message'):
			print(f"Message   : {result.message}")
		_print_results(result.params, rings)

	return best_rings, result


# ---------------------------------------------------------------------------
# Convenience: modify params after build
# ---------------------------------------------------------------------------

def set_bounds(params, attr, n_rings, lo, hi):
	"""
	Set min/max bounds on all free parameters for a given attribute.

	Required for differential_evolution.  Can also be used to constrain
	Nelder-Mead (lmfit will clip values to bounds if set).

	Parameters
	----------
	params  : lmfit.Parameters  (from build_params)
	attr	: str			   e.g. 'c_m1', 'pa', 'inc'
	n_rings : int
	lo, hi  : float			 lower and upper bound

	Example
	-------
	params = build_params(rings, spec)
	set_bounds(params, 'c_m1', len(rings), 50, 300)
	set_bounds(params, 'pa',   len(rings),  0, 360)
	"""
	for i in range(n_rings):
		name = _lm_name(attr, i)
		if name in params:
			params[name].min = lo
			params[name].max = hi


def tie(params, attr, n_rings, anchor=0):
	"""
	Tie all rings for a given attribute to a single anchor ring.

	Useful for post-hoc adjustments — e.g. you fitted pa freely per
	ring but now want to re-run with a single global pa.

	Parameters
	----------
	params  : lmfit.Parameters
	attr	: str
	n_rings : int
	anchor  : int   ring index that becomes the free parameter (default 0)
	"""
	anchor_name = _lm_name(attr, anchor)
	params[anchor_name].vary = True
	params[anchor_name].expr = ''
	for i in range(n_rings):
		if i == anchor:
			continue
		name = _lm_name(attr, i)
		if name in params:
			params[name].vary = False
			params[name].expr = anchor_name


def free_all(params, attr, n_rings):
	"""
	Make all rings free and independent for a given attribute.

	Parameters
	----------
	params  : lmfit.Parameters
	attr	: str
	n_rings : int
	"""
	for i in range(n_rings):
		name = _lm_name(attr, i)
		if name in params:
			params[name].expr = ''
			params[name].vary = True


def fix_all(params, attr, n_rings, value=None):
	"""
	Fix all rings for a given attribute.

	Parameters
	----------
	params  : lmfit.Parameters
	attr	: str
	n_rings : int
	value   : float or None
		If given, set all rings to this value before fixing.
	"""
	for i in range(n_rings):
		name = _lm_name(attr, i)
		if name in params:
			params[name].expr = ''
			params[name].vary = False
			if value is not None:
				params[name].value = float(value)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_params_summary(params, rings):
	"""Print a summary table of all parameters before fitting."""
	n = len(rings)
	print(f"Fitting  {n} rings  |  "
		  f"free params: {sum(1 for p in params.values() if p.vary)}")
	print(f"  {'Name':<18}  {'Value':>10}  {'Vary':>5}  {'Expr'}")
	print("  " + "-" * 55)
	for name, par in params.items():
		expr_str = par.expr if par.expr else ""
		print(f"  {name:<18}  {par.value:10.4f}  "
			  f"{'yes' if par.vary else 'no':>5}  {expr_str}")
	print()


def _print_results(params, rings):
	"""Print best-fit values for all free (or tied) parameters."""
	print(f"\n  {'Name':<18}  {'Best-fit':>10}  {'Stderr':>10}")
	print("  " + "-" * 42)
	for name, par in params.items():
		if par.vary or par.expr:
			stderr = f"{par.stderr:.4f}" if par.stderr is not None else "n/a"
			print(f"  {name:<18}  {par.value:10.4f}  {stderr:>10}")


def rotation_curve(rings):
	"""
	Extract the rotation curve (c_m1) from a list of rings.

	Returns
	-------
	radii : np.ndarray
	v_rot : np.ndarray   (= c_m1 values)
	"""
	radii = np.array([r.radius for r in rings])
	v_rot = np.array([r.harmonics.get(1, (r.v_rot, 0.0))[0]
					  for r in rings])
	return radii, v_rot


def residual_cube(obs_cube, mod_cube):
	"""
	Normalised residual cube consistent with the chi2 cost function:

		residual(x,y,v) = [F_obs - F_mod_norm] / peak(obs)

	Parameters
	----------
	obs_cube, mod_cube : np.ndarray, shape (nv, ny, nx)

	Returns
	-------
	np.ndarray, shape (nv, ny, nx)
	"""
	peak = obs_cube.max()
	if peak == 0:
		return obs_cube - mod_cube
	return (obs_cube - normalize_to_moment0(mod_cube, obs_cube)) / peak


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	print("=" * 60)
	print("  cloud_fit_engine.py  —  lmfit fitting demo")
	print("=" * 60)

	# ------------------------------------------------------------------
	# 1.  Build a "true" model (mock observation)
	# ------------------------------------------------------------------
	cfg = CubeConfig(
		nx=128, ny=128, nv=80,
		dx=1.0, dy=1.0,
		dv=5.0, v_min=-200.0,
		beam_fwhm=3.0, chan_width=5.0,
		radial_step=1.0,
	)

	true_params = dict(
		v_sys=0.0, inc=55.0, pa=120.0,
		x_center=64.0, y_center=64.0,
		z_scale=2.0, z_profile="gaussian",
		n_clouds=6000, n_subclouds=5,
	)
	# m=1: rotation + radial outflow in inner 2 rings
	# m=2: elliptical streaming in inner 2 rings
	true_rings = [
		Ring(radius=10, width=5, v_rot=100, v_disp=10,
			 harmonics={1: (100.0, -30.0), 2: (15.0, 8.0)}, **true_params),
		Ring(radius=25, width=5, v_rot=150, v_disp= 9,
			 harmonics={1: (150.0, -30.0), 2: (10.0, 5.0)}, **true_params),
		Ring(radius=40, width=5, v_rot=170, v_disp= 8,
			 harmonics={1: (170.0,   0.0), 2: ( 0.0, 0.0)}, **true_params),
		Ring(radius=55, width=5, v_rot=160, v_disp= 9,
			 harmonics={1: (160.0,   0.0), 2: ( 0.0, 0.0)}, **true_params),
	]

	print("\nBuilding mock observed cube...")
	true_model = TiltedRingModel(cfg, seed=0)
	obs_cube   = true_model.build(true_rings, verbose=False)
	rng		= np.random.default_rng(1)
	obs_cube  += rng.normal(0, 0.02 * obs_cube.max(), obs_cube.shape)
	obs_cube   = np.clip(obs_cube, 0, None)

	# ------------------------------------------------------------------
	# 2.  Initial guess rings
	# ------------------------------------------------------------------
	guess_params = dict(
		v_sys=0.0, inc=60.0, pa=110.0,
		x_center=64.0, y_center=64.0,
		z_scale=2.0, z_profile="gaussian",
		n_clouds=6000, n_subclouds=5,
	)
	guess_rings = [
		Ring(radius=10, width=5, v_rot= 90, v_disp=10,
			 harmonics={1: ( 90.0, 0.0), 2: (0.0, 0.0)}, **guess_params),
		Ring(radius=25, width=5, v_rot=140, v_disp= 9,
			 harmonics={1: (140.0, 0.0), 2: (0.0, 0.0)}, **guess_params),
		Ring(radius=40, width=5, v_rot=160, v_disp= 8,
			 harmonics={1: (160.0, 0.0), 2: (0.0, 0.0)}, **guess_params),
		Ring(radius=55, width=5, v_rot=150, v_disp= 9,
			 harmonics={1: (150.0, 0.0), 2: (0.0, 0.0)}, **guess_params),
	]

	# ------------------------------------------------------------------
	# 3.  Parameter specification
	# ------------------------------------------------------------------
	spec = {
		'c_m1'	: 'free',					# v_rot — free per ring
		's_m1'	: ['free', 'free', 0.0, 0.0],# v_rad — free inner, 0 outer
		'c_m2'	: ['free', 'free', 0.0, 0.0],# m=2 cosine — free inner only
		's_m2'	: ['free', 'free', 0.0, 0.0],# m=2 sine   — free inner only
		'v_disp'  : 'free',					# dispersion — free per ring
		'pa'	  : 'tied',					# one shared PA
		'inc'	 : 'tied',					# one shared inclination
		'v_sys'   : 'fixed',
		'x_center': 'fixed',
		'y_center': 'fixed',
	}

	# Build params and optionally set bounds (required for diff. evolution)
	params = build_params(guess_rings, spec)
	n	  = len(guess_rings)
	set_bounds(params, 'c_m1',   n,  50, 300)
	set_bounds(params, 's_m1',   n, -80,  80)
	set_bounds(params, 'c_m2',   n, -50,  50)
	set_bounds(params, 's_m2',   n, -50,  50)
	set_bounds(params, 'v_disp', n,   3,  40)
	params['pa_r0'].min  =   0.0
	params['pa_r0'].max  = 360.0
	params['inc_r0'].min =  10.0
	params['inc_r0'].max =  85.0

	# ------------------------------------------------------------------
	# 4.  Fit — Nelder-Mead (fast local)
	# ------------------------------------------------------------------
	print("\nRunning Nelder-Mead fit...")
	best_rings, result = fit_rings(
		obs_cube, guess_rings, spec, cfg,
		weight_alpha=(2.0,1), method='nelder', seed=42, verbose=True,
		fit_kws={'options': {'xatol': 1e-3, 'fatol': 1e-3,
							 'maxiter': 5000}},
	)
	print("\n" + fit_report(result))

	# ------------------------------------------------------------------
	# 5.  Demonstrate lmfit parameter toggling
	# ------------------------------------------------------------------
	print("\n--- Toggling example ---")
	print("Tying inc to a fixed value of 55° and re-fitting pa only...")
	fix_all(result.params, 'inc', n, value=55.0)
	free_all(result.params, 'pa',  n)		   # pa free per ring now
	# (would call lm_minimize again here with result.params as start)

	# ------------------------------------------------------------------
	# 6.  Rotation curve
	# ------------------------------------------------------------------
	r_true, v_true = rotation_curve(true_rings)
	r_best, v_best = rotation_curve(best_rings)

	# ------------------------------------------------------------------
	# 7.  Residual cube and weight map
	# ------------------------------------------------------------------
	best_model = TiltedRingModel(cfg, seed=42)
	mod_cube   = best_model.build(best_rings, verbose=False)
	res_cube   = residual_cube(obs_cube, mod_cube)
	W		  = make_weight_map(cfg, best_rings, alpha=2.0)

	# ------------------------------------------------------------------
	# 8.  Plots
	# ------------------------------------------------------------------
	def mom0(c): return np.sum(c, axis=0)
	def mom1(c, thr=0):
		vel  = cfg.v_min + np.arange(cfg.nv) * cfg.dv
		mask = c > thr
		m0   = np.sum(c * mask, axis=0)
		m1   = np.sum(c * mask * vel[:, None, None], axis=0)
		with np.errstate(invalid="ignore", divide="ignore"):
			return np.where(m0 > 0, m1 / m0, np.nan)

	thr = 0.05 * obs_cube.max()

	fig, axes = plt.subplots(2, 3, figsize=(15, 9))
	fig.suptitle("cloud_fit_engine.py (lmfit) — Fitting demo", fontsize=13)

	axes[0, 0].imshow(mom0(obs_cube), origin="lower", cmap="inferno")
	axes[0, 0].set_title("Obs — Moment 0")

	axes[0, 1].imshow(mom0(mod_cube), origin="lower", cmap="inferno")
	axes[0, 1].set_title("Model — Moment 0")

	axes[0, 2].imshow(mom0(res_cube), origin="lower", cmap="RdBu_r",
					  vmin=-0.3, vmax=0.3)
	axes[0, 2].set_title("Residual (obs - mod) / peak")

	axes[1, 0].imshow(mom1(obs_cube, thr), origin="lower",
					  cmap="RdBu_r", vmin=-200, vmax=200)
	axes[1, 0].set_title("Obs — Moment 1")

	axes[1, 1].imshow(mom1(mod_cube, thr), origin="lower",
					  cmap="RdBu_r", vmin=-200, vmax=200)
	axes[1, 1].set_title("Model — Moment 1")

	im_w = axes[1, 2].imshow(W, origin="lower", cmap="hot", vmin=0, vmax=1)
	axes[1, 2].set_title("Weight map  W(x,y) = |cos phi|2")
	plt.colorbar(im_w, ax=axes[1, 2])

	plt.tight_layout()
	plt.savefig("cloud_fit_maps.png", dpi=150, bbox_inches="tight")
	print("Saved: cloud_fit_maps.png")

	fig2, ax = plt.subplots(figsize=(7, 4))
	ax.plot(r_true, v_true, "o-", label="True",   color="steelblue")
	ax.plot(r_best, v_best, "s--", label="Fitted", color="tomato")
	ax.set_xlabel("Radius [pixels]")
	ax.set_ylabel("v_rot  [km/s]")
	ax.set_title("Rotation Curve — True vs Fitted")
	ax.legend(); plt.tight_layout()
	plt.savefig("cloud_fit_rotcurve.png", dpi=150, bbox_inches="tight")
	print("Saved: cloud_fit_rotcurve.png")
	plt.show()
