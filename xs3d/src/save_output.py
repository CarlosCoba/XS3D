import numpy as np
from astropy.io import fits
				
# ---------------------------------------------------------------------------
# FITS table output
# ---------------------------------------------------------------------------
 
# Metadata for every Ring attribute that is physically meaningful to save.
# Each entry: (fits_name, unit, format, description)
# format follows FITS convention: 'E' = float32, 'D' = float64, 'J' = int32
# Only scalar numeric attributes are included (str/bool/dict skipped).
_RING_COLUMN_META = {
	'radius'	: ('RADIUS',	'arcsec',  'E', 'Mean ring radius'),
	'width'	 : ('WIDTH',	 'arcsec',  'E', 'Radial width of ring'),
	'v_rot'	 : ('VROT',	  'km/s',	'E', 'Circular rotation velocity'),
	'v_disp'	: ('VDISP',	 'km/s',	'E', 'Velocity dispersion (1-D sigma)'),
	'v_sys'	 : ('VSYS',	  'km/s',	'E', 'Systemic velocity'),
	'v_rad'	 : ('VRAD',	  'km/s',	'E', 'Radial (inflow/outflow) velocity'),
	'inc'	   : ('INC',	   'deg',	 'E', 'Inclination angle'),
	'pa'		: ('PA',		'deg',	 'E', 'Position angle of receding axis (N->E)'),
	'x_center'  : ('XCEN',	  'pix',	 'E', 'Kinematic centre X pixel'),
	'y_center'  : ('YCEN',	  'pix',	 'E', 'Kinematic centre Y pixel'),
	'z_scale'   : ('ZSCALE',	'arcsec',  'E', 'Vertical scale height'),
	'v_2r'	  : ('V2R',	   'km/s',	'E', 'Bisymmetric radial amplitude'),
	'v_2t'	  : ('V2T',	   'km/s',	'E', 'Bisymmetric tangential amplitude'),
	'phi_bar'   : ('PHIBAR',	'deg',	 'E', 'Bar PA in disk plane'),
	'n_clouds'  : ('NCLOUDS',   '',		'J', 'Number of clouds per ring'),
}

# Parameters that are physically irrelevant for each velocity model.
# These fields exist on the Ring dataclass but are not used by the
# model and should not be interpreted as fitted values.  save_rings_fits
# sets their columns to NaN so the table does not mislead the reader.
_IRRELEVANT_BY_MODEL = {
	'circular'	: {'v_rad', 'v_2r', 'v_2t', 'phi_bar'},
	'radial'	  : {'v_2r', 'v_2t', 'phi_bar'},
	'bisymmetric' : {'v_rad'},
	'hrm'	: {'v_rad', 'v_2r', 'v_2t', 'phi_bar'},
}

# Whether to write harmonic columns (C_Mm / S_Mm) for each model.
# For bisymmetric: Ring.__post_init__ always builds harmonics={1:(v_rot,0)}
# as a side effect, but the bisymmetric model never fits harmonic
# coefficients — the kinematic information lives in v_2r/v_2t/phi_bar.
# For harmonic: the harmonics dict IS the primary representation.
# For rotation/radial: m=1 coefficients equal v_rot/-v_rad which are
# already in their own named columns; skip to avoid redundancy.
_SAVE_HARMONICS_FOR = frozenset({'hrm'})

# Parameters whose lmfit uncertainties should be saved when available.
# These are the kinematic quantities that are actually fitted.
_FITTED_ATTRS = frozenset({
	'v_rot', 'v_disp', 'v_rad', 'v_sys',
	'inc', 'pa', 'x_center', 'y_center',
	'z_scale', 'v_2r', 'v_2t', 'phi_bar',
})
 
def save_rings_fits(name, vmode, best_rings, result, psf_lsf, extra_header=None, out = '.'):
	"""
	Save the best-fit ring parameters as a FITS binary table.
 
	Each row corresponds to one anchor ring.  Each fitted kinematic
	parameter becomes a column with a physical unit.  When a lmfit
	result object is provided, 1-sigma uncertainties are saved in
	companion error columns (e.g. VROT_ERR).
 
	Parameters
	----------
	rings : list of Ring
		Best-fit anchor rings (output of fit_rings()).
	filename : str
		Output FITS filename (e.g. 'best_model.fits').
	result : lmfit.MinimizerResult or None
		Result object from fit_rings().  Used to extract per-parameter
		1-sigma uncertainties from result.params[name].stderr.
		If None, or if a parameter was fixed/tied, error columns
		contain NaN.
	dx_arcsec : float or None
		Pixel scale [arcsec/pixel].  When provided, an additional
		RADIUS_KPC column is NOT added (that requires a distance),
		but the pixel scale is recorded in the FITS header so the
		user can convert units themselves.
	extra_header : dict or None
		Extra key-value pairs added to the primary HDU header.
		Example: {'GALAXY': 'NGC1234', 'REDSHIFT': 0.003}
 
	Output columns
	--------------
	RADIUS	[arcsec]  ring radius
	WIDTH	 [arcsec]  ring width
	VROT	  [km/s]	rotation velocity
	VROT_ERR  [km/s]	1-sigma uncertainty (NaN if not available)
	VDISP	 [km/s]	velocity dispersion
	VDISP_ERR [km/s]	1-sigma uncertainty
	INC	   [deg]	 inclination
	INC_ERR   [deg]	 1-sigma uncertainty
	PA		[deg]	 position angle
	PA_ERR	[deg]	 1-sigma uncertainty
	VSYS	  [km/s]	systemic velocity
	VRAD	  [km/s]	radial velocity
	XCEN	  [pix]	 kinematic centre X
	YCEN	  [pix]	 kinematic centre Y
	ZSCALE	[arcsec]  vertical scale height
	V2R	   [km/s]	bisymmetric radial amplitude
	V2T	   [km/s]	bisymmetric tangential amplitude
	PHIBAR	[deg]	 bar position angle
	NCLOUDS   []		number of clouds per ring
 
	Harmonic decomposition columns (present only when harmonics were fitted):
	C_M1	  [km/s]	cosine coefficient of order m=1  (= v_rot)
	S_M1	  [km/s]	sine   coefficient of order m=1  (= -v_rad)
	C_M1_ERR  [km/s]	1-sigma uncertainty
	S_M1_ERR  [km/s]	1-sigma uncertainty
	C_M2	  [km/s]	cosine coefficient of order m=2
	S_M2	  [km/s]	sine   coefficient of order m=2
	...				 one pair per harmonic order present in any ring
 
	Notes
	-----
	Tied parameters share the same value across rings.  Their error
	column reflects the anchor ring's uncertainty for all tied rows.
	Fixed parameters have NaN in their error column.
 
	Examples
	--------
	best_rings, result = fit_rings(obs_cube, guess_rings, spec, cfg)
	save_rings_fits(best_rings, 'best_model.fits', result=result,
					dx_arcsec=0.2,
					extra_header={'GALAXY': 'NGC1234',
								  'OBJECT': 'NGC1234',
								  'REDSHIFT': 0.003})
	"""


	rings = best_rings
	n = len(rings)
	dx_arcsec = psf_lsf.pix_arcs
 


	# Determine which attributes are irrelevant for this velocity model.
	# All rings should use the same model (take from the first ring).
	vel_model   = rings[0].velocity_model
	irrelevant  = _IRRELEVANT_BY_MODEL.get(vel_model, set())
 
	# ── Build data arrays ──────────────────────────────────────────────
	# For each attribute: extract values from rings and, if available,
	# errors from result.params.
 
	cols = []
 
	for attr, (fits_name, unit, fmt, desc) in _RING_COLUMN_META.items():
		# Irrelevant attributes for this velocity model → NaN column
		# (NaN only applies to float columns; int columns get -1)
		if attr in irrelevant:
			if fmt == 'E':
				values = np.full(n, np.nan, dtype=np.float32)
			else:
				values = np.full(n, -1, dtype=np.int32)
		else:
			values = np.array([getattr(r, attr) for r in rings],
							  dtype=np.float32 if fmt == 'E' else np.int32)
 
		col = fits.Column(
			name   = fits_name,
			format = fmt,
			unit   = unit if unit else None,
			array  = values,
		)
		cols.append(col)
 
		# Error column for fitted kinematic attributes
		if attr in _FITTED_ATTRS:
			errs = np.full(n, np.nan, dtype=np.float32)
 
			if result is not None:
				for i in range(n):
					pname = f"{attr}_r{i}"
					if pname in result.params:
						par = result.params[pname]
						# par.stderr is None when parameter was fixed
						# or when the minimiser did not estimate errors
						if par.stderr is not None:
							errs[i] = float(par.stderr)
						elif par.expr:
							# Tied parameter: copy anchor's stderr
							# Find the anchor by evaluating the expr name
							anchor_name = par.expr.strip()
							if anchor_name in result.params:
								anchor_par = result.params[anchor_name]
								if anchor_par.stderr is not None:
									errs[i] = float(anchor_par.stderr)
 
			err_col = fits.Column(
				name   = f"{fits_name}_ERR",
				format = 'E',
				unit   = unit if unit else None,
				array  = errs,
			)
			cols.append(err_col)
 
	# ── Harmonic decomposition columns ────────────────────────────────
	# Written only for velocity_model='harmonic'.
	# For all other models (rotation, radial, bisymmetric) harmonic
	# columns are suppressed:
	#   - rotation/radial: m=1 coefficients duplicate VROT/VRAD columns
	#   - bisymmetric: __post_init__ always builds harmonics={1:(v_rot,0)}
	#	 as a side effect, but those values were never fitted — the
	#	 bisymmetric kinematic information lives in V2R/V2T/PHIBAR
	# ── Harmonic decomposition columns ────────────────────────────────
	# Written only for velocity_model='harmonic'.
	# For all other models harmonic columns are suppressed:
	#   rotation/radial : m=1 coefficients duplicate VROT/VRAD columns
	#   bisymmetric	 : __post_init__ always builds harmonics={1:(v_rot,0)}
	#					 as a side effect, but those values were never
	#					 fitted — kinematic info lives in V2R/V2T/PHIBAR
	if vel_model in _SAVE_HARMONICS_FOR:
		# Collect all harmonic orders present in any ring
		all_orders = set()
		for r in rings:
			if r.harmonics:
				all_orders.update(r.harmonics.keys())
		all_orders = sorted(all_orders)   # deterministic column order
 
		for m in all_orders:
			# Cosine coefficient c_m
			c_vals = np.array(
				[r.harmonics[m][0] if (r.harmonics and m in r.harmonics)
				 else 0.0 for r in rings],
				dtype=np.float32)
			# Sine coefficient s_m
			s_vals = np.array(
				[r.harmonics[m][1] if (r.harmonics and m in r.harmonics)
				 else 0.0 for r in rings],
				dtype=np.float32)
 
			c_name = f"C_M{m}"
			s_name = f"S_M{m}"
 
			cols.append(fits.Column(
				name=c_name, format='E', unit='km/s', array=c_vals))
			cols.append(fits.Column(
				name=s_name, format='E', unit='km/s', array=s_vals))
 
			# Error columns from lmfit result
			for coeff, cname in [('c', c_name), ('s', s_name)]:
				errs = np.full(n, np.nan, dtype=np.float32)
				if result is not None:
					for i in range(n):
						pname = f"{coeff}_m{m}_r{i}"
						if pname in result.params:
							par = result.params[pname]
							if par.stderr is not None:
								errs[i] = float(par.stderr)
							elif par.expr:
								anchor_name = par.expr.strip()
								if anchor_name in result.params:
									ap = result.params[anchor_name]
									if ap.stderr is not None:
										errs[i] = float(ap.stderr)
				cols.append(fits.Column(
					name=f"{cname}_ERR", format='E', unit='km/s', array=errs))
 
	# ── Build FITS HDU list ────────────────────────────────────────────
	# HDU 0: empty primary with header metadata
	# HDU 1: binary table with ring parameters
 
	primary_hdr = fits.Header()
	primary_hdr['COMMENT'] = 'XS3D Tilted-ring kinematic model - best-fit parameters'
	primary_hdr['NRINGS']  = (n, 'Number of anchor rings')
 
	if dx_arcsec is not None:
		primary_hdr['CDELT_AS'] = (dx_arcsec, 'Pixel scale [arcsec/pixel]')
 
	if result is not None:
		primary_hdr['CHI2']	= (float(result.chisqr),
								   'Final chi-squared value')
		primary_hdr['NFEV']	= (int(result.nfev),
								   'Number of function evaluations')
		primary_hdr['SUCCESS'] = (bool(result.success),
								   'Convergence flag')
		if hasattr(result, 'method'):
			primary_hdr['METHOD'] = (str(result.method),
									  'Optimisation method')
 
	# Velocity model (same for all rings — take from first ring)
	primary_hdr['VELMODEL'] = (rings[0].velocity_model,
								'Velocity model used in fitting')
	if irrelevant:
		primary_hdr['IRREL']  = (', '.join(sorted(irrelevant)),
								  'Params irrelevant for this model (NaN)')
 
	# Extra user-supplied header entries
	if extra_header:
		for key, val in extra_header.items():
			primary_hdr[key] = val
 
	primary_hdu = fits.PrimaryHDU(header=primary_hdr)
 
	# Binary table
	table_hdu = fits.BinTableHDU.from_columns(cols)
	table_hdu.name = 'RINGS'
	
	# Add column descriptions via TDESC keywords
	for i, (attr, (fits_name, unit, fmt, desc)) in \
			enumerate(_RING_COLUMN_META.items(), start=1):
		key = f'TDESC{i}'
		if key not in table_hdu.header:
			table_hdu.header[key] = desc

	hdul = fits.HDUList([primary_hdu, table_hdu])
	hdul.writeto(f"{out}/models/{name}.{vmode}.table.fits",overwrite=True)	

	return None
