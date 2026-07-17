import numpy as np
from astropy.io import fits
from .vertical_rotation import get_table


def save_vrot_z_fits(name, vmode, rings, z_values, filename, profile=None,
					 dx_arcsec=None, extra_header=None, out = '.'):
	"""
	Save v_c(r, z) — the circular rotation velocity at each ring radius
	evaluated at a user-supplied set of heights above the midplane —
	as a FITS file, using the vertical rotation-velocity gradient
	machinery in vertical_rotation.py (see Ring.vertical_vrot_gradient).
 
	This does NOT require the rings to have been built with
	vertical_vrot_gradient=True — it is a standalone post-processing
	/ diagnostic step: given the best-fit v_rot(r) and z_scale(r) for
	each ring, it evaluates what v_c(r,z) WOULD be for the requested
	vertical density profile, at any z values you choose, including
	z values larger than what any individual ring's z_scale would
	naturally populate with clouds.
 
	Typical use: after fitting, check how much the model predicts
	rotation to lag at the height where diffuse ionised gas (DIG) or
	other extraplanar tracers are observed, for comparison with
	measurements such as den Brok et al. 2020 (MNRAS 491, 4089) or
	Levy et al. 2018.
 
	Parameters
	----------
	rings : list of Ring
		Rings to evaluate (e.g. best_rings from fit_rings()).  Each
		ring's v_rot and z_scale are used; radius must be > 0 and
		z_scale must be > 0 for a ring to be evaluated (rings that
		fail this are skipped with a printed warning and filled with
		NaN in the output, so the row count always matches len(rings)).
	z_values : array_like
		Heights above the midplane [arcsec] at which to evaluate
		v_c(r,z).  The SAME z_values are used for every ring (a single
		common grid), so the output is a clean 2D table.  Can include
		z=0 (returns exactly v_rot, since v_c(r,0)=v_rot by definition).
	filename : str
		Output FITS filename.
	profile : str or None
		Vertical profile to use ('gaussian', 'exponential', 'sech2').
		If None (default), each ring's own vertical_vrot_profile
		attribute is used (rings may differ); if rings disagree, a
		per-ring choice is honoured and PROFILE in the header is set
		to 'mixed'.  Pass an explicit profile to override every ring's
		attribute with the same choice for this evaluation.
	dx_arcsec : float or None
		Pixel scale [arcsec/pixel], recorded in the header only (no
		unit conversion is performed; z_values and radii stay in
		whatever units the rings themselves use, normally arcsec).
	extra_header : dict or None
		Extra key-value pairs added to the primary HDU header.
 
	Output (FITS structure)
	------------------------
	HDU 0 (primary):  header only.  NRINGS, NZ, PROFILE, and any
					   extra_header entries.
	HDU 1 'RINGS':	one row per ring —
					   RADIUS [arcsec], VROT [km/s], ZSCALE [arcsec],
					   PROFILE (string, per-ring vertical profile used),
					   VRZ ['NZ'E] — vector column, v_c(r,z) for this
					   ring evaluated at every entry of Z_VALUES, same
					   order.  NaN row (all VRZ entries NaN) for any
					   ring skipped because z_scale<=0 or radius<=0.
	HDU 2 'Z_VALUES': single column Z [arcsec], length NZ — the common
					   height grid shared by every row of VRZ in HDU 1.
 
	Examples
	--------
	import numpy as np
	z_grid = np.linspace(0, 5, 11)   # 0 to 5 arcsec above the midplane
	save_vrot_z_fits(best_rings, z_grid, 'vrot_vs_z.fits',
					 profile='sech2',
					 extra_header={'GALAXY': 'NGC1234'})
 
	# Reading it back:
	from astropy.io import fits
	with fits.open('vrot_vs_z.fits') as hdul:
		radius = hdul['RINGS'].data['RADIUS']
		vrz	= hdul['RINGS'].data['VRZ']		# shape (n_rings, n_z)
		z_vals = hdul['Z_VALUES'].data['Z']		# shape (n_z,)
		# v_c at ring 0, height z_vals[3]:
		v = vrz[0, 3]
	"""
	if not rings[0].vz_gradient:
		return None
			
	z_cale0		= rings[0].z_scale
	zscale		= z_cale0
	nhz 		= 6
	nhz_arcs	=  nhz*zscale
	z_values	= np.linspace(0,nhz_arcs, nhz) 
	z_values	= np.asarray(z_values, dtype=float)
	n_z	 		= len(z_values)
	n			= len(rings)
 
	radius_vals  = np.empty(n, dtype=np.float32)
	vrot_vals	= np.empty(n, dtype=np.float32)
	zscale_vals  = np.empty(n, dtype=np.float32)
	profile_vals = np.empty(n, dtype='U11')
	vrz		  = np.full((n, n_z), np.nan, dtype=np.float32)
 
	n_skipped = 0
	for i, ring in enumerate(rings):
		radius_vals[i] = ring.radius
		vrot_vals[i]   = ring.v_rot
		zscale_vals[i] = ring.z_scale
 
		ring_profile = profile if profile is not None \
					   else ring.vertical_vrot_profile
		profile_vals[i] = ring_profile
 
		if ring.radius <= 0 or ring.z_scale <= 0:
			n_skipped += 1
			continue
 
		table = get_table(ring_profile)
		alpha = ring.z_scale / ring.radius
		ratio = table.vc_ratio(np.abs(z_values) / ring.z_scale, alpha)
		vrz[i, :] = ring.v_rot * ratio
 
	if n_skipped:
		print(f"  save_vrot_z_fits: {n_skipped}/{n} ring(s) skipped "
			  f"(radius<=0 or z_scale<=0) -- filled with NaN")
 
	unique_profiles = set(profile_vals.tolist())
	profile_header  = (profile_vals[0] if len(unique_profiles) == 1
					   else 'mixed')
 
	# ── HDU 1: per-ring table with vector VRZ column ──────────────────
	cols = [
		fits.Column(name='RADIUS', format='E', unit='arcsec',
						 array=radius_vals),
		fits.Column(name='VROT',   format='E', unit='km/s',
						 array=vrot_vals),
		fits.Column(name='ZSCALE', format='E', unit='arcsec',
						 array=zscale_vals),
		fits.Column(name='PROFILE', format='11A',
						 array=profile_vals),
		fits.Column(name='VRZ', format=f'{n_z}E', unit='km/s',
						 array=vrz),
	]
	table_hdu = fits.BinTableHDU.from_columns(cols)
	table_hdu.name = 'RINGS'
	table_hdu.header['TDESC1'] = 'Ring radius'
	table_hdu.header['TDESC2'] = 'Midplane rotation velocity v_c(r,0)'
	table_hdu.header['TDESC3'] = 'Vertical scale height used for this ring'
	table_hdu.header['TDESC4'] = 'Vertical profile used for this ring'
	table_hdu.header['TDESC5'] = (
		f'v_c(r,z) at each of the {n_z} heights in the Z_VALUES HDU; '
		'NaN row = ring skipped (radius<=0 or z_scale<=0)'
	)
 
	# ── HDU 2: the shared z grid ───────────────────────────────────────
	z_col = fits.Column(name='Z', format='E', unit='arcsec',
							 array=z_values.astype(np.float32))
	z_hdu = fits.BinTableHDU.from_columns([z_col])
	z_hdu.name = 'Z_VALUES'
	z_hdu.header['TDESC1'] = 'Height above midplane shared by every VRZ row'
 
	# ── HDU 0: primary header ──────────────────────────────────────────
	primary_hdr = fits.Header()
	primary_hdr['COMMENT'] = 'v_c(r,z): vertical rotation-velocity gradient'
	primary_hdr['NRINGS']  = (n, 'Number of rings evaluated')
	primary_hdr['NZ']	  = (n_z, 'Number of z values evaluated')
	primary_hdr['PROFILE'] = (str(profile_header),
							   "Vertical profile ('mixed' if rings differ)")
	if dx_arcsec is not None:
		primary_hdr['CDELT_AS'] = (dx_arcsec, 'Pixel scale [arcsec/pixel]')
	if extra_header:
		for key, val in extra_header.items():
			primary_hdr[key] = val
	primary_hdu = fits.PrimaryHDU(header=primary_hdr)


	filename = f'{name}.{vmode}.vz.fits' 
	hdul = fits.HDUList([primary_hdu, table_hdu, z_hdu])
	hdul.writeto(f'{out}/models/{filename}', overwrite=True)
	#print(f"Saved: {filename}  ({n} rings x {n_z} z-values, "f"profile={profile_header})")
	return filename
