"""
Tilted-Ring Model Builder — Cloud Approximation Method
=======================================================
Implements the cloud approximation methodology for constructing
3D kinematic models of galaxies from spectroscopic data cubes.

Key conventions
---------------
  * PA : position angle of the RECEDING major axis, measured N→E
		 (counter-clockwise from North/+y toward East/−x).
		 Standard astronomical image orientation: North=up, East=left.
		 PA =   0° → receding side points North (+y)
		 PA =  90° → receding side points East  (left, −x)
		 PA = 270° → receding side points West  (right, +x)
  * Inc: 0 = face-on, 90 = edge-on.
  * Three velocity models, selected via Ring.velocity_model:
	  'circular' : v_los = v_sys + v_rot * cos(phi) * sin(inc)
	  'radial'   : v_los = v_sys + v_rot * cos(phi) * sin(inc)
								 - v_rad * sin(phi) * sin(inc)
	  'harmonic' : v_los = v_sys + sin(inc) * Σ_m [c_m*cos(m*phi)
												   + s_m*sin(m*phi)]
	where phi = 0 is the receding major axis in the disk plane.

Smoothness strategy
-------------------
  User-supplied rings define the rotation curve at discrete radii.
  Before building, all ring parameters are interpolated onto a fine
  radial grid (spacing = 1 pixel by default) so there are no gaps
  between adjacent rings.  n_clouds is scaled proportionally to the
  ring circumference so surface brightness stays uniform.
"""

import numpy as np
import matplotlib.pylab as plt
from dataclasses import dataclass

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


# Convolution engine (pyfftw or scipy fallback) lives in convolution.py
from .conv_fftw2 import ConvolutionEngine, PYFFTW_AVAILABLE, ADVISOR_AVAILABLE
 


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Ring:
	"""
	Parameters describing a single tilted ring (annulus).

	Attributes
	----------
	radius : float
		Mean radius of the ring [arcsec or pixels — must match CubeConfig].
	width : float
		Radial width of the ring (same units as radius).
	v_rot : float
		Rotation velocity [km/s].  Stored as a convenience shorthand;
		internally mapped to c_1 in the harmonics dict via __post_init__.
	v_disp : float
		Velocity dispersion (1-D Gaussian sigma) [km/s].
	v_sys : float
		Systemic velocity of the galaxy [km/s].
	inc : float
		Inclination [degrees].  0 = face-on, 90 = edge-on.
	pa : float
		Position angle of the RECEDING major axis [degrees], N→E
		(counter-clockwise from North).
	x_center : float
		X pixel coordinate of the kinematic centre.
	y_center : float
		Y pixel coordinate of the kinematic centre.
	v_rad : float
		Radial (inflow/outflow) velocity [km/s].  Stored as a convenience
		shorthand; internally mapped to s_1 = -v_rad in the harmonics dict
		via __post_init__.  0 = pure rotation.
	z_scale : float
		Scale height of the vertical density distribution (same units
		as radius).  0 = infinitely thin disk.
	z_profile : str
		Shape of the vertical density profile:
		'gaussian' | 'sech2' | 'exponential' | 'uniform'.
	n_clouds : int
		Number of discrete clouds on this ring.  When rings are
		interpolated this is rescaled by circumference automatically.
	n_subclouds : int
		Sub-clouds per cloud.  Each gets an independent random v_disp
		kick, improving sampling of the velocity PDF.
	harmonics : dict or None
		Harmonic velocity decomposition.  Keys are integer harmonic
		orders m (1, 2, 3, ...); values are (c_m, s_m) tuples [km/s].

		The line-of-sight velocity model is:

			v_los = v_sys + sin(inc) * Σ_m [ c_m*cos(m*phi) + s_m*sin(m*phi) ]

		where phi = 0 points to the receding major axis.

		Relation to physical velocities at m = 1:
			c_1 =  v_rot   (rotation, cosine term)
			s_1 = -v_rad   (radial flow, sine term; minus sign so that
							v_rad > 0 means outflow)

		If harmonics is None on construction, __post_init__ automatically
		builds  harmonics = {1: (v_rot, -v_rad)}  so that the simple
		interface (v_rot, v_rad) remains fully backward-compatible.

		Higher-order terms (m >= 2) represent non-circular motions:
			m=2: elliptical streaming / bar perturbations
			m=3: spiral arm perturbations
			etc.

		Missing orders default to (0.0, 0.0) during interpolation.
	velocity_model : str
		Selects the line-of-sight velocity formula used by _los_velocity.
		Three models are available:

		'circular'
			Pure circular rotation only.  Uses v_rot and v_sys.

				v_los = v_sys + v_rot * cos(phi) * sin(inc)

			The simplest physical model.  Appropriate when the gas
			follows circular orbits and non-circular motions are
			negligible.  v_rad is ignored.

		'radial'
			Circular rotation plus a radial (inflow/outflow) component.
			Uses v_rot, v_rad, and v_sys.

				v_los = v_sys + v_rot * cos(phi) * sin(inc)
							  - v_rad * sin(phi) * sin(inc)

			v_rad > 0 means outflow.  Radial flows are invisible on the
			major axis (sin(phi)=0 there) and appear as a kinematic
			asymmetry between the two halves of the minor axis.

		'harmonic'
			General harmonic decomposition.  Uses the harmonics dict.
			Subsumes the two models above (at m=1 with s_1=0 this
			reduces to 'circular'; with s_1=-v_rad it reduces to
			'radial').

				v_los = v_sys + sin(inc) * Σ_m [c_m*cos(m*phi)
											   + s_m*sin(m*phi)]

			Supports any combination of harmonic orders m=1,2,3,...

		Default is 'harmonic' so that existing code using the harmonics
		dict is unaffected.  Use 'circular' or 'radial' for simpler,
		faster models that do not require a harmonics dict.
	"""
	radius:		 	float
	width:		  	float
	v_rot:		  	float
	v_disp:		 	float
	v_sys:		  	float
	inc:			float
	pa:			 	float
	x_center:	   	float
	y_center:	   	float
	v_rad:		  	float  = 0.0
	z_scale:		float  = 0.0
	z_profile:	  	str	= "gaussian"
	n_clouds:	   	int	= 5000
	n_subclouds:	int	= 5
	harmonics:	  	object = None   # dict {m: (c_m, s_m)} or None
	velocity_model: str	= "hrm"
	v_2r:		   	float  = 0.0
	v_2t:		   	float  = 0.0
	phi_bar:		float  = 0.0
	nclouds:		int = 1
	vz_gradient: 	bool = False
	vz_profile:  	str  = z_profile	

	def __post_init__(self):
		"""
		Validate velocity_model and initialise the harmonics dict.
		"""
		_valid = ("circular", "radial", "hrm", "bisymmetric")
		if self.velocity_model not in _valid:
			raise ValueError(
				f"velocity_model must be one of {_valid}, "
				f"got '{self.velocity_model}'")

		if self.harmonics is None:
			self.harmonics = {1: (float(self.v_rot), -float(self.v_rad))}


@dataclass
class CubeConfig:
	"""
	Configuration for the output data cube.

	Attributes
	----------
	nx, ny : int
		Spatial dimensions [pixels].
	nv : int
		Number of velocity channels.
	dx, dy : float
		Pixel scale [arcsec/pixel].
	dv : float
		Velocity channel width [km/s].
	v_min : float
		Velocity of channel 0 [km/s].
	beam_fwhm : float
		Beam FWHM for 2-D Gaussian convolution [pixels].
		0 = no spatial smoothing.
	chan_width : float
		Spectral resolution expressed as Gaussian sigma [km/s].
		0 = no spectral smoothing.
	radial_step : float
		Spacing of the interpolated radial grid [same units as ring
		radii].  Smaller values → smoother cube but more computation.
	"""
	nx:		  int   = 128
	ny:		  int   = 128
	nv:		  int   = 64
	dx:		  float = 1.0	  # arcsec/pixel
	dy:		  float = 1.0	  # arcsec/pixel
	dv:		  float = 5.0	  # km/s per channel
	v_min:	   float = -160.0   # km/s
	beam_fwhm:   float = 3.0	  # pixels
	chan_width:  float = 0.0	  # km/s spectral sigma
	radial_step: float = 1.0	  # interpolation step (pixels / arcsec)


# ---------------------------------------------------------------------------
# Vertical Density Profiles
# ---------------------------------------------------------------------------

class VerticalProfile:
	"""Samplers for the vertical (z) gas density distribution."""

	@staticmethod
	def gaussian(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
		"""rho(z) ∝ exp(−z²/2σ²),  σ = scale."""
		return rng.normal(0.0, scale, n)

	@staticmethod
	def sech2(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
		"""rho(z) ∝ sech²(z/2h),  h = scale.  Inverse-CDF sampling."""
		u = rng.uniform(0.01, 0.99, n)
		return 2.0 * scale * np.arctanh(2.0 * u - 1.0)

	@staticmethod
	def exponential(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
		"""rho(z) ∝ exp(−|z|/h),  h = scale."""
		z_mag = rng.exponential(scale, n)
		signs = rng.choice([-1, 1], size=n)
		return z_mag * signs

	@staticmethod
	def uniform(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
		"""Uniform between −scale and +scale."""
		return rng.uniform(-scale, scale, n)

	@classmethod
	def sample(cls, profile: str, n: int, scale: float,
			   rng: np.random.Generator) -> np.ndarray:
		if scale == 0.0:
			return np.zeros(n)
		dispatch = {
			"gaussian":	cls.gaussian,
			"sech2":	   cls.sech2,
			"exponential": cls.exponential,
			"uniform":	 cls.uniform,
		}
		if profile not in dispatch:
			raise ValueError(
				f"Unknown z_profile '{profile}'. "
				f"Choose from: {list(dispatch.keys())}")
		return dispatch[profile](n, scale, rng)


# ---------------------------------------------------------------------------
# Radial interpolation
# ---------------------------------------------------------------------------

def _interpolate_rings(rings, radial_step):
	"""
	Linearly interpolate all ring parameters onto a fine radial grid.
 
	Scalar fields (v_disp, v_sys, inc, pa, x_center, y_center, z_scale)
	are interpolated directly.  Harmonic coefficients c_m(r) and s_m(r)
	are interpolated independently for every order m present in any ring.
	Missing orders in a ring default to (0.0, 0.0) so that interpolation
	across a boundary where a harmonic is turned off is smooth.
 
	Parameters
	----------
	rings : list of Ring
		Reference rings, sorted by increasing radius.
	radial_step : float
		Grid spacing in the same units as ring radii.
 
	Returns
	-------
	fine_rings : list of Ring
		One Ring per step across the full radial range.
	"""
	rings = sorted(rings, key=lambda r: r.radius)
	radii = np.array([r.radius for r in rings])
	clouds = np.array([r.nclouds for r in rings])
 
	# interp1d requires at least 2 points.  If only one ring is supplied
	# return it directly as a single fine ring (no interpolation possible).
	if len(rings) == 1:
		return [rings[0]]
 
	# --- Scalar fields ---
	scalar_fields = ["v_rot", "v_rad", "v_disp", "v_sys", "inc", "pa",
					 "x_center", "y_center", "z_scale",
					 "v_2r", "v_2t", "phi_bar"]
	interps = {}
	for f in scalar_fields:
		vals = np.array([getattr(r, f) for r in rings])
		interps[f] = interp1d(radii, vals, kind="linear",
							  fill_value="extrapolate")
 
	# --- Harmonic coefficients ---
	# Collect every order m present across all rings
	all_orders = set()
	for ring in rings:
		if ring.harmonics:
			all_orders.update(ring.harmonics.keys())
	all_orders = sorted(all_orders)
 
	harm_interps = {}   # (m, 'c') and (m, 's')
	for m in all_orders:
		c_vals = np.array([ring.harmonics.get(m, (0.0, 0.0))[0]
						   for ring in rings])
		s_vals = np.array([ring.harmonics.get(m, (0.0, 0.0))[1]
						   for ring in rings])
		harm_interps[(m, 'c')] = interp1d(radii, c_vals, kind="linear",
										   fill_value="extrapolate")
		harm_interps[(m, 's')] = interp1d(radii, s_vals, kind="linear",
										   fill_value="extrapolate")
 
	# --- Cloud density reference ---
	# Use the annulus area formula  A = 2*pi*r*w  for r > 0.
	# For r = 0 (central disk) use the filled-circle area  A = pi*(w/2)^2.
	# If the first ring is at r=0, use the first ring with r>0 as the
	# density reference to avoid the degenerate r=0 case dominating.
	rwidth = ring.width
	ref_candidates = [r for r in rings if r.radius > 0]
	ref_ring	   = ref_candidates[0] if ref_candidates else rings[0]
	ref_area	   = 2.0 * np.pi * ref_ring.radius * ref_ring.width
	ref_cloud_den  = ref_ring.n_clouds / max(ref_area, 1.0)
 
	# --- Build fine grid ---
	r_min	  = radii[0]
	r_max	  = radii[-1]
	fine_radii = np.arange(r_min, r_max + radial_step, radial_step)
 
	fine_rings = []
	for k, r in enumerate(fine_radii):
		width = rwidth
 
		# Correct area formula for each ring
		if r == 0.0:
			# Central disk: filled circle of radius width/2
			area = np.pi * (0.5 * width) ** 2
		else:
			area = 2.0 * np.pi * r * width
 
		ref_cloud_den=clouds[0]
		n_cl  = max(int(ref_cloud_den * area), 1000)
 
		# Interpolate harmonic coefficients at this radius
		h = {m: (float(harm_interps[(m, 'c')](r)),
				 float(harm_interps[(m, 's')](r)))
			 for m in all_orders}
 
		# v_rot and v_rad are interpolated directly (authoritative).
		# Keep harmonics[1] in sync so the 'harmonic' model path is
		# consistent with the scalar attributes.
		v_rot_r = float(interps["v_rot"](r))
		v_rad_r = float(interps["v_rad"](r))
		h[1] = (v_rot_r, -v_rad_r)
 
		fine_rings.append(Ring(
			radius		= float(r),
			width	   	= float(width),
			v_rot	   	= v_rot_r,
			v_disp	  	= float(interps["v_disp"](r)),
			v_sys	   	= float(interps["v_sys"](r)),
			v_rad	   	= v_rad_r,
			inc		 	= float(interps["inc"](r)),
			pa		  	= float(interps["pa"](r)),
			x_center	= float(interps["x_center"](r)),
			y_center	= float(interps["y_center"](r)),
			z_scale	 	= float(interps["z_scale"](r)),
			z_profile   = ref_ring.z_profile,
			n_clouds	= n_cl,
			n_subclouds = ref_ring.n_subclouds,
			harmonics   = h,
			velocity_model = ref_ring.velocity_model,
			v_2r	= float(interps["v_2r"](r)),
			v_2t	= float(interps["v_2t"](r)),
			phi_bar = float(interps["phi_bar"](r)),
			vz_gradient = ref_ring.vz_gradient,
			vz_profile  = ref_ring.vz_profile,			
		))
	return fine_rings
 

# ---------------------------------------------------------------------------
# Ring Builder
# ---------------------------------------------------------------------------

class RingBuilder:
	"""
	Populates a single tilted ring with clouds and deposits their
	signal into the 3-D data cube.
	"""

	def __init__(self, cube_config: CubeConfig, rng: np.random.Generator):
		self.cfg = cube_config
		self.rng = rng

	# ------------------------------------------------------------------
	# Cloud placement
	# ------------------------------------------------------------------

	def _sample_ring_positions(self, ring: Ring, n: int):
		"""
		Draw n cloud positions uniformly over the annulus surface.
 
		The disk frame has the receding major axis along +x_disk and
		+y_disk in the plane of the disk (perpendicular to the minor
		axis before inclination is applied).
 
		Special case: r = 0
			A ring at radius 0 is a central disk, not an annulus.
			Clouds are drawn uniformly over a filled circle of radius
			width/2, using r = sqrt(U) * width/2 (area-weighted) so
			that the surface density is uniform.  Negative radii are
			never produced.
 
		Returns
		-------
		x_disk, y_disk : positions in the galactic plane [arcsec]
		z_disk		 : heights above the plane [arcsec]
		phi			: azimuthal angles [rad],  phi=0 -> receding side
		"""
		phi = self.rng.uniform(0.0, 2.0 * np.pi, n)
		if ring.radius == 0.0:
			# Central disk: fill a circle of radius width/2 uniformly.
			# Area-weighted radial sampling: r = sqrt(U) * R_max
			r = np.sqrt(self.rng.uniform(0.0, 1.0, n)) * (0.5 * ring.width)
			r[0]=0
		else:
			r = ring.radius + self.rng.uniform(
					-0.5 * ring.width, 0.5 * ring.width, n)
					
		x_disk = r * np.cos(phi)
		y_disk = r * np.sin(phi)
		z_disk = VerticalProfile.sample(
					ring.z_profile, n, ring.z_scale, self.rng)

		return x_disk, y_disk, z_disk, phi

	def _disk_to_sky(self, x_disk, y_disk, z_disk, ring: Ring):
		"""
		Project galactic-plane coordinates to sky pixel coordinates.

		Convention
		----------
		PA is measured N→E, where North = +y (up) and East = −x (left),
		following the standard astronomical image orientation.
		PA points toward the RECEDING side of the galaxy.

		  PA =   0° → receding side points North  (+y)
		  PA =  90° → receding side points East   (−x, i.e. left)
		  PA = 180° → receding side points South  (−y)
		  PA = 270° → receding side points West   (+x, i.e. right)

		Transformation steps
		--------------------
		1. Inclination: compress y_disk by cos(inc); project z_disk
		   onto the sky y-axis as z * sin(inc).
		2. PA rotation: the receding major axis (+x_disk) must map to
		   the sky unit vector pointing PA degrees east of north:
			   r̂_receding = (−sin(PA), cos(PA))   [sky x, sky y]
		   The minor axis (+y_disk_proj) maps to the perpendicular:
			   r̂_minor	= (−cos(PA), −sin(PA))  [sky x, sky y]
		   Giving:
			   x_sky = −x_inc * sin(PA) − y_inc_proj * cos(PA)
			   y_sky =  x_inc * cos(PA) − y_inc_proj * sin(PA)
		3. Convert arcsec → pixels and add the kinematic centre.
		"""
		inc_rad = np.radians(ring.inc)
		pa_rad  = np.radians(ring.pa)

		# Step 1 — inclination
		x_inc	  = x_disk
		y_inc	  = y_disk * np.cos(inc_rad)	# = 0 at inc=90°
		z_proj	 = z_disk * np.sin(inc_rad)   	# z lifts along sky-y
		y_inc_proj = y_inc + z_proj

		# Step 2 — PA rotation
		# East = -x on a standard astronomical image (RA increases left)
		# Receding direction on sky: (-sin(PA), +cos(PA)) in (x, y)
		x_sky = -x_inc * np.sin(pa_rad) - y_inc_proj * np.cos(pa_rad)
		y_sky =  x_inc * np.cos(pa_rad) - y_inc_proj * np.sin(pa_rad)

		# Step 3 — pixels
		x_pix = ring.x_center + x_sky / self.cfg.dx
		y_pix = ring.y_center + y_sky / self.cfg.dy

		return x_pix, y_pix

	# ------------------------------------------------------------------
	# Velocity models
	# ------------------------------------------------------------------

	def _los_rotation(self, phi, ring, vrot_scale=None):
		"""
		Model 1 — Pure circular rotation.

			v_los = v_sys + v_rot * cos(phi) * sin(inc)

		Parameters
		----------
		phi  : np.ndarray   azimuthal angles [rad], phi=0 on receding axis
		ring : Ring

		Returns
		-------
		np.ndarray  line-of-sight velocity [km/s] per cloud
		"""
		inc_rad = np.radians(ring.inc)
		v_rot   = ring.v_rot if vrot_scale is None else ring.v_rot * vrot_scale
		return ring.v_sys + v_rot * np.cos(phi) * np.sin(inc_rad)

	def _los_radial(self, phi, ring):
		"""
		Model 2 — Circular rotation + radial inflow/outflow.

			v_los = v_sys + v_rot * cos(phi) * sin(inc)
						  - v_rad * sin(phi) * sin(inc)

		v_rad > 0 is an outflow (gas moving away from the centre).
		Radial flows are invisible on the major axis (cos(phi=0)=1,
		sin(phi=0)=0) and appear as a kinematic asymmetry between the
		two halves of the minor axis.

		Parameters
		----------
		phi  : np.ndarray
		ring : Ring

		Returns
		-------
		np.ndarray
		"""
		inc_rad = np.radians(ring.inc)
		sin_inc = np.sin(inc_rad)
		v_rot_los = ring.v_rot * np.cos(phi) * sin_inc
		v_rad_los = ring.v_rad * np.sin(phi) * sin_inc
		return ring.v_sys + v_rot_los - v_rad_los

	def _los_bisymmetric(self, phi, ring):
		"""
		Spekkens & Sellwood (2007) bisymmetric model.

			v_los = V_sys
				  + V_rot  * cos(theta)		 * sin(inc)
				  - V_2r   * sin(2*phi_disk)	* sin(theta) * sin(inc)
				  - V_2t   * cos(2*phi_disk)	* cos(theta) * sin(inc)

		where phi_disk = theta - phi_bar  (phi_bar in radians).
		theta = phi in our notation (azimuthal angle from receding major axis).
		"""
		inc_rad	 = np.radians(ring.inc)
		phi_bar_rad = np.radians(ring.phi_bar)
		sin_inc	 = np.sin(inc_rad)

		phi_disk = phi - phi_bar_rad		# angle relative to bar axis

		v  = ring.v_sys
		v += ring.v_rot * np.cos(phi)							* sin_inc
		v -= ring.v_2r  * np.sin(2.0 * phi_disk) * np.sin(phi)  * sin_inc
		v -= ring.v_2t  * np.cos(2.0 * phi_disk) * np.cos(phi)  * sin_inc

		return v
	
	def _los_harmonic(self, phi, ring):
		"""
		Model 3 — General harmonic decomposition.

			v_los = v_sys + sin(inc) * Σ_m [ c_m*cos(m*phi)
										   + s_m*sin(m*phi) ]

		The sum runs over all orders m stored in ring.harmonics.
		Missing orders contribute nothing.

		Relation to simpler models at m=1 only:
			c_1 =  v_rot  →  reduces to 'circular' model
			s_1 = -v_rad  →  adds radial component ('radial' model)

		Parameters
		----------
		phi  : np.ndarray
		ring : Ring

		Returns
		-------
		np.ndarray
		"""
		inc_rad = np.radians(ring.inc)
		sin_inc = np.sin(inc_rad)
		v = np.full_like(phi, ring.v_sys, dtype=float)
		if ring.harmonics:
			for m, (c_m, s_m) in ring.harmonics.items():
				v += (c_m * np.cos(m * phi) + s_m * np.sin(m * phi)) * sin_inc
		return v

	def _los_velocity(self, phi, ring, vrot_scale=None):
		"""
		Dispatch to the correct velocity model based on ring.velocity_model.

		Models
		------
		'circular'	: pure circular rotation	  (v_rot, v_sys)
		'radial'	  : rotation + radial flow	  (v_rot, v_rad, v_sys)
		'bisymmetric' : Spekkens & Sellwood bar	 (v_rot, v_2r, v_2t,
													 phi_bar, v_sys)
		'harmonic'	: general harmonic sum		(harmonics dict, v_sys)
		"""
		if ring.velocity_model == "circular":
			return self._los_rotation(phi, ring, vrot_scale)
		elif ring.velocity_model == "radial":
			return self._los_radial(phi, ring)
		elif ring.velocity_model == "bisymmetric":
			return self._los_bisymmetric(phi, ring)
		else:									# 'harmonic' (default)
			return self._los_harmonic(phi, ring)

	# ------------------------------------------------------------------
	# Cube deposition
	# ------------------------------------------------------------------

	def build_ring(self, ring: Ring, cube: np.ndarray) -> None:
		"""
		Deposit all clouds from this ring into the cube (in place).

		Each cloud is split into n_subclouds.  Every subcloud receives
		the same sky position as its parent cloud but an independently
		drawn random velocity kick ~ N(0, v_disp).
		"""
		cfg = self.cfg
		n   = ring.n_clouds

		x_disk, y_disk, z_disk, phi 	= self._sample_ring_positions(ring, n)
		x_pix, y_pix			   		= self._disk_to_sky(
										 x_disk, y_disk, z_disk, ring)
										 

		vrot_scale = None
		if ring.vz_gradient and ring.z_scale > 0 and ring.radius > 0:
			from .vertical_rotation import get_table
			table = get_table(ring.vz_profile)
			alpha = ring.z_scale / ring.radius
			vrot_scale = table.vc_ratio(z_disk / ring.z_scale, alpha)

		v_sys_cloud				 		= self._los_velocity(phi, ring, vrot_scale )

		# Replicate each cloud n_subclouds times
		ns	= ring.n_subclouds
		x_sub = np.repeat(x_pix,		ns)
		y_sub = np.repeat(y_pix,		ns)
		v_sub = np.repeat(v_sys_cloud,  ns)

		# Independent random velocity per subcloud
		v_rand  = self.rng.normal(0.0, abs(ring.v_disp), len(v_sub))
		v_total = v_sub + v_rand

		# Convert to integer indices
		xi = np.round(x_sub).astype(int)
		yi = np.round(y_sub).astype(int)
		ci = np.round((v_total - cfg.v_min) / cfg.dv).astype(int)

		# Mask out-of-bounds
		mask = (
			(xi >= 0) & (xi < cfg.nx) &
			(yi >= 0) & (yi < cfg.ny) &
			(ci >= 0) & (ci < cfg.nv)
		)
		np.add.at(cube, (ci[mask], yi[mask], xi[mask]), 1.0)


# ---------------------------------------------------------------------------
# Tilted-Ring Model  (main class)
# ---------------------------------------------------------------------------

class TiltedRingModel:
	"""
	Full tilted-ring model builder using the cloud approximation method.

	The user provides a list of Ring objects at discrete radii.
	Before building, the code interpolates all parameters onto a fine
	radial grid (spacing = CubeConfig.radial_step) to ensure smooth,
	gap-free emission.

	Parameters
	----------
	cube_config : CubeConfig
	seed : int, optional
		Random seed for reproducibility.

	Example
	-------
	>>> cfg = CubeConfig(nx=128, ny=128, nv=80, dv=5.0, v_min=-200.0,
	...				  beam_fwhm=3.0, radial_step=1.0)
	>>> model = TiltedRingModel(cfg, seed=42)
	>>> rings = [
	...	 Ring(radius=10, width=10, v_rot=100, v_disp=8, v_sys=0,
	...		  inc=60, pa=45, x_center=64, y_center=64),
	...	 Ring(radius=50, width=10, v_rot=170, v_disp=8, v_sys=0,
	...		  inc=60, pa=45, x_center=64, y_center=64),
	... ]
	>>> cube = model.build(rings)
	"""

	def __init__(self, cube_config, psf_lsf, seed=None, planner_effort='FFTW_ESTIMATE'):
		self.cfg   = cube_config
		self.cfg_lsf   = psf_lsf		
		self.rng   = np.random.default_rng(seed)
		self._rb   = RingBuilder(cube_config, self.rng)
		
		# ConvolutionEngine owns the FFT plan and PSF cache.
		# It is created once and reused across all build() calls so
		# that pyfftw planning cost is paid only on the first call.
		self._conv = ConvolutionEngine(cube_config, psf_lsf, planner_effort=planner_effort)
	
	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def build(self, rings, verbose=True):
		"""
		Build the full model cube.

		Parameters
		----------
		rings : list of Ring
			Reference rings at discrete radii (any order).
		verbose : bool
			Print per-ring progress.

		Returns
		-------
		cube : np.ndarray, shape (nv, ny, nx)
		"""
		cfg = self.cfg
		cfg_lsf = self.cfg_lsf
		# Interpolate onto fine radial grid to eliminate gaps
		fine_rings = _interpolate_rings(rings, cfg_lsf.radial_step)
		if verbose:
			print(f"  Interpolated {len(rings)} input rings → "
				  f"{len(fine_rings)} fine rings "
				  f"(step = {cfg.radial_step})")

		cube = np.zeros((cfg.nv, cfg.ny, cfg.nx), dtype=np.float64)
		for i, ring in enumerate(fine_rings):
			if verbose and (i % max(1, len(fine_rings) // 10) == 0):
				print(f"  Ring {i+1:4d}/{len(fine_rings)} "
					  f"r={ring.radius:6.1f}  "
					  f"v_rot={ring.v_rot:6.1f} km/s  "
					  f"PA={ring.pa:5.1f}°  "
					  f"inc={ring.inc:4.1f}°")
			self._rb.build_ring(ring, cube)

		# Delegate convolution to ConvolutionEngine (convolution.py)
		#xc,yc = 28, 37
		#plt.plot(np.arange(cube.shape[0]), cube[:, yc, xc], 'k-');plt.show()
		#plt.imshow(np.sum(cube, axis = 0), origin = 'lower');plt.show()						
		#msk = np.sum(cube, axis = 0) =! 0
		cube = self._conv.apply(cube, verbose=False)
		#plt.imshow(np.sum(cube, axis = 0), origin = 'lower');plt.show()				
		#plt.plot(np.arange(cube.shape[0]), cube[:, yc, xc], 'r-');plt.show()
		return cube#*msk

	def velocity_axis(self) -> np.ndarray:
		"""Return velocity axis [km/s] for all channels."""
		return self.cfg.v_min + np.arange(self.cfg.nv) * self.cfg.dv

	def spatial_axes(self):
		"""Return (x_arcsec, y_arcsec) relative to the cube centre."""
		cfg = self.cfg
		x = (np.arange(cfg.nx) - cfg.nx / 2.0) * cfg.dx
		y = (np.arange(cfg.ny) - cfg.ny / 2.0) * cfg.dy
		return x, y

	# ------------------------------------------------------------------
	# Instrumental effects
	# ------------------------------------------------------------------

	def _apply_beam(self, cube: np.ndarray) -> np.ndarray:
		"""Convolve each channel with a 2-D Gaussian beam."""
		if self.cfg.beam_fwhm <= 0:
			return cube
		sigma = self.cfg.beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
		out = np.empty_like(cube)
		for v in range(cube.shape[0]):
			out[v] = gaussian_filter(cube[v], sigma=sigma)
		return out

	def _apply_spectral_response(self, cube: np.ndarray) -> np.ndarray:
		"""Convolve along the velocity axis with a Gaussian."""
		if self.cfg.chan_width <= 0:
			return cube
		sigma = self.cfg.chan_width / self.cfg.dv
		# gaussian_filter applied along axis 0 only
		return gaussian_filter(cube, sigma=[sigma, 0.0, 0.0])

	# ------------------------------------------------------------------
	# Moment maps
	# ------------------------------------------------------------------

	def moment0(self, cube: np.ndarray) -> np.ndarray:
		"""Integrated intensity (moment 0)."""
		return np.sum(cube, axis=0)

	def moment1(self, cube: np.ndarray,
				threshold: float = 0.0) -> np.ndarray:
		"""Intensity-weighted mean velocity (moment 1)."""
		vel  = self.velocity_axis()
		mask = cube > threshold
		m0   = np.sum(cube * mask, axis=0)
		m1   = np.sum(cube * mask * vel[:, None, None], axis=0)
		with np.errstate(invalid="ignore", divide="ignore"):
			return np.where(m0 > 0, m1 / m0, np.nan)

	def moment2(self, cube: np.ndarray,
				threshold: float = 0.0) -> np.ndarray:
		"""Intensity-weighted velocity dispersion (moment 2)."""
		vel  = self.velocity_axis()
		m1   = self.moment1(cube, threshold)
		mask = cube > threshold
		m0   = np.sum(cube * mask, axis=0)
		dv2  = (vel[:, None, None] - m1[None, :, :]) ** 2
		with np.errstate(invalid="ignore", divide="ignore"):
			return np.where(m0 > 0,
							np.sqrt(np.sum(cube * mask * dv2, axis=0) / m0),
							np.nan)

	# ------------------------------------------------------------------
	# PV diagram
	# ------------------------------------------------------------------

	def pv_diagram(self, cube, pa, center=None, width=3):
		"""
		Extract a position-velocity diagram along a given PA.

		The slice runs along the direction PA degrees east of north.
		With the standard astronomical convention (North=+y, East=-x):

			r_hat_slice = (-sin(PA),  cos(PA))   [dx, dy per pixel step]
			r_hat_perp  = (-cos(PA), -sin(PA))   [dx, dy perpendicular]

		Positive offsets correspond to the receding side (PA direction).

		Parameters
		----------
		pa : float
			Slice PA [degrees], N->E, East=-x convention.
		center : (x_pix, y_pix), optional.  Default: cube centre.
		width : int
			Number of pixels averaged perpendicular to the slice.

		Returns
		-------
		pv : np.ndarray, shape (nv, n_offsets)
		offsets : np.ndarray  [pixels]
		"""
		cfg = self.cfg
		cx, cy = (cfg.nx / 2.0, cfg.ny / 2.0) if center is None else center
		pa_rad = np.radians(pa)

		sin_pa, cos_pa = np.sin(pa_rad), np.cos(pa_rad)

		half = max(cfg.nx, cfg.ny) // 2
		offsets = np.arange(-half, half + 1)
		pv = np.zeros((cfg.nv, len(offsets)))

		for k, off in enumerate(offsets):
			# Step along the slice direction: r_hat = (-sin(PA), cos(PA))
			xc = cx - off * sin_pa
			yc = cy + off * cos_pa

			total, count = np.zeros(cfg.nv), 0
			for w in range(-(width // 2), width // 2 + 1):
				# Step perpendicular: r_hat_perp = (-cos(PA), -sin(PA))
				xp = int(round(xc - w * cos_pa))
				yp = int(round(yc - w * sin_pa))
				if 0 <= xp < cfg.nx and 0 <= yp < cfg.ny:
					total += cube[:, yp, xp]
					count += 1
			if count:
				pv[:, k] = total / count

		return pv, offsets


# ---------------------------------------------------------------------------
# FITS export
# ---------------------------------------------------------------------------

def save_fits(cube: np.ndarray, model: TiltedRingModel,
			  filename: str = "tilted_ring_model.fits") -> None:
	"""Save the model cube as a FITS file with basic WCS (requires astropy)."""
	try:
		from astropy.io import fits
		from astropy.wcs import WCS
	except ImportError:
		raise ImportError("pip install astropy")

	cfg = model.cfg
	w = WCS(naxis=3)
	w.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
	w.wcs.cdelt = [-cfg.dx / 3600.0, cfg.dy / 3600.0, cfg.dv * 1e3]
	w.wcs.crpix = [cfg.nx // 2, cfg.ny // 2, 1]
	w.wcs.crval = [0.0, 0.0, cfg.v_min * 1e3]
	w.wcs.cunit = ["deg", "deg", "m/s"]

	hdr = w.to_header()
	hdr["BUNIT"] = "Jy/beam"
	hdr["BMAJ"]  = cfg.beam_fwhm * cfg.dx / 3600.0
	hdr["BMIN"]  = cfg.beam_fwhm * cfg.dy / 3600.0

	fits.PrimaryHDU(data=cube.astype(np.float32),
					header=hdr).writeto(filename, overwrite=True)
	print(f"Saved: {filename}")







class PsF_LsF:
	def __init__(self, cube_config):
		self.cfg   = cube_config











# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	print("=" * 60)
	print("  Tilted-Ring Model — Cloud Approximation Method")
	print("=" * 60)

	cfg = CubeConfig(
		nx=128, ny=128, nv=80,
		dx=1.0, dy=1.0,
		dv=5.0, v_min=-200.0,
		beam_fwhm=3.0,
		chan_width=5.0,
		radial_step=1.0,	  # 1-pixel fine grid → no gaps
	)

	# Reference rings — parameters at key radii only.
	# Interpolation fills everything in between.
	common = dict(
		v_sys=0.0, inc=60.0, pa=45.0,   # PA=45° → receding side NE
		x_center=64.0, y_center=64.0,
		z_scale=2.0, z_profile="gaussian",
		n_clouds=8000, n_subclouds=5,
	)
	rings = [
		Ring(radius= 5, width=5, v_rot= 80, v_disp=10, **common),
		Ring(radius=15, width=5, v_rot=130, v_disp= 9, **common),
		Ring(radius=25, width=5, v_rot=160, v_disp= 8, **common),
		Ring(radius=35, width=5, v_rot=170, v_disp= 8, **common),
		Ring(radius=45, width=5, v_rot=165, v_disp= 9, **common),
		Ring(radius=55, width=5, v_rot=150, v_disp=10, **common),
	]

	print("\nBuilding model...")
	model = TiltedRingModel(cfg, seed=42)
	cube  = model.build(rings, verbose=True)
	print(f"\nCube shape : {cube.shape}  (nv, ny, nx)")
	print(f"Total flux : {cube.sum():.0f} counts")

	# Moment maps & PV
	thr  = 0.05 * cube.max()
	mom0 = model.moment0(cube)
	mom1 = model.moment1(cube, threshold=thr)
	mom2 = model.moment2(cube, threshold=thr)
	pv, off = model.pv_diagram(cube, pa=45.0, center=(64, 64), width=5)
	vel  = model.velocity_axis()

	fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
	fig.suptitle("Tilted-Ring Model — Cloud Approximation  "
				 "(PA = 45° N→E, receding side)", fontsize=12)

	im0 = axes[0].imshow(mom0, origin="lower", cmap="inferno")
	axes[0].set_title("Moment 0 — Flux"); plt.colorbar(im0, ax=axes[0])

	im1 = axes[1].imshow(mom1, origin="lower", cmap="RdBu_r",
						 vmin=-200, vmax=200)
	axes[1].set_title("Moment 1 — Velocity [km/s]")
	plt.colorbar(im1, ax=axes[1])

	im2 = axes[2].imshow(mom2, origin="lower", cmap="magma", vmin=0)
	axes[2].set_title("Moment 2 — Dispersion [km/s]")
	plt.colorbar(im2, ax=axes[2])

	axes[3].contourf(off, vel, pv, levels=20, cmap="viridis")
	axes[3].set_title("PV Diagram (PA = 45°)")
	axes[3].set_xlabel("Offset [pix]"); axes[3].set_ylabel("Velocity [km/s]")
	axes[3].axhline(0, color="w", lw=0.8, ls="--")
	axes[3].axvline(0, color="w", lw=0.8, ls="--")

	plt.tight_layout()
	plt.savefig("tilted_ring_model_maps.png", dpi=150, bbox_inches="tight")
	print("\nSaved: tilted_ring_model_maps.png")
	plt.show()
