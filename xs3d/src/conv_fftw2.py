"""
convolution.py
==============
3D PSF convolution engine for cloud_fit_engine.py (tilted_ring_model.py).

Provides the ConvolutionEngine class which applies a combined spatial
(elliptical Gaussian beam) and spectral (Gaussian instrumental response)
convolution to a 3D datacube in a single FFT pass.

When pyfftw is installed the convolution uses pre-planned rfftn/irfftn
transforms that are cached across calls, giving a 3-10x speedup over
scipy.ndimage.gaussian_filter.  When pyfftw is not installed the code
falls back to scipy.ndimage.gaussian_filter transparently.

The optimal FFT size for each cube dimension is determined automatically
by fft_size_advisor.py (if available), which decides whether zero-padding
is beneficial based on prime factorisation of the cube dimensions.

Public API
----------
ConvolutionEngine(cfg)
	Initialise the engine for a given CubeConfig.
	Pre-planning happens lazily on the first call to apply().

ConvolutionEngine.apply(cube, verbose=False) -> np.ndarray
	Apply the 3D PSF convolution to the cube.  Returns a new array of
	the same shape as the input.

Module-level flags
------------------
PYFFTW_AVAILABLE : bool
	True if pyfftw is installed and importable.
ADVISOR_AVAILABLE : bool
	True if fft_size_advisor.py is importable.

Install pyfftw
--------------
	pip install pyfftw
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
	import pyfftw
	import pyfftw.builders as _fftw_builders
	import multiprocessing as _mp
	PYFFTW_AVAILABLE = True
	pyfftw.interfaces.cache.enable()   # cache FFT plans between calls
except ImportError:
	PYFFTW_AVAILABLE = False

try:
	from fft_size_advisor import (
		pad_cube		as _pad_cube,
		unpad_cube	  as _unpad_cube,
		optimal_fft_size as _optimal_fft_size,
	)
	ADVISOR_AVAILABLE = True
except ImportError:
	ADVISOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# ConvolutionEngine
# ---------------------------------------------------------------------------

class ConvolutionEngine:
	"""
	Applies a 3D PSF convolution (spatial beam + spectral response) to
	a datacube.

	The PSF is the outer product of:
	  - A 2D elliptical Gaussian beam  (BMAJ, BMIN, beam_pa)
	  - A 1D Gaussian spectral response (chan_width / dv channels)

	When pyfftw is available:
	  - The FFT plan is computed once on the first call and reused.
	  - The PSF FFT is computed once and cached.
	  - The cube is zero-padded to FFTW-optimal sizes if beneficial
		(determined by fft_size_advisor).
	  - rfftn is used (exploits real-valued input to halve memory).

	When pyfftw is not available:
	  - Falls back to scipy.ndimage.gaussian_filter (two separate passes:
		2D spatial per channel + 1D spectral along axis 0).
	  - Only circular beams are supported in the fallback path.

	Parameters
	----------
	cfg : CubeConfig
		Must have the following attributes:
		  nx, ny, nv	  : cube dimensions
		  dv			  : velocity channel width [km/s]
		  beam_fwhm	   : beam major-axis FWHM [pixels]
		  bmin_fwhm	   : beam minor-axis FWHM [pixels] (-1 = circular)
		  beam_pa		 : beam position angle [degrees] N->E
		  chan_width	  : spectral Gaussian sigma [km/s]
		  n_fft_threads   : number of pyfftw threads (-1 = all cores)

	Examples
	--------
	>>> from convolution import ConvolutionEngine
	>>> engine = ConvolutionEngine(cfg)
	>>> cube_convolved = engine.apply(cube, verbose=True)
	"""

	def __init__(self, cfg, psf_lsf, planner_effort='FFTW_ESTIMATE'):
		self.cfg 		= cfg
		pixel 			= cfg.pix_arcs
		self.beam_fwhm	= psf_lsf.bmaj/pixel
		self.bmin_fwhm	= psf_lsf.bmin/pixel
		self.beam_pa 	= psf_lsf.bpa
		self.dv			= psf_lsf.cdelt3_kms
		self.sigma_kms	= psf_lsf.sigma_inst_kms
		self.nthreads	= psf_lsf.nthreads


		self.planner_effort = planner_effort
        
		# Cached state — populated lazily on first apply() call
		self._psf_f 	= None	  # FFT of 3D PSF (complex ndarray)
		self._fft_plan  = None	  # pyfftw forward rfftn plan
		self._ifft_plan = None	  # pyfftw inverse irfftn plan
		self._fft_nx	= None	  # padded FFT size along x
		self._fft_ny	= None	  # padded FFT size along y
		self._fft_nv	= None	  # padded FFT size along v
		
	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------

	def apply(self, cube, verbose=False):
		"""
		Convolve the cube with the 3D PSF.

		Parameters
		----------
		cube : np.ndarray, shape (nv, ny, nx)
			Raw model cube (output of cloud deposition).
		verbose : bool
			Print progress messages.

		Returns
		-------
		np.ndarray, shape (nv, ny, nx)
			Convolved cube.  Same shape and dtype as input.
		"""
		cfg = self.cfg

		# Skip entirely if no smoothing is requested
		no_spatial  = self.beam_fwhm  <= 0
		no_spectral = self.sigma_kms <= 0
		if no_spatial and no_spectral:
			return cube

		if not PYFFTW_AVAILABLE:
			if verbose:
				print("  [PSF] pyfftw not available — "
					  "using scipy.ndimage.gaussian_filter (fallback)")
			return self._apply_scipy(cube)

		return self._apply_pyfftw(cube, verbose=verbose)

	# ------------------------------------------------------------------
	# pyfftw path
	# ------------------------------------------------------------------

	def _apply_pyfftw(self, cube, verbose=False):
		"""
		Full pyfftw 3D FFT convolution path.

		Steps
		-----
		1. Determine optimal FFT sizes (fft_size_advisor).
		2. Zero-pad cube if beneficial.
		3. Pre-plan rfftn / irfftn (once only — cached).
		4. Build and FFT the 3D PSF (once only — cached).
		5. Forward FFT the cube.
		6. Pointwise multiply in Fourier space.
		7. Inverse FFT.
		8. Trim back to original shape.
		"""
		cfg = self.cfg
		nv, ny, nx = cube.shape

		if verbose:
			bmin_str = (f"{self.bmin_fwhm:.1f}px"
						if self.bmin_fwhm > 0 else "circular")
			print(f"  [PSF] pyfftw 3D FFT  "
				  f"BMAJ={self.beam_fwhm:.1f}px  BMIN={bmin_str}  "
				  f"PA={self.beam_pa:.1f}°  "
				  f"chan_width={self.sigma_kms:.2f} km/s  "
				  f"threads={self.nthreads}")

		# Step 1: optimal FFT sizes
		nx_fft, ny_fft, nv_fft = self._optimal_sizes(nx, ny, nv, cfg)

		if verbose and (nx_fft, ny_fft, nv_fft) != (nx, ny, nv):
			print(f"  [PSF] padding {nx}×{ny}×{nv} → "
				  f"{nx_fft}×{ny_fft}×{nv_fft} (FFTW-optimal)")

		# Step 2: zero-pad
		cube_pad, orig_shape = self._pad(cube, nx_fft, ny_fft, nv_fft)

		# Step 3: plan FFT (once only)
		if self._needs_replanning(nx_fft, ny_fft, nv_fft):
			if verbose:
				print(f"  [PSF] planning FFT "
					  f"({nv_fft}×{ny_fft}×{nx_fft}) — one-time cost ...")
			self._plan(cube_pad, nx_fft, ny_fft, nv_fft, cfg)
			self._psf_f = None   # invalidate PSF cache when plan changes

		# Step 4: build PSF FFT (once only)
		if self._psf_f is None:
			if verbose:
				print("  [PSF] computing PSF FFT ...")
			psf_3d	  = self._build_psf_3d(nx_fft, ny_fft, nv_fft, cfg)
			self._psf_f = np.fft.rfftn(psf_3d)

		# Step 5: forward FFT
		np.copyto(self._fft_plan.input_array, cube_pad)
		cube_f = self._fft_plan()

		# Step 6: pointwise multiply in Fourier space
		cube_f *= self._psf_f

		# Step 7: inverse FFT
		np.copyto(self._ifft_plan.input_array, cube_f)
		result_pad = self._ifft_plan()

		# Step 8: trim back
		result = self._unpad(result_pad, orig_shape)
		
		return np.ascontiguousarray(result)

	def _optimal_sizes(self, nx, ny, nv, cfg):
		"""Return (nx_fft, ny_fft, nv_fft) — padded sizes if beneficial."""
		if not ADVISOR_AVAILABLE:
			return nx, ny, nv
		sigma_chan = self.sigma_kms / self.dv if self.sigma_kms > 0 else 0.0
		return _optimal_fft_size(
			nx, ny, nv,
			beam_fwhm_pix  = self.beam_fwhm,
			chan_sigma_chan = sigma_chan,
			verbose		= False,
		)

	def _pad(self, cube, nx_fft, ny_fft, nv_fft):
		"""Zero-pad cube to (nv_fft, ny_fft, nx_fft).  No-op if equal."""
		if ADVISOR_AVAILABLE:
			return _pad_cube(cube, nx_fft, ny_fft, nv_fft)
		nv, ny, nx = cube.shape
		if (nx_fft, ny_fft, nv_fft) == (nx, ny, nv):
			return cube, (nv, ny, nx)
		pad_width = [(0, nv_fft - nv), (0, ny_fft - ny), (0, nx_fft - nx)]
		return (np.pad(cube, pad_width, mode='constant',
					   constant_values=0.0), (nv, ny, nx))

	def _unpad(self, padded, orig_shape):
		"""Trim padded cube back to orig_shape."""
		if ADVISOR_AVAILABLE:
			return _unpad_cube(padded, orig_shape)
		nv, ny, nx = orig_shape
		return padded[:nv, :ny, :nx]

	def _needs_replanning(self, nx_fft, ny_fft, nv_fft):
		"""True if no plan exists or the padded sizes have changed."""
		return (self._fft_plan is None or
				self._fft_nx != nx_fft or
				self._fft_ny != ny_fft or
				self._fft_nv != nv_fft)

	def _plan(self, cube_pad, nx_fft, ny_fft, nv_fft, cfg):
		"""
		Pre-plan the pyfftw rfftn / irfftn transforms.

		FFTW_MEASURE spends ~1 second profiling to find the fastest
		algorithm for this exact array shape and thread count.  The plan
		is then reused for every subsequent call — amortised to zero cost
		over a full fitting run with hundreds of evaluations.
		"""
		n_threads = (self.nthreads if self.nthreads > 0
					 else _mp.cpu_count())

		# Forward: real cube → complex Fourier cube
		self._fft_plan = _fftw_builders.rfftn(
			cube_pad,
			threads		= n_threads,
			planner_effort = self.planner_effort,
		)

		# Allocate output buffer matching forward FFT output shape
		cube_f_shape = self._fft_plan.output_array.shape
		cube_f_buf   = pyfftw.empty_aligned(cube_f_shape, dtype='complex128')

		# Inverse: complex → real, output shape = original padded cube shape
		self._ifft_plan = _fftw_builders.irfftn(
			cube_f_buf,
			s			  = cube_pad.shape,
			threads		= n_threads,
			planner_effort = self.planner_effort,
		)

		self._fft_nx = nx_fft
		self._fft_ny = ny_fft
		self._fft_nv = nv_fft


	def _build_psf_3d(self,nx_fft, ny_fft, nv_fft, cfg):
		"""
		Build the 3D PSF as an outer product of spatial and spectral kernels.

		Spatial kernel: 2D elliptical Gaussian on the FFT frequency grid.
		  BMAJ = self.beam_fwhm [pixels]
		  BMIN = self.bmin_fwhm [pixels] (-1 = use BMAJ, circular beam)
		  PA   = self.beam_pa   [degrees], N->E

		Spectral kernel: 1D Gaussian on the FFT frequency grid.
		  sigma = self.sigma_kms / self.dv  [channels]

		Both kernels are centred at element (0,0,0) — the zero-frequency
		element — as required for FFT convolution without fftshift.

		Parameters
		----------
		nx_fft, ny_fft, nv_fft : int   padded FFT dimensions
		cfg : CubeConfig

		Returns
		-------
		psf_3d : np.ndarray, shape (nv_fft, ny_fft, nx_fft)
			Normalised 3D PSF.
		"""
		# Frequency-domain coordinate grids (centred at 0 = zero frequency)
		x = np.fft.fftfreq(nx_fft) * nx_fft	# pixel offsets
		y = np.fft.fftfreq(ny_fft) * ny_fft
		v = np.fft.fftfreq(nv_fft) * nv_fft

		XX, YY = np.meshgrid(x, y, indexing='xy')   # (ny_fft, nx_fft)

		# --- 2D spatial PSF ---
		sigma_maj = self.beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
		sigma_min = (self.bmin_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
					 if self.bmin_fwhm > 0 else sigma_maj)

		if sigma_maj > 0:
			pa_rad = np.radians(self.beam_pa)
			x_rot  =  XX * np.cos(pa_rad) + YY * np.sin(pa_rad)
			y_rot  = -XX * np.sin(pa_rad) + YY * np.cos(pa_rad)
			psf_2d = np.exp(-0.5 * (x_rot**2 / sigma_maj**2
								   + y_rot**2 / sigma_min**2))
			psf_2d /= psf_2d.sum()
		else:
			psf_2d	   = np.zeros((ny_fft, nx_fft))
			psf_2d[0, 0] = 1.0   # delta function — no spatial smoothing

		# --- 1D spectral PSF ---
		sigma_v = self.sigma_kms / self.dv if self.sigma_kms > 0 else 0.0

		if sigma_v > 0:
			psf_1d = np.exp(-0.5 * v**2 / sigma_v**2)
			psf_1d /= psf_1d.sum()
		else:
			psf_1d	   = np.zeros(nv_fft)
			psf_1d[0]	= 1.0   # delta function — no spectral smoothing

		# --- Outer product → 3D PSF ---
		psf_3d = psf_1d[:, np.newaxis, np.newaxis] * psf_2d[np.newaxis, :, :]
		return psf_3d

	# ------------------------------------------------------------------
	# scipy fallback path
	# ------------------------------------------------------------------

	def _apply_scipy(self, cube):
		"""
		Fallback using scipy.ndimage.gaussian_filter.

		Two separate passes:
		  1. 2D Gaussian per velocity channel (spatial beam, circular only)
		  2. 1D Gaussian along axis 0 (spectral response)

		Note: only circular beams are supported here.  For elliptical
		beams install pyfftw.
		"""
		cfg = self.cfg
		out = cube

		if self.beam_fwhm > 0:
			sigma_px = self.beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
			smoothed = np.empty_like(out)
			for v in range(out.shape[0]):
				smoothed[v] = gaussian_filter(out[v], sigma=sigma_px)
			out = smoothed

		if self.sigma_kms > 0:
			sigma_ch = self.sigma_kms / self.dv
			out = gaussian_filter(out, sigma=[sigma_ch, 0.0, 0.0])

		return out

	# ------------------------------------------------------------------
	# Diagnostics
	# ------------------------------------------------------------------

	def info(self):
		"""Print a summary of the engine state."""
		cfg = self.cfg
		print("ConvolutionEngine")
		print(f"  Backend	 : {'pyfftw' if PYFFTW_AVAILABLE else 'scipy (fallback)'}")
		print(f"  Advisor	 : {'fft_size_advisor' if ADVISOR_AVAILABLE else 'not available'}")
		print(f"  beam_fwhm   : {self.beam_fwhm:.2f} px")
		bmin = self.bmin_fwhm if self.bmin_fwhm > 0 else self.beam_fwhm
		print(f"  bmin_fwhm   : {bmin:.2f} px  "
			  f"({'elliptical' if self.bmin_fwhm > 0 else 'circular'})")
		print(f"  beam_pa	 : {self.beam_pa:.1f}°")
		print(f"  chan_width  : {self.sigma_kms:.2f} km/s")
		print(f"  n_threads   : {self.nthreads} "
			  f"({'all cores' if self.nthreads < 0 else 'fixed'})")
		if self._fft_plan is not None:
			print(f"  FFT planned : {self._fft_nv}×{self._fft_ny}×{self._fft_nx}")
			print(f"  PSF cached  : {'yes' if self._psf_f is not None else 'no'}")
		else:
			print("  FFT planned : not yet (lazy — on first apply() call)")

# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def apply_psf_3d(cube, cfg, verbose=False):
	"""
	Convenience wrapper: create a ConvolutionEngine and apply it once.

	Useful for one-off convolutions.  For repeated calls (e.g. inside a
	fitting loop) create a ConvolutionEngine once and call .apply() to
	benefit from plan and PSF caching.

	Parameters
	----------
	cube	: np.ndarray, shape (nv, ny, nx)
	cfg	 : CubeConfig
	verbose : bool

	Returns
	-------
	np.ndarray, shape (nv, ny, nx)
	"""
	return ConvolutionEngine(cfg).apply(cube, verbose=verbose)
