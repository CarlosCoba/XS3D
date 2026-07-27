"""
fft_size_advisor.py
===================
Utility to determine the optimal FFT size for 3D datacube convolution,
deciding automatically whether zero-padding is beneficial, and applying
the padding / unpadding when needed.

The decision is based on three criteria:

  1. FFTW smoothness: all prime factors must be in {2, 3, 5, 7}.
     Dimensions with large prime factors degrade to O(N^2) sub-transforms
     and MUST be padded to the nearest smooth number.

  2. Cost/benefit: padding is only worthwhile if the per-element speedup
     from a smoother FFT size outweighs the cost of processing more
     elements.  We compute the net operation count for each candidate
     size and pad only if the net cost decreases.

  3. Circular convolution wrap-around: for very large kernels relative
     to the cube size, padding may be needed to avoid aliasing.
     We check whether the kernel radius exceeds the safe margin.

Public API
----------
    optimal_fft_size(nx, ny, nv, ...)
        Returns (nx_opt, ny_opt, nv_opt) — the best FFT sizes.

    pad_cube(cube, nx_opt, ny_opt, nv_opt)
        Zero-pads cube to the optimal sizes.  Returns (padded_cube,
        original_shape) where original_shape is needed for unpadding.

    unpad_cube(padded_cube, original_shape)
        Slices the padded cube back to the original spatial/spectral
        extent, discarding the zero-padding.

    fft_size_advisor(nx, ny, nv, ...)
        Prints a full analysis report and returns (nx_opt, ny_opt, nv_opt).

Typical usage
-------------
    from fft_size_advisor import optimal_fft_size, pad_cube, unpad_cube

    nx_opt, ny_opt, nv_opt = optimal_fft_size(
        nx=577, ny=577, nv=83,
        beam_fwhm_pix=14.0,
        chan_sigma_chan=0.07,
    )

    cube_pad, orig_shape = pad_cube(cube, nx_opt, ny_opt, nv_opt)
    # ... apply FFT convolution on cube_pad ...
    cube_conv = your_fft_convolve(cube_pad)
    cube_result = unpad_cube(cube_conv, orig_shape)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Prime factorisation helpers
# ---------------------------------------------------------------------------

def _prime_factors(n):
    """Return dict {prime: exponent} for integer n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def _factorize_str(n):
    """Human-readable factorisation string."""
    factors = _prime_factors(n)
    parts   = []
    for p in sorted(factors):
        e = factors[p]
        parts.append(f"{p}^{e}" if e > 1 else str(p))
    return " × ".join(parts)


def _is_fftw_smooth(n, good_primes=(2, 3, 5, 7)):
    """
    Return (is_smooth, bad_primes).
    FFTW handles {2,3,5,7} efficiently; larger primes are problematic.
    """
    factors   = _prime_factors(n)
    bad       = {p for p in factors if p not in good_primes}
    return len(bad) == 0, bad


def _fftw_penalty(n):
    """
    Estimated per-element slowdown relative to the nearest power of 2.
    Based on empirical FFTW benchmark data:
      - factor of 2: no penalty
      - factor of 3: ~5% overhead per occurrence
      - factor of 5: ~8% overhead per occurrence
      - factor of 7: ~15% overhead per occurrence
      - prime > 7  : severe — roughly O(p^0.7) times slower per element
    """
    factors = _prime_factors(n)
    penalty = 1.0
    for p, e in factors.items():
        if p == 2:
            penalty *= 1.00 ** e
        elif p == 3:
            penalty *= 1.05 ** e
        elif p == 5:
            penalty *= 1.08 ** e
        elif p == 7:
            penalty *= 1.15 ** e
        else:
            # Large prime: severe degradation
            penalty *= (p ** 0.5) ** e
    return penalty


def _next_power2(n):
    """Smallest power of 2 >= n."""
    return int(2 ** np.ceil(np.log2(max(n, 1))))


def _next_smooth(n, good_primes=(2, 3, 5, 7), max_search=10000):
    """
    Find the smallest integer >= n whose prime factors are all in
    good_primes.  Searches up to n + max_search.
    """
    for k in range(n, n + max_search):
        ok, _ = _is_fftw_smooth(k, good_primes)
        if ok:
            return k
    return _next_power2(n)   # fallback


def _net_fft_cost(n):
    """
    Proxy for absolute FFT cost: n * log2(n) * penalty(n).
    The penalty captures per-element slowdown from non-power-of-2 sizes.
    """
    if n <= 1:
        return 1.0
    return n * np.log2(n) * _fftw_penalty(n)


# ---------------------------------------------------------------------------
# Wrap-around (circular convolution) analysis
# ---------------------------------------------------------------------------

def _wraparound_risk(n, kernel_sigma, nsigma=3.0):
    """
    Estimate the fraction of the cube affected by circular convolution
    wrap-around for a Gaussian kernel of given sigma (in samples).

    Returns fraction in [0, 1].  Above ~0.05 (5%) padding may be needed.
    """
    kernel_radius = nsigma * kernel_sigma
    if kernel_radius <= 0:
        return 0.0
    return min(kernel_radius / (n / 2.0), 1.0)


# ---------------------------------------------------------------------------
# Per-dimension advisor
# ---------------------------------------------------------------------------

def _advise_dimension(n, kernel_sigma, name="dim",
                       wraparound_threshold=0.05,
                       cost_threshold=0.95):
    """
    Analyse one FFT dimension and return a recommendation dict.

    Parameters
    ----------
    n                   : int    original size
    kernel_sigma        : float  Gaussian kernel sigma in samples
    name                : str    label for display
    wraparound_threshold: float  flag if wrap-around fraction > this
    cost_threshold      : float  pad if net_cost(padded)/net_cost(n) < this

    Returns
    -------
    dict with keys: n, n_smooth, n_pow2, pad_recommended, pad_size,
                    reason, is_smooth, wraparound_frac, cost_ratio
    """
    is_smooth, bad_primes = _is_fftw_smooth(n)
    n_smooth = _next_smooth(n)
    n_pow2   = _next_power2(n)

    # Current cost
    cost_n = _net_fft_cost(n)

    # Cost of padding to next smooth number
    cost_smooth = _net_fft_cost(n_smooth)

    # Cost of padding to next power of 2
    cost_pow2   = _net_fft_cost(n_pow2)

    # Wrap-around risk
    wa_frac = _wraparound_risk(n, kernel_sigma)

    # --- Decision logic ---
    pad_recommended = False
    pad_size        = n
    reasons         = []

    # Rule 1: dimension has large prime factors → MUST pad
    if not is_smooth:
        pad_recommended = True
        reasons.append(f"large prime factors {bad_primes} → FFTW O(N²) fallback")

    # Rule 2: padding to next smooth number is cheaper than current size
    if n_smooth != n:
        cost_ratio_smooth = cost_smooth / cost_n
        if cost_ratio_smooth < cost_threshold:
            pad_recommended = True
            reasons.append(
                f"next smooth number {n_smooth} costs "
                f"{cost_ratio_smooth:.2f}× current → net saving")

    # Rule 3: wrap-around is significant AND padding is cost-effective
    if wa_frac > wraparound_threshold:
        # Need to pad to at least n + 2*kernel_radius to make convolution linear
        n_safe   = n + int(2 * 3.0 * kernel_sigma) + 1
        n_safe_s = _next_smooth(n_safe)
        cost_safe = _net_fft_cost(n_safe_s)
        cost_ratio_safe = cost_safe / cost_n
        # Only recommend if: padding is not too expensive AND it is
        # actually cheaper than current (accounts for more elements)
        if cost_ratio_safe < cost_threshold:
            pad_recommended = True
            pad_size        = n_safe_s
            reasons.append(
                f"wrap-around affects {wa_frac*100:.1f}% of half-width "
                f"(>{wraparound_threshold*100:.0f}% threshold) and "
                f"padding costs {cost_ratio_safe:.2f}× — net saving")
        else:
            # Wrap-around exists but padding is too expensive —
            # note it as a warning only, do not pad
            reasons.append(
                f"wrap-around {wa_frac*100:.1f}% noted but padding to "
                f"{n_safe_s} costs {cost_ratio_safe:.2f}× — not worth it")

    # Choose best pad size if padding is recommended
    if pad_recommended and pad_size == n:
        # Pick the cheaper of next_smooth and next_power2
        if cost_smooth <= cost_pow2:
            pad_size = n_smooth
        else:
            pad_size = n_pow2

    # If not recommended, verify current size cost vs padded
    cost_ratio = _net_fft_cost(pad_size) / cost_n if pad_size != n else 1.0

    return {
        "name"            : name,
        "n"               : n,
        "factorization"   : _factorize_str(n),
        "is_smooth"       : is_smooth,
        "bad_primes"      : bad_primes,
        "n_smooth"        : n_smooth,
        "n_pow2"          : n_pow2,
        "cost_n"          : cost_n,
        "cost_smooth"     : cost_smooth,
        "cost_pow2"       : cost_pow2,
        "cost_ratio"      : cost_ratio,
        "wraparound_frac" : wa_frac,
        "pad_recommended" : pad_recommended,
        "pad_size"        : pad_size,
        "reasons"         : reasons if reasons else ["no benefit from padding"],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimal_fft_size(nx, ny, nv,
                     beam_fwhm_pix=0.0,
                     chan_sigma_chan=0.0,
                     wraparound_threshold=0.05,
                     cost_threshold=0.95,
                     verbose=False):
    """
    Return the optimal (nx, ny, nv) for 3D FFT convolution.

    Analyses each dimension independently and pads only where beneficial.

    Parameters
    ----------
    nx, ny          : int    spatial cube dimensions [pixels]
    nv              : int    spectral cube dimension [channels]
    beam_fwhm_pix   : float  beam FWHM [pixels]  (0 = no spatial kernel)
    chan_sigma_chan  : float  spectral kernel sigma [channels]
                             (= chan_width_kms / dv_kms)
    wraparound_threshold : float
        Flag wrap-around if it affects more than this fraction of the
        half-width.  Default 0.05 (5%).
    cost_threshold  : float
        Pad if net FFT cost of padded size < cost_threshold × current cost.
        Default 0.95 (pad only if at least 5% cheaper net).
    verbose         : bool   print full analysis

    Returns
    -------
    nx_opt, ny_opt, nv_opt : int
        Recommended FFT sizes (>= original, or equal if no padding needed).
    """
    beam_sigma  = beam_fwhm_pix  / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    r_x = _advise_dimension(nx, beam_sigma,       "nx (spatial x)",
                             wraparound_threshold, cost_threshold)
    r_y = _advise_dimension(ny, beam_sigma,       "ny (spatial y)",
                             wraparound_threshold, cost_threshold)
    r_v = _advise_dimension(nv, chan_sigma_chan,   "nv (spectral)",
                             wraparound_threshold, cost_threshold)

    if verbose:
        _print_analysis(nx, ny, nv, beam_fwhm_pix, chan_sigma_chan,
                        r_x, r_y, r_v)

    return r_x["pad_size"], r_y["pad_size"], r_v["pad_size"]


def fft_size_advisor(nx, ny, nv,
                     beam_fwhm_pix=0.0,
                     chan_sigma_chan=0.0,
                     wraparound_threshold=0.05,
                     cost_threshold=0.95):
    """
    Full interactive analysis and recommendation.

    Prints a detailed report and returns (nx_opt, ny_opt, nv_opt).

    Parameters
    ----------
    nx, ny          : int    spatial dimensions
    nv              : int    spectral dimension
    beam_fwhm_pix   : float  beam FWHM [pixels]
    chan_sigma_chan  : float  spectral kernel sigma [channels]
    wraparound_threshold : float  (default 0.05)
    cost_threshold       : float  (default 0.95)

    Returns
    -------
    nx_opt, ny_opt, nv_opt : int
    """
    nx_opt, ny_opt, nv_opt = optimal_fft_size(
        nx, ny, nv,
        beam_fwhm_pix        = beam_fwhm_pix,
        chan_sigma_chan       = chan_sigma_chan,
        wraparound_threshold = wraparound_threshold,
        cost_threshold       = cost_threshold,
        verbose              = True,
    )
    return nx_opt, ny_opt, nv_opt


# ---------------------------------------------------------------------------
# Padding / unpadding
# ---------------------------------------------------------------------------

def pad_cube(cube, nx_opt, ny_opt, nv_opt):
    """
    Zero-pad a datacube to the optimal FFT sizes.

    Padding is applied at the **end** of each axis (trailing zeros) so
    that the original data is always at the beginning of each dimension
    and the unpadding step is a simple leading slice.

    The cube shape convention throughout barolo.py is (nv, ny, nx).

    Parameters
    ----------
    cube   : np.ndarray, shape (nv, ny, nx)
        The original datacube.
    nx_opt : int   optimal x size  (from optimal_fft_size)
    ny_opt : int   optimal y size
    nv_opt : int   optimal v size

    Returns
    -------
    padded : np.ndarray, shape (nv_opt, ny_opt, nx_opt)
        Zero-padded cube.  If no padding is needed along any axis the
        original array is returned unchanged (no copy).
    original_shape : tuple  (nv, ny, nx)
        The original shape, needed by unpad_cube to trim back.

    Examples
    --------
    >>> nx_opt, ny_opt, nv_opt = optimal_fft_size(577, 577, 83,
    ...                                            beam_fwhm_pix=14.0)
    >>> cube_pad, orig = pad_cube(cube, nx_opt, ny_opt, nv_opt)
    >>> # ... FFT convolution on cube_pad ...
    >>> cube_result = unpad_cube(cube_conv, orig)
    """
    nv, ny, nx = cube.shape
    original_shape = (nv, ny, nx)

    # Check whether any padding is actually needed
    if nx_opt == nx and ny_opt == ny and nv_opt == nv:
        return cube, original_shape   # no-op — return original array

    # Build padding widths: (before, after) per axis
    # Convention: pad only at the end (after) — data stays at index [0:n]
    pad_width = [
        (0, nv_opt - nv),   # spectral axis
        (0, ny_opt - ny),   # spatial y
        (0, nx_opt - nx),   # spatial x
    ]

    if any(p[1] < 0 for p in pad_width):
        raise ValueError(
            f"Optimal sizes ({nv_opt},{ny_opt},{nx_opt}) are smaller than "
            f"original cube ({nv},{ny},{nx}). This should not happen — "
            f"check optimal_fft_size output.")

    padded = np.pad(cube, pad_width, mode='constant', constant_values=0.0)
    return padded, original_shape


def unpad_cube(padded_cube, original_shape):
    """
    Remove zero-padding from a convolved cube, restoring the original shape.

    Since pad_cube places padding at the end of each axis, unpadding is
    simply a leading slice along each dimension.

    Parameters
    ----------
    padded_cube    : np.ndarray, shape (nv_opt, ny_opt, nx_opt)
        The convolved padded cube (output of FFT convolution).
    original_shape : tuple  (nv, ny, nx)
        The original shape returned by pad_cube.

    Returns
    -------
    cube : np.ndarray, shape (nv, ny, nx)
        The convolved cube trimmed back to the original spatial and
        spectral extent.

    Examples
    --------
    >>> cube_result = unpad_cube(cube_conv, orig_shape)
    """
    nv, ny, nx = original_shape
    return padded_cube[:nv, :ny, :nx]


def _print_analysis(nx, ny, nv, beam_fwhm_pix, chan_sigma_chan,
                    r_x, r_y, r_v):
    """Print the full analysis report."""
    beam_sigma = beam_fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    print("=" * 68)
    print("  FFT Size Advisor — 3D Datacube Convolution")
    print("=" * 68)
    print(f"\n  Input cube    : {nx} × {ny} × {nv}")
    print(f"  Beam FWHM     : {beam_fwhm_pix:.2f} pixels  "
          f"(sigma = {beam_sigma:.2f} px)")
    print(f"  Spectral sigma: {chan_sigma_chan:.3f} channels")

    print("\n" + "─" * 68)
    print("  Per-dimension analysis")
    print("─" * 68)

    for r in [r_x, r_y, r_v]:
        print(f"\n  {r['name']} = {r['n']}")
        print(f"    Factorisation    : {r['factorization']}")
        print(f"    FFTW-smooth      : {'YES' if r['is_smooth'] else 'NO  ← large primes: '+str(r['bad_primes'])}")
        print(f"    Next smooth      : {r['n_smooth']}  "
              f"(net cost × {r['cost_smooth']/r['cost_n']:.3f})")
        print(f"    Next power-of-2  : {r['n_pow2']}  "
              f"(net cost × {r['cost_pow2']/r['cost_n']:.3f})")
        print(f"    Wrap-around risk : {r['wraparound_frac']*100:.1f}%  "
              f"({'OK' if r['wraparound_frac'] <= 0.05 else 'HIGH — may need padding'})")
        verdict = "PAD →" if r['pad_recommended'] else "KEEP "
        print(f"    Decision         : {verdict} {r['pad_size']}  "
              f"({'  '.join(r['reasons'])})")

    # Memory comparison
    print("\n" + "─" * 68)
    print("  Memory impact (float64 cube + complex FFT arrays)")
    print("─" * 68)

    nx_opt = r_x["pad_size"]
    ny_opt = r_y["pad_size"]
    nv_opt = r_v["pad_size"]

    def mem_mb(nx_, ny_, nv_):
        cube_mb = nx_ * ny_ * nv_ * 8 / 1e6
        fft_mb  = nx_ * ny_ * (nv_//2 + 1) * 16 / 1e6
        return cube_mb, fft_mb, 2*cube_mb + 2*fft_mb

    c0, f0, t0 = mem_mb(nx,     ny,     nv)
    c1, f1, t1 = mem_mb(nx_opt, ny_opt, nv_opt)

    print(f"\n  Original  {nx}×{ny}×{nv}:")
    print(f"    Cube        : {c0:.1f} MB")
    print(f"    FFT arrays  : {f0:.1f} MB  (complex128, rfftn)")
    print(f"    Total (×4)  : {t0:.1f} MB")

    if (nx_opt, ny_opt, nv_opt) != (nx, ny, nv):
        print(f"\n  Padded    {nx_opt}×{ny_opt}×{nv_opt}:")
        print(f"    Cube        : {c1:.1f} MB")
        print(f"    FFT arrays  : {f1:.1f} MB")
        print(f"    Total (×4)  : {t1:.1f} MB")
        print(f"    Memory overhead: +{(t1/t0-1)*100:.0f}%")

    # 3D net cost comparison
    print("\n" + "─" * 68)
    print("  3D FFT net cost comparison")
    print("─" * 68)

    def cost3d(nx_, ny_, nv_):
        rfft_n = nx_ * ny_ * (nv_//2 + 1)
        pen    = (_fftw_penalty(nx_) *
                  _fftw_penalty(ny_) *
                  _fftw_penalty(nv_//2 + 1))
        return rfft_n * np.log2(rfft_n) * pen

    c3d_orig   = cost3d(nx,     ny,     nv)
    c3d_padded = cost3d(nx_opt, ny_opt, nv_opt)
    ratio3d    = c3d_padded / c3d_orig

    print(f"\n  Original  {nx}×{ny}×{nv}   : relative cost = 1.000")
    print(f"  Padded    {nx_opt}×{ny_opt}×{nv_opt} : relative cost = {ratio3d:.3f}")

    if ratio3d < 1.0:
        print(f"  → Padding saves {(1-ratio3d)*100:.1f}% of FFT cost  ✓")
    else:
        print(f"  → Padding costs {(ratio3d-1)*100:.1f}% MORE  ✗")

    # Final verdict
    print("\n" + "=" * 68)
    print("  VERDICT")
    print("=" * 68)

    any_pad = (nx_opt, ny_opt, nv_opt) != (nx, ny, nv)

    if not any_pad:
        print(f"""
  DO NOT PAD.

  All dimensions are already FFTW-smooth and padding to the
  next power of 2 would increase cost by {(ratio3d-1)*100:.0f}%.
  Use the cube exactly as-is: {nx}×{ny}×{nv}
""")
    else:
        dims_changed = []
        if nx_opt != nx:
            dims_changed.append(f"nx: {nx} → {nx_opt}")
        if ny_opt != ny:
            dims_changed.append(f"ny: {ny} → {ny_opt}")
        if nv_opt != nv:
            dims_changed.append(f"nv: {nv} → {nv_opt}")

        print(f"""
  PAD the following dimensions:
    {('  ' + chr(10)).join(dims_changed) if dims_changed else '(none)'}

  Optimal padded size: {nx_opt}×{ny_opt}×{nv_opt}
  Net FFT cost change: {ratio3d:.3f}×  {'(saving)' if ratio3d<1 else '(overhead)'}
  Memory overhead    : +{(t1/t0-1)*100:.0f}%

  Recommended CubeConfig padding:
    pad = [(0, 0),
           (0, {ny_opt-ny}),
           (0, {nx_opt-nx})]
    cube_padded = np.pad(cube, pad, mode='constant', constant_values=0)
    # Trim back after convolution:
    cube_conv = cube_padded[:{nv}, :{ny}, :{nx}]
""")

    print(f"  Recommended call:")
    print(f"    nx_opt, ny_opt, nv_opt = optimal_fft_size(")
    print(f"        nx={nx}, ny={ny}, nv={nv},")
    print(f"        beam_fwhm_pix={beam_fwhm_pix},")
    print(f"        chan_sigma_chan={chan_sigma_chan})")
    print(f"    # Returns: ({nx_opt}, {ny_opt}, {nv_opt})")
    print()


# ---------------------------------------------------------------------------
# Example / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.ndimage import gaussian_filter

    print("\n" + "="*68)
    print("  Example 1: Your ALMA cube (576×576×81) — no padding needed")
    print("="*68)
    nx_opt, ny_opt, nv_opt = fft_size_advisor(
        nx=576, ny=576, nv=81,
        beam_fwhm_pix=14.0,
        chan_sigma_chan=2.0/30.0,
    )
    # Demonstrate pad/unpad (no-op case)
    cube_alma = np.zeros((81, 576, 576))
    cube_pad, orig = pad_cube(cube_alma, nx_opt, ny_opt, nv_opt)
    cube_result    = unpad_cube(cube_pad, orig)
    assert cube_result.shape == cube_alma.shape
    print(f"  pad_cube:   {cube_alma.shape} → {cube_pad.shape}  (no-op)")
    print(f"  unpad_cube: {cube_pad.shape} → {cube_result.shape}")

    print("\n" + "="*68)
    print("  Example 2: Prime dimensions (577×577×83) — padding required")
    print("="*68)
    nx_opt, ny_opt, nv_opt = fft_size_advisor(
        nx=577, ny=577, nv=83,
        beam_fwhm_pix=14.0,
        chan_sigma_chan=0.07,
    )
    # Demonstrate full pad → convolve → unpad workflow
    rng      = np.random.default_rng(0)
    cube_bad = rng.standard_normal((83, 577, 577)).astype(np.float64)

    # Step 1: pad
    cube_pad, orig_shape = pad_cube(cube_bad, nx_opt, ny_opt, nv_opt)
    print(f"\n  Step 1 — pad_cube:")
    print(f"    Original : {cube_bad.shape}  ({cube_bad.nbytes/1e6:.1f} MB)")
    print(f"    Padded   : {cube_pad.shape}  ({cube_pad.nbytes/1e6:.1f} MB)")

    # Step 2: convolve on padded cube (using scipy as stand-in for pyfftw)
    sigma_spatial  = 14.0 / (2 * np.sqrt(2 * np.log(2)))
    sigma_spectral = 0.07
    cube_conv = gaussian_filter(cube_pad,
                                sigma=[sigma_spectral, sigma_spatial,
                                       sigma_spatial])
    print(f"  Step 2 — convolve (on padded cube): {cube_conv.shape}")

    # Step 3: unpad — trim back to original shape
    cube_result = unpad_cube(cube_conv, orig_shape)
    print(f"  Step 3 — unpad_cube: {cube_conv.shape} → {cube_result.shape}")
    assert cube_result.shape == cube_bad.shape, \
        f"Shape mismatch: {cube_result.shape} vs {cube_bad.shape}"
    print(f"  OK: output shape matches input shape exactly")

    # Verify padding does not alter the data region
    assert np.allclose(cube_pad[:83, :577, :577], cube_bad), \
        "Padding altered original data region!"
    print(f"  OK: original data region unaltered by padding")

    print("\n" + "="*68)
    print("  Example 3: MUSE-like cube (300×300×3700)")
    print("="*68)
    nx_opt, ny_opt, nv_opt = fft_size_advisor(
        nx=300, ny=300, nv=3700,
        beam_fwhm_pix=10.0,
        chan_sigma_chan=0.85,
    )
    cube_muse = np.zeros((3700, 300, 300))
    cube_pad, orig = pad_cube(cube_muse, nx_opt, ny_opt, nv_opt)
    cube_res       = unpad_cube(cube_pad, orig)
    assert cube_res.shape == cube_muse.shape
    print(f"  pad_cube:   {cube_muse.shape} → {cube_pad.shape}")
    print(f"  unpad_cube: {cube_pad.shape} → {cube_res.shape}")

    print("\n" + "="*68)
    print("  Example 4: VLA HI (512×512×60) — no padding needed")
    print("="*68)
    nx_opt, ny_opt, nv_opt = fft_size_advisor(
        nx=512, ny=512, nv=60,
        beam_fwhm_pix=8.0,
        chan_sigma_chan=0.3,
    )
    cube_vla = np.zeros((60, 512, 512))
    cube_pad, orig = pad_cube(cube_vla, nx_opt, ny_opt, nv_opt)
    cube_res       = unpad_cube(cube_pad, orig)
    assert cube_res.shape == cube_vla.shape
    print(f"  pad_cube:   {cube_vla.shape} → {cube_pad.shape}  (no-op)")

    print("\n" + "="*68)
    print("  Example 5: Awkward dimensions (1000×1000×100)")
    print("="*68)
    nx_opt, ny_opt, nv_opt = fft_size_advisor(
        nx=1000, ny=1000, nv=100,
        beam_fwhm_pix=6.0,
        chan_sigma_chan=0.5,
    )
    cube_awk = np.zeros((100, 1000, 1000))
    cube_pad, orig = pad_cube(cube_awk, nx_opt, ny_opt, nv_opt)
    cube_res       = unpad_cube(cube_pad, orig)
    assert cube_res.shape == cube_awk.shape
    print(f"  pad_cube:   {cube_awk.shape} → {cube_pad.shape}  (no-op)")

    print("\n  All examples completed successfully.")

