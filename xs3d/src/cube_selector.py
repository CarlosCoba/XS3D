"""
cube_selector.py — Interactive cube region + channel selector.

Left  panel : collapsed cube (moment-0 map) — drag to define a spatial ROI.
Right panel : integrated spectrum — drag to add spectral channel ranges.

The two panels are linked:
  • Selecting a spatial ROI   spectrum updates to show only that region.
  • Selecting channel ranges  moment-0 updates to show only those channels.

When you click "Done" (or close the window) the selections are printed in a
format ready to paste into a XS3D parameter file.

Usage
-----
  # standalone demo (synthetic cube)
  python cube_selector.py

  # from a FITS file
  python cube_selector.py galaxy.fits

  # as a library
  from cube_selector import run_selector
  results = run_selector("galaxy.fits")		  # from FITS
  results = run_selector(cube=arr, wl=wl_arr)	# from numpy array

  # results dict
  {
	'roi'			: (x1, x2, y1, y2),   # pixel, 0-indexed, or None
	'channel_ranges' : [(ch1, ch2), ...],  # 0-indexed, each inclusive
	'all_channels'   : [ch, ...],		  # sorted flat list
  }
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector, SpanSelector, Button
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits as _fits

# For mac Users active this line:
#matplotlib.use('MacOSX')   # or 'TkAgg' or 'Qt5Agg' — any interactive backend

from .colormaps_CLC import vel_map
cmap = vel_map()
cmap_mom2 = vel_map('mom2')
cmap_mom0 = vel_map('mom0')

# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _pclim(arr, lo=0.5, hi=99.5):
	"""Robust percentile color limits ignoring NaN/Inf."""
	#v = arr[np.isfinite(arr)]
	v = arr[(np.isfinite(arr)) & (arr > 0)]	
	return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class CubeSelector:
	"""
	Interactive spatial-ROI + spectral-range selector for a 3-D IFU cube.

	Parameters
	----------
	cube	: (nz, ny, nx) float array
	wl	  : (nz,) wavelength array; channel indices used if None
	wl_unit : label for the spectral axis
	header  : optional FITS header (for display only)
	"""

	# ── Colour palette (dark astronomy theme) ────────────────────────────────
	_BG			= '#16181c'
	_PANEL_BG	= '#0f1014'
	_FG			= '#dde0e8'
	_MUTED		= '#7a7f90'
	_GRID		= '#25282f'
	_ROI_COL	= '#ff5555'	  # red rectangle / ROI info
	_CH_COL		= '#4eb8f0'	  # blue spans   / channel info
	_SPEC_FULL	= '#4a4f5e'	  # grey — full-cube spectrum
	_BTN_BG		= '#21242c'
	_BTN_DONE	= '#1b3d24'
	_BTN_DONE_FG= '#7af59a'

	def __init__(self, cube, wl=None, wl_unit='CUNIT3', header=None):
		if cube.ndim != 3:
			raise ValueError(f"cube must be 3-D, got {cube.shape}")
		self.cube	= np.asarray(cube, dtype=float)
		self.nz, self.ny, self.nx = self.cube.shape
		self.wl	  = np.asarray(wl, dtype=float) if wl is not None \
					   else np.arange(self.nz, dtype=float)
		self.wl_unit = wl_unit
		self.header  = header

		# ── stored selections ──────────────────────────────────────────────
		self.roi			 = None   # (x1, x2, y1, y2) pixel 0-indexed
		self.channel_ranges  = []	 # [(ch1,ch2), …] 0-indexed inclusive
		self._span_artists   = []	 # persistent axvspan patches

		# ── full-cube collapsed arrays ─────────────────────────────────────
		self._mom0_full = np.nansum(self.cube, axis=0)
		self._spec_full = np.nansum(
			self.cube.reshape(self.nz, -1), axis=1)

		self._build_figure()

	# ── Figure ───────────────────────────────────────────────────────────────

	def _build_figure(self):
		#plt.rcParams['font.size'] = 10
		#plt.rcParams['font.family'] = 'monospace'

		self.fig = plt.figure(figsize=(15, 6.8),
							  facecolor=self._BG)
		self.fig.suptitle(
			'Cube Selector',
			color=self._FG, fontsize=12, fontweight='bold', y=0.97,
			fontfamily='sans-serif',
		)

		# Layout: [moment0 | spectrum]  above button row
		gs = self.fig.add_gridspec(
			1, 2,
			width_ratios=[1, 1.55],
			left=0.07, right=0.97,
			top=0.89, bottom=0.16,
			wspace=0.38,
		)
		self.ax_m0   = self.fig.add_subplot(gs[0, 0])
		self.ax_spec = self.fig.add_subplot(gs[0, 1])

		for ax in (self.ax_m0, self.ax_spec):
			ax.set_facecolor(self._PANEL_BG)
			ax.tick_params(colors=self._MUTED, labelsize=8)
			for sp in ax.spines.values():
				sp.set_edgecolor(self._GRID)
			ax.xaxis.label.set_color(self._MUTED)
			ax.yaxis.label.set_color(self._MUTED)
			ax.xaxis.set_minor_locator(AutoMinorLocator())
			ax.yaxis.set_minor_locator(AutoMinorLocator())
			ax.grid(True, color=self._GRID, linewidth=0.4, zorder=0)

		self._init_mom0_panel()
		self._init_spec_panel()
		self._init_selectors()
		self._init_buttons()

		#self.fig.canvas.mpl_connect('close_event', lambda e: self._print_results())
		self.fig.canvas.mpl_connect('close_event', print('  Interactive window'))		
		self._refresh_titles()

	# ── Moment-0 panel ───────────────────────────────────────────────────────

	def _init_mom0_panel(self):
		ax = self.ax_m0
		vmin, vmax = _pclim(self._mom0_full)
		self._im = ax.imshow(
			self._mom0_full, origin='lower',
			cmap=cmap_mom0, vmin=vmin, vmax=vmax,
			aspect='auto', interpolation='nearest',
			zorder=1,
		)
		ax.set_xlabel('x  [pixel]')
		ax.set_ylabel('y  [pixel]')

		cbar = self.fig.colorbar(
			self._im, ax=ax, pad=0.02, fraction=0.046)
		cbar.ax.tick_params(colors=self._MUTED, labelsize=7)
		cbar.set_label('Int. flux', color=self._MUTED, fontsize=8)
		cbar.outline.set_edgecolor(self._GRID)

		# Persistent ROI rectangle (drawn after RectangleSelector releases)
		self._roi_patch = mpatches.Rectangle(
			(0, 0), 0, 0,
			linewidth=1.8, edgecolor=self._ROI_COL,
			facecolor=self._ROI_COL + '18',
			linestyle='--', zorder=6,
		)
		ax.add_patch(self._roi_patch)

		# ROI centre cross-hair (initially invisible)
		self._roi_cx, = ax.plot([], [], '+',
								color=self._ROI_COL, ms=10, mew=1.5,
								zorder=7)

	# ── Spectrum panel ───────────────────────────────────────────────────────

	def _init_spec_panel(self):
		ax = self.ax_spec
		ax.set_xlabel(f'Spectral axis  [{self.wl_unit}]')
		ax.set_ylabel('Integrated flux  [arb.]')
		ax.axhline(0, color=self._MUTED, lw=0.5, ls=':', zorder=2)

		# Full-cube spectrum (gray reference)
		self._spec_full_line, = ax.plot(
			self.wl, self._spec_full,
			color=self._SPEC_FULL, lw=2,
			label='All pixels', zorder=3
		)
		# ROI spectrum (bright, drawn on top)
		self._spec_roi_line, = ax.plot(
			self.wl, self._spec_full,
			color=self._ROI_COL, lw=1.5,
			label='ROI pixels', zorder=4,
		)
		ax.legend(
			fontsize=11, framealpha=0.3, ncol=2,
			labelcolor=self._FG, facecolor='#1c1f28',
			edgecolor=self._GRID, loc='upper right',
		)
		# Pin x-range to the wavelength array — never let autoscale touch it.
		ax.set_xlim(self.wl[0], self.wl[-1])

	# ── Interactive selectors ────────────────────────────────────────────────

	def _init_selectors(self):
		# Rectangle selector on moment-0
		rect_kw = dict(
			useblit=True, button=[1],
			minspanx=2, minspany=2, spancoords='data',
			interactive=True,
			props=dict(
				facecolor=self._ROI_COL + '25',
				edgecolor=self._ROI_COL,
				alpha=1.0, fill=True,
				linewidth=1.5, linestyle='--',
			),
			handle_props=dict(
				markerfacecolor=self._ROI_COL,
				markeredgecolor=self._ROI_COL,
			),
		)
		try:
			self._rect_sel = RectangleSelector(
				self.ax_m0, self._cb_rect, **rect_kw)
		except TypeError:
			# matplotlib < 3.7 — no handle_props
			rect_kw.pop('handle_props', None)
			self._rect_sel = RectangleSelector(
				self.ax_m0, self._cb_rect, **rect_kw)

		# Span selector on spectrum (each drag ADDS a range)
		span_kw = dict(
			direction='horizontal',
			useblit=True,
			props=dict(alpha=0.30, facecolor=self._CH_COL),
			interactive=False,
			drag_from_anywhere=False,
		)
		try:
			self._span_sel = SpanSelector(
				self.ax_spec, self._cb_span, **span_kw)
		except TypeError:
			span_kw.pop('drag_from_anywhere', None)
			self._span_sel = SpanSelector(
				self.ax_spec, self._cb_span, **span_kw)

	# ── Buttons ──────────────────────────────────────────────────────────────

	def _init_buttons(self):
		def _ax(left, w=0.13, h=0.055):
			return self.fig.add_axes([left, 0.03, w, h],
									  facecolor=self._BTN_BG)

		def _btn(ax, label, cb, fg=None, bg=None):
			b = Button(ax, label,
					   color=bg or self._BTN_BG,
					   hovercolor='#2e323e')
			b.label.set_color(fg or self._FG)
			b.label.set_fontsize(12)
			b.on_clicked(cb)
			for sp in ax.spines.values():
				sp.set_edgecolor(self._GRID)
			return b

		self._btn_clear_roi = _btn(_ax(0.08),  'Clear ROI',	  self._clear_roi)
		self._btn_clear_ch  = _btn(_ax(0.23),  'Clear Channels', self._clear_channels)
		self._btn_clear_all = _btn(_ax(0.38),  'Clear All',	  self._clear_all)
		self._btn_print		= _btn(_ax(0.60),  'Print Results', lambda e: self._print_results())
		self._btn_done		= _btn(_ax(0.76),  'Done ',		self._done, fg=self._BTN_DONE_FG, bg=self._BTN_DONE)

		# Instruction text
		self.fig.text(
			0.5, 0.005,
			'Left: drag to set ROI (updates spectrum)  ·  '
			'Right: drag to add channel range (updates moment-0)  ·  '
			'Multiple channel ranges allowed',
			ha='center', fontsize=9, color=self._MUTED,
			transform=self.fig.transFigure,
		)

	# ── Callbacks ────────────────────────────────────────────────────────────

	def _cb_rect(self, eclick, erelease):
		"""Called each time the ROI rectangle is drawn / resized."""
		x1, x2 = sorted([eclick.xdata, erelease.xdata])
		y1, y2 = sorted([eclick.ydata, erelease.ydata])
		# Clamp to cube bounds
		x1 = max(0, int(round(x1)));  x2 = min(self.nx-1, int(round(x2)))
		y1 = max(0, int(round(y1)));  y2 = min(self.ny-1, int(round(y2)))
		if x2 <= x1 or y2 <= y1:
			return
		self.roi = (x1, x2, y1, y2)
		# Update persistent ROI patch and cross-hair
		self._roi_patch.set_bounds(x1, y1, x2-x1, y2-y1)
		cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
		self._roi_cx.set_data([cx], [cy])
		# Recompute spectrum from ROI region
		self._refresh_spec()
		self._refresh_titles()

	def _cb_span(self, wmin, wmax):
		"""Called each time a span is drawn; accumulates ranges."""
		ch1 = int(np.argmin(np.abs(self.wl - wmin)))
		ch2 = int(np.argmin(np.abs(self.wl - wmax)))
		ch1, ch2 = min(ch1, ch2), max(ch1, ch2)
		if ch2 - ch1 < 1:
			return
		self.channel_ranges.append((ch1, ch2))
		# Persistent shading for this range
		art = self.ax_spec.axvspan(
			self.wl[ch1], self.wl[ch2],
			alpha=0.22, color=self._CH_COL, zorder=2,
		)
		# Add text label above the span
		mid_wl = (self.wl[ch1] + self.wl[ch2]) / 2.0
		ypos = self.ax_spec.get_ylim()[1] * 0.92
		lbl = self.ax_spec.text(
			mid_wl, ypos,
			f'ch {ch1}–{ch2}',
			color=self._CH_COL, fontsize=7,
			ha='center', va='top', zorder=8,
		)
		self._span_artists.extend([art, lbl])
		# Update moment-0 from all selected channels
		self._refresh_mom0()
		self._refresh_titles()
		self.fig.canvas.draw_idle()

	# ── Panel refreshes ──────────────────────────────────────────────────────

	def _refresh_spec(self):
		"""Recompute and redraw the ROI-spectrum line."""
		if self.roi is not None:
			x1, x2, y1, y2 = self.roi
			sub = self.cube[:, y1:y2+1, x1:x2+1]
			spec = np.nansum(sub.reshape(self.nz, -1), axis=1)
		else:
			spec = self._spec_full
		self._spec_roi_line.set_ydata(spec)
		self.ax_spec.relim()
		self.ax_spec.autoscale_view(scalex=False)   # y only; x is pinned to wl
		self.fig.canvas.draw_idle()

	def _refresh_mom0(self):
		"""Recompute and redraw moment-0 from selected channels."""
		if self.channel_ranges:
			mask = np.zeros(self.nz, dtype=bool)
			for ch1, ch2 in self.channel_ranges:
				mask[ch1:ch2+1] = True
			arr = np.nansum(self.cube[mask], axis=0)
		else:
			arr = self._mom0_full
		vmin, vmax = _pclim(arr)
		self._im.set_data(arr)
		self._im.set_clim(vmin, vmax)
		self.fig.canvas.draw_idle()

	def _refresh_titles(self):
		"""Update panel titles to reflect current selections."""
		if self.roi:
			x1, x2, y1, y2 = self.roi
			roi_str = (f'ROI  x[{x1}:{x2}]  y[{y1}:{y2}]'
					   f'  ctr({(x1+x2)/2:.0f},{(y1+y2)/2:.0f})')
		else:
			roi_str = 'drag to set ROI'
		self.ax_m0.set_title(
			f'Moment 0  —  {roi_str}',
			color=self._ROI_COL if self.roi else self._MUTED,
			fontsize=8.5, pad=6,
		)

		if self.channel_ranges:
			n_ch = sum(c2-c1+1 for c1,c2 in self.channel_ranges)
			ch_str = (f'{len(self.channel_ranges)} range(s)  '
					  f'·  {n_ch} channels selected')
		else:
			ch_str = 'drag to add channel range'
		self.ax_spec.set_title(
			f'Spectrum  —  {ch_str}',
			color=self._CH_COL if self.channel_ranges else self._MUTED,
			fontsize=8.5, pad=6,
		)
		self.fig.canvas.draw_idle()

	# ── Button callbacks ─────────────────────────────────────────────────────

	def _clear_roi(self, _=None):
		self.roi = None
		self._roi_patch.set_bounds(0, 0, 0, 0)
		self._roi_cx.set_data([], [])
		# Deactivate + reactivate to reset the widget's internal state
		self._rect_sel.set_active(False)
		self._rect_sel.set_active(True)
		self._refresh_spec()
		self._refresh_titles()

	def _clear_channels(self, _=None):
		self.channel_ranges.clear()
		for a in self._span_artists:
			try:
				a.remove()
			except Exception:
				pass
		self._span_artists.clear()
		self._refresh_mom0()
		self._refresh_titles()
		self.fig.canvas.draw_idle()

	def _clear_all(self, _=None):
		self._clear_roi()
		self._clear_channels()

	def _done(self, _=None):
		self._print_results()
		plt.close(self.fig)

	# ── Output ───────────────────────────────────────────────────────────────

	def _print_results(self):
	
		self.n=42 
		self.deli='  '+"-"*(self.n-2)
		print(self.deli)
		print('  CUBE SELECTOR — RESULTS')
		print(self.deli)

		# ── spatial ──
		print()
		if self.roi:
			x1, x2, y1, y2 = self.roi
			cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
			print('  Spatial ROI  (pixel, 0-indexed):')
			print(f'	x  :  {x1}    {x2}   ({x2-x1} px wide)')
			print(f'	y  :  {y1}    {y2}   ({y2-y1} px tall)')
			print(f'	centre  :  ({cx:.1f}, {cy:.1f})')
			print()
			print('   XS3D  (1-indexed FITS pixel centre):')
			print(f'	XPOS  {cx+1:.2f}')
			print(f'	YPOS  {cy+1:.2f}')
		else:
			print('  Spatial ROI : not set')

		# ── spectral ──
		print()
		if self.channel_ranges:
			print('  Spectral channel ranges  (0-indexed, inclusive):')
			for i, (ch1, ch2) in enumerate(self.channel_ranges):
				wl1, wl2 = self.wl[ch1], self.wl[ch2]
				nchan = ch2 - ch1 + 1
				print(f'	range {i+1}: channels [{ch1}, {ch2}]  '
					  f'  window = {wl1:.2f} – {wl2:.2f} {self.wl_unit}'
					  f'  ({nchan} channels)')

			all_ch = sorted({ch
							 for r in self.channel_ranges
							 for ch in range(r[0], r[1]+1)})
			print()
			print(f'  Flat channel list  (0-indexed, {len(all_ch)} total):')
			#print(f'	{all_ch}')
			print()
			print('   XS3D  WRANGE  (1-indexed):')
			for ch1, ch2 in self.channel_ranges:
				wl1, wl2 = self.wl[ch1], self.wl[ch2]
				print(f'	WRANGE  {ch1+1}  {ch2+1}'
					  f'   # {wl1:.2f}–{wl2:.2f} {self.wl_unit}')
		else:
			print('  Channel ranges : not set')

		#print()
		#print('  Python variables:')
		#print(f'	roi			= {self.roi}')
		#print(f'	channel_ranges = {self.channel_ranges}')
		#print(f'{bar}\n')

	def get_results(self):
		all_ch = sorted({ch
						 for r in self.channel_ranges
						 for ch in range(r[0], r[1]+1)})
		return {
			'roi'			: self.roi,
			'channel_ranges' : self.channel_ranges,
			'all_channels'   : all_ch,
		}


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_cube(path, hdu_index=0):
	"""Load a 3-D FITS cube.  Returns (cube, wl, wl_unit, header)."""
	with _fits.open(path) as hdul:
		hdu  = hdul[hdu_index]
		data = np.asarray(hdu.data, dtype=float)
		hdr  = hdu.header
	if data.ndim == 4:			# drop Stokes axis
		data = data[0]
	if data.ndim != 3:
		raise ValueError(f'Expected 3-D cube, got shape {data.shape}')
	nz = data.shape[0]
	try:
		crpix   = hdr.get('CRPIX3', 1.0)
		crval   = hdr.get('CRVAL3', 0.0)
		cdelt   = hdr.get('CD3_3', hdr.get('CDELT3', 1.0))
		wl	  = crval + (np.arange(nz) - (crpix - 1)) * cdelt
		wl_unit = hdr.get('CUNIT3', 'channel')
	except Exception:
		wl, wl_unit = np.arange(nz, dtype=float), 'channel'
	return data, wl, wl_unit, hdr


def run_selector(fits_path=None, cube=None, wl=None, wl_unit='Å',
				 hdu_index=0):
	"""
	Launch the interactive selector and return the results dict when the
	window is closed.

	Parameters
	----------
	fits_path : str, optional	  path to a 3-D FITS cube
	cube	  : ndarray, optional  (nz, ny, nx) float array
	wl		: array-like		 wavelengths; channel indices if None
	wl_unit   : str				spectral axis label
	hdu_index : int				FITS HDU index

	Returns
	-------
	dict  {'roi', 'channel_ranges', 'all_channels'}
	"""
	if fits_path is not None:
		cube, wl, wl_unit, header = load_cube(fits_path, hdu_index)
	elif cube is None:
		raise ValueError('Provide either fits_path or cube.')
	else:
		header = None
	if wl is None:
		wl = np.arange(cube.shape[0], dtype=float)

	sel = CubeSelector(cube, wl=wl, wl_unit=wl_unit, header=header)
	plt.show(block=True)
	return sel.get_results()


# ─────────────────────────────────────────────────────────────────────────────
#  Demo / CLI
# ─────────────────────────────────────────────────────────────────────────────

def _demo_cube():
	"""
	Synthetic MUSE-like cube:
	  • Extended galaxy disk centred at ~(38,36)
	  • Compact off-centre clump at ~(55,52)
	  • Emission line near λ 6625 Å  (channels ~100-130)
	  • Low-level continuum + noise
	"""
	rng = np.random.default_rng(42)
	nz, ny, nx = 220, 80, 80
	YY, XX = np.mgrid[0:ny, 0:nx]

	def gauss2d(x0, y0, sig):
		return np.exp(-((XX-x0)**2 + (YY-y0)**2) / (2*sig**2))

	def gauss1d(mu, sig):
		return np.exp(-((np.arange(nz)-mu)**2) / (2*sig**2))

	# Main galaxy (broad, tilted ellipse approximated by two Gaussians)
	s_main = 0.7*gauss2d(38, 36, 14) + 0.3*gauss2d(42, 30, 8)
	sp_main = gauss1d(110, 14)
	cube = s_main[np.newaxis] * sp_main[:, np.newaxis, np.newaxis]

	# Compact companion
	s_comp = 0.35 * gauss2d(56, 53, 5)
	sp_comp = gauss1d(118, 6)
	cube += s_comp[np.newaxis] * sp_comp[:, np.newaxis, np.newaxis]

	# Faint continuum
	cont = 0.04 * gauss2d(39, 37, 20)
	cube += cont[np.newaxis]

	# Noise
	cube += rng.normal(0, 0.018, cube.shape)

	wl = 6500.0 + np.arange(nz) * 1.25   # 6500–6773 Å
	return cube, wl


if __name__ == '__main__':
	if len(sys.argv) > 1:
		print(f'Loading {sys.argv[1]} …')
		results = run_selector(fits_path=sys.argv[1])
else:
	#cube, wl = _demo_cube()
	def window(cube, wl):
		results = run_selector(cube=cube, wl=wl, wl_unit='CUNIT3')
		
		#print('\nFinal result dict:')
		#print(f"  roi			= {results['roi']}")
		#print(f"  channel_ranges = {results['channel_ranges']}")
		#print(f"  all_channels   = {results['all_channels']}")
				
		return results


