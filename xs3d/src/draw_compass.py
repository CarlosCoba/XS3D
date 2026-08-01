"""
draw_compass.py
===============
Draw a North/East compass rose on a matplotlib Axes.

Design
------
- Single connected L-shape: both arrows share the same base vertex.
- Equal physical length: normalised in display-pixel space.
- Labels rotate with the arrows — never upside-down (clamped to ±90°).
- Labels always inside the figure: anchor auto-clamped from edges.
"""

import numpy as np
import matplotlib.patheffects as pe


def compass(ax, header=None, pa=0,
				 length_frac=0.10,
				 color='k',
				 fontsize=9,
				 lw=1.5):
	"""
	Draw a N/E compass rose anchored in the upper-right of the axes.

	Labels rotate with the arrows so they remain tangent to the arrow
	direction at every image orientation.

	Parameters
	----------
	ax		  : matplotlib Axes
	header	  : dict-like or None   FITS header (CD or PC+CDELT)
	CD1_1/CD1_2/CD2_1/CD2_2 : float  CD matrix [deg/pixel] (if no header)
	length_frac : float   arrow length as fraction of shorter axes dim (0.10)
	anchor	  : (x,y)   axes-fraction base of both arrows (0.85, 0.85)
	color	   : str	 arrow and label colour ('white')
	fontsize	: int	 label size (9)
	lw		  : float   arrow line width (1.5)

	Returns
	-------
	list of matplotlib artists (4 items: 2 arrows + 2 labels)
	"""
	if np.all([k in header for k in ['CD1_1','CD1_2','CD2_1','CD2_2']]):
		pass
	elif np.all([k in header for k in ['PC1_1','PC1_2','PC2_1','PC2_2']]):
		if np.all([k in header for k in ['CDELT1','CDELT2']]):
			pass
	else:
		return None

	quadrant= ( ((pa+90) % 360) // 90 ) + 1
	pos		= 	(quadrant % 4) + 1
	if pos == 2:
		pos+=1
	if pos == 1: anchor=(0.85, 0.85)
	if pos == 3: anchor=(0.15, 0.15)
	if pos == 4: anchor=(0.85, 0.15)	

	# ── 1. CD matrix ──────────────────────────────────────────────
	def _g(k, pc, cd, fb):
		if k  in header: return float(header[k])
		if pc in header: return float(header[pc]) * float(header.get(cd, fb))
		return fb
			
	c11 = _g('CD1_1','PC1_1','CDELT1',-1e-4)
	c12 = _g('CD1_2','PC1_2','CDELT1', 0.0)
	c21 = _g('CD2_1','PC2_1','CDELT2', 0.0)
	c22 = _g('CD2_2','PC2_2','CDELT2', 1e-4)

	det = c11*c22 - c12*c21
	if abs(det) < 1e-30:
		raise ValueError("CD matrix is singular")

	# ── 2. Sky directions in pixel space (CD^{-1}) ────────────────
	dx_N_px, dy_N_px = -c12/det,  c11/det   # North: CD^{-1}@(0,+1)
	dx_E_px, dy_E_px =  c22/det, -c21/det   # East:  CD^{-1}@(+1,0)

	# ── 3. Normalise to equal physical length in display pixels ───
	fig	= ax.figure
	bbox   = ax.get_window_extent(renderer=fig.canvas.get_renderer())
	ax_w   = bbox.width
	ax_h   = bbox.height
	target = length_frac * min(ax_w, ax_h)   # display pixels

	xlim = ax.get_xlim();  ylim = ax.get_ylim()
	xr   = abs(xlim[1] - xlim[0]);  yr = abs(ylim[1] - ylim[0])
	sx   = ax_w / max(xr, 1e-30);   sy = ax_h / max(yr, 1e-30)

	def normalise(dx_px, dy_px):
		"""Normalise to target display-pixel length.
		Returns (axes-fraction vector, display-pixel unit vector).
		The display-pixel unit vector is used for the rotation angle —
		it gives the angle the arrow VISUALLY appears on screen,
		regardless of axes aspect ratio or data-range distortion.
		"""
		disp	  = np.array([dx_px * sx, dy_px * sy])
		unit_disp = disp / np.linalg.norm(disp)
		disp_norm = unit_disp * target
		ax_frac   = np.array([disp_norm[0] / ax_w, disp_norm[1] / ax_h])
		return ax_frac, unit_disp

	dN, uN = normalise(dx_N_px, dy_N_px)
	dE, uE = normalise(dx_E_px, dy_E_px)

	# ── 4. Anchor clamping — keep tips+labels inside [m, 1-m] ─────
	margin	= 0.02
	label_gap = 0.01		  # label centre offset beyond the tip

	base = np.array(anchor, dtype=float)

	for d in [dN, dE]:
		unit = d / np.linalg.norm(d)
		for dim in [0, 1]:
			hi = base[dim] + d[dim] + max(unit[dim],0)*label_gap - (1-margin)
			lo = margin - (base[dim] + d[dim] + min(unit[dim],0)*label_gap)
			if hi > 0: base[dim] -= hi
			if lo > 0: base[dim] += lo

	tip_N = base + dN
	tip_E = base + dE

	# ── 5. Label: position and alignment (text always upright) ────
	# The letter N or E should always be readable — never rotated.
	# Instead of rotating the text, we:
	#   1. Place it BEYOND the arrow tip, offset along the arrow direction
	#   2. Choose ha/va from the arrow's display-space quadrant so the
	#	  label sits cleanly outside the arrowhead without overlapping it.
	#
	# This matches standard astronomical compass roses (HST, MUSE
	# pipeline outputs, aplpy): letters are always upright; their
	# position relative to the tip conveys the direction.

	def label_info(tip, d_ax, u_disp):
		"""
		tip	: axes-fraction tip of arrow
		d_ax   : axes-fraction arrow vector (for gap offset direction)
		u_disp : display-pixel unit vector (for quadrant → ha/va)
		"""
		unit_ax = d_ax / np.linalg.norm(d_ax)
		pos	 = tip + unit_ax * label_gap

		# Choose alignment so label sits beyond the tip, not on top of it.
		# Threshold 0.3 avoids ambiguous corners (diagonal arrows).
		ux, uy = u_disp		   # display-space direction
		ha = ('left'   if ux >  0.3 else
			  'right'  if ux < -0.3 else 'center')
		va = ('bottom' if uy >  0.3 else
			  'top'	if uy < -0.3 else 'center')
		return pos, ha, va

	pos_N, ha_N, va_N = label_info(tip_N, dN, uN)
	pos_E, ha_E, va_E = label_info(tip_E, dE, uE)

	# ── 6. Draw ───────────────────────────────────────────────────
	outline = [pe.withStroke(linewidth=3.0, foreground='w')]
	trans   = ax.transAxes
	artists = []

	for tip, (pos, ha, va), label in [
		(tip_N, (pos_N, ha_N, va_N), 'N'),
		(tip_E, (pos_E, ha_E, va_E), 'E'),
	]:
		arr = ax.annotate(
			'',
			xy	  	= tuple(tip),
			xytext  = tuple(base),
			xycoords='axes fraction',
			textcoords='axes fraction',
			arrowprops=dict(
				arrowstyle='->', color=color,
				lw=lw, mutation_scale=10,
				shrinkA=0, shrinkB=0,
			),
			annotation_clip=False,
		)
		arr.arrow_patch.set_path_effects(outline)
		artists.append(arr)

		txt = ax.text(
			pos[0], pos[1], label,
			ha=ha, va=va,
			transform=trans,
			color=color,
			fontsize=fontsize,
			clip_on=False,
		)
		txt.set_path_effects(outline)
		artists.append(txt)

	return artists
	
# ( ((-1+90) % 360) // 90 ) + 1	
