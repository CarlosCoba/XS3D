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


def compass(ax, header=None,
				 CD1_1=None, CD1_2=None, CD2_1=None, CD2_2=None,
				 length_frac=0.10,
				 anchor=(0.85, 0.85),
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
	#fig	= ax.figure
	#bbox   = ax.get_window_extent(renderer=fig.canvas.get_renderer())
	ax_w   = 1#bbox.width
	ax_h   = 1#bbox.height
	target = length_frac * min(ax_w, ax_h)   # display pixels

	xlim = ax.get_xlim();  ylim = ax.get_ylim()
	xr   = abs(xlim[1] - xlim[0]);  yr = abs(ylim[1] - ylim[0])
	sx   = ax_w / max(xr, 1e-30);   sy = ax_h / max(yr, 1e-30)

	def normalise(dx_px, dy_px):
		"""Return display-pixel unit vector scaled to target length,
		then back in axes-fraction."""
		disp = np.array([dx_px*sx, dy_px*sy])
		disp = disp / np.linalg.norm(disp) * target
		return np.array([disp[0]/ax_w, disp[1]/ax_h])

	dN = normalise(dx_N_px, dy_N_px)
	dE = normalise(dx_E_px, dy_E_px)
	print(dN,dE)
	# ── 4. Anchor clamping — keep tips+labels inside [m, 1-m] ─────
	margin	= 0.01
	label_gap = 0.04		  # label centre offset beyond the tip

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

	# ── 5. Label: position + rotation ─────────────────────────────
	# Place the label centre at (tip + unit*gap) and rotate it to
	# align with the arrow direction.
	#
	# Rotation angle = angle of the arrow from axes +x (CCW degrees).
	# Clamped to [-90°, 90°] so the text is never upside-down:
	#   arrows pointing left (angle ∈ (90°,270°)) are flipped by 180°.
	# Alignment: ha='center', va='bottom' — the text baseline sits just
	# above the arrow tip, centred on the arrow axis.  This keeps the
	# label close to the arrowhead regardless of rotation.

	def label_info(tip, d_ax):
		unit = d_ax / np.linalg.norm(d_ax)
		pos  = tip + unit * label_gap

		# Arrow angle in DISPLAY space (not axes-fraction, to account
		# for non-square display — same scale factors sx, sy used above)
		dx_disp = d_ax[0] * ax_w
		dy_disp = d_ax[1] * ax_h
		angle_raw = np.degrees(np.arctan2(dy_disp, dx_disp))

		# Clamp to [-90, 90] — never upside-down
		if   angle_raw >  90: angle_text = angle_raw + 180
		elif angle_raw < -90: angle_text = angle_raw + 180
		else:				 angle_text = angle_raw + 90

		return pos, angle_text

	pos_N, rot_N = label_info(tip_N, dN)
	pos_E, rot_E = label_info(tip_E, dE)

	# ── 6. Draw ───────────────────────────────────────────────────
	outline = [pe.withStroke(linewidth=3.0, foreground='w')]
	trans   = ax.transAxes
	artists = []

	for tip, (pos, rot), label in [
		(tip_N, (pos_N, rot_N), 'N'),
		(tip_E, (pos_E, rot_E), 'E'),
	]:
		arr = ax.annotate(
			'',
			xy	  = tuple(tip),
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
			ha='center', va='bottom',   # bottom = above the baseline
			rotation=rot,
			rotation_mode='anchor',
			transform=trans,
			color=color,
			fontsize=fontsize,
			fontweight='bold',
			clip_on=False,
		)
		#txt.set_path_effects(outline)
		artists.append(txt)

	return artists
	
	
