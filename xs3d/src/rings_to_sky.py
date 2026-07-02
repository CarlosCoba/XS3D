import numpy as np
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Ring sky-projection utilities
# ---------------------------------------------------------------------------
 
def project_ring(r_arcsec, pa_deg, inc_deg, cx_pix, cy_pix,
				 dx_arcsec, n_phi=360):
	"""
	Project a circular ring onto the sky plane.
 
	Samples ``n_phi`` points uniformly in azimuthal angle φ ∈ [0, 2π)
	and applies the same three-step transform used in ``_disk_to_sky``:
 
		Step 1 — inclination  : y_sky = y_disk × cos(inc)
		Step 2 — PA rotation  : rotate (x,y) by PA
		Step 3 — pixel coords : x_pix = cx + x_sky / dx
 
	This gives the **exact locus** of projected ring positions —
	physically identical to where clouds are deposited by
	``RingBuilder.build_ring``.  It is NOT an analytical ellipse
	formula, so it correctly handles:
 
	- Any inclination (including inc=90°, where the ring projects
	  to a line segment along the major axis)
	- Rings with different PA per radius (kinematic warp)
	- Finite ring width when combined with ``project_ring_edges``
 
	The result is the same as a matplotlib ``Ellipse`` patch for a
	flat disk (z=0), but the sampled-point approach generalises to
	all cases without special-casing.
 
	Parameters
	----------
	r_arcsec  : float   ring radius [arcsec]
	pa_deg	: float   position angle [degrees, N→E, receding side]
	inc_deg   : float   inclination [degrees]
	cx_pix	: float   kinematic centre X [pixels]
	cy_pix	: float   kinematic centre Y [pixels]
	dx_arcsec : float   pixel scale [arcsec/pixel]
	n_phi	 : int	 number of azimuthal sample points (default 360)
 
	Returns
	-------
	x_pix : np.ndarray (n_phi,)   sky x coordinates [pixels]
	y_pix : np.ndarray (n_phi,)   sky y coordinates [pixels]
 
	Examples
	--------
	# Overlay best-fit ring loci on a moment-0 map
	for ring in best_rings:
		xp, yp = project_ring(ring.radius, ring.pa, ring.inc,
							   ring.x_center, ring.y_center, cfg.dx)
		ax.plot(xp, yp, '--', color='lime', lw=1.2)
	"""
	pa_r  = np.radians(pa_deg)
	inc_r = np.radians(inc_deg)
	phi   = np.linspace(0., 2.*np.pi, n_phi, endpoint=False)
 
	# Disk-plane midplane (z=0)
	x_disk = r_arcsec * np.cos(phi)
	y_disk = r_arcsec * np.sin(phi)
 
	# Step 1: inclination — compress minor axis by cos(inc)
	x_inc = x_disk
	y_inc = y_disk * np.cos(inc_r)
 
	# Step 2: PA rotation (North→East, matches _disk_to_sky)
	x_sky = -x_inc * np.sin(pa_r) - y_inc * np.cos(pa_r)
	y_sky = x_inc * np.cos(pa_r) - y_inc * np.sin(pa_r)
 
	# Step 3: pixel coordinates
	x_pix = cx_pix + x_sky / dx_arcsec
	y_pix = cy_pix + y_sky / dx_arcsec
 
	return x_pix, y_pix
 
 
def project_ring_edges(r_arcsec, width_arcsec, pa_deg, inc_deg,
					   cx_pix, cy_pix, dx_arcsec, n_phi=360):
	"""
	Project the inner and outer edges of a ring with finite width.
 
	Calls ``project_ring`` twice — once for the inner radius
	(r − width/2) and once for the outer radius (r + width/2).
	The two loci bracket the annular band on the sky.
 
	Use the returned arrays to fill the annular strip:
 
		x_poly = np.concatenate([x_outer, x_inner[::-1], [x_outer[0]]])
		y_poly = np.concatenate([y_outer, y_inner[::-1], [y_outer[0]]])
		ax.fill(x_poly, y_poly, ...)
 
	Parameters
	----------
	r_arcsec	  : float   ring centre radius [arcsec]
	width_arcsec  : float   full radial width of ring [arcsec]
	pa_deg		: float   position angle [degrees]
	inc_deg	   : float   inclination [degrees]
	cx_pix		: float   kinematic centre X [pixels]
	cy_pix		: float   kinematic centre Y [pixels]
	dx_arcsec	 : float   pixel scale [arcsec/pixel]
	n_phi		 : int	 azimuthal sample points (default 360)
 
	Returns
	-------
	x_inner, y_inner : np.ndarray (n_phi,)   inner edge [pixels]
	x_outer, y_outer : np.ndarray (n_phi,)   outer edge [pixels]
	"""
	r_inner = max(r_arcsec - width_arcsec / 2., 0.)
	r_outer = r_arcsec + width_arcsec / 2.
	xi, yi  = project_ring(r_inner, pa_deg, inc_deg,
							cx_pix, cy_pix, dx_arcsec, n_phi=n_phi)
	xo, yo  = project_ring(r_outer, pa_deg, inc_deg,
							cx_pix, cy_pix, dx_arcsec, n_phi=n_phi)
	return xi, yi, xo, yo
 
 

def overlay_rings(ax, rings, dx_arcsec, mode='locus',
                  col_fill='#362B32', alpha=0.7, lw=1.2,
                  fill_alpha=0.15, n_phi=360, label_rings=False,
                  show_major_axis=False,
                  receding_color='red', approaching_color='blue',
                  spine_alpha=0.8, spine_lw=1.0):
    """
    Overlay projected ring loci or bands on a sky-plane axes.
 
    Optionally draws a colour-coded major-axis spine connecting the
    receding endpoints (red) and approaching endpoints (blue) across
    all rings.  This immediately reveals any kinematic warp or twist
    in the PA across rings.
 
    Parameters
    ----------
    ax              : matplotlib Axes
    rings           : list of Ring   anchor or best-fit rings
    dx_arcsec       : float          pixel scale [arcsec/pixel]
    mode            : str
        'locus' — midplane ring centre line only
        'band'  — inner/outer edges + fill
        'both'  — locus + band
    color           : str or list    ring colour (single or one per ring)
    alpha           : float          ring line opacity
    fill_alpha      : float          fill opacity (band/both modes)
    n_phi           : int            azimuthal sample points
    label_rings     : bool           annotate each ring with its radius
    show_major_axis : bool
        If True (default), draw the receding and approaching major-axis
        spines connecting the outermost points of each ring along
        phi=0 (receding, red) and phi=pi (approaching, blue).
        This reveals warp: a straight spine means no PA twist,
        a bent spine shows how PA changes with radius.
    receding_color  : str   colour for the receding side spine (default 'red')
    approaching_color: str  colour for the approaching side spine (default 'blue')
    spine_alpha     : float opacity of the spine lines
    spine_lw        : float linewidth of the spine lines
 
    Examples
    --------
    # Overlay best-fit ring bands with major-axis spines
    overlay_rings(ax, best_rings, dx_arcsec=0.2, mode='band',
                  color='lime', fill_alpha=0.2,
                  show_major_axis=True)
 
    # Suppress spines (rings only)
    overlay_rings(ax, best_rings, 0.2, show_major_axis=False)
    """
    # Normalise color argument
    color = 'k'
    if isinstance(color, str) or isinstance(color, tuple):
        colors = [color] * len(rings)
    else:
        colors = list(color)
 
    # Collect major-axis endpoints across all rings for the spine
    receding_x,    receding_y    = [], []   # phi = 0   (red)
    approaching_x, approaching_y = [], []   # phi = pi  (blue)
 
    for i, (ring, col) in enumerate(zip(rings, colors)):
        cx   = ring.x_center
        cy   = ring.y_center
        r_as = ring.radius
        w_as = ring.width
        pa   = ring.pa
        inc  = ring.inc
 
        if mode in ('locus', 'both'):
            xp, yp = project_ring(r_as, pa, inc, cx, cy, dx_arcsec, n_phi)
            # Close the curve
            xp = np.append(xp, xp[0])
            yp = np.append(yp, yp[0])
            ax.plot(xp, yp, '-', color=col, alpha=alpha, lw=lw)
 
        if mode in ('band', 'both'):
            xi, yi, xo, yo = project_ring_edges(r_as, w_as, pa, inc,
                                                 cx, cy, dx_arcsec, n_phi)
            xo_c = np.append(xo, xo[0]); yo_c = np.append(yo, yo[0])
            xi_c = np.append(xi, xi[0]); yi_c = np.append(yi, yi[0])
            ax.plot(xo_c, yo_c, color=col, alpha=alpha, lw=lw*0.7)
            ax.plot(xi_c, yi_c, color=col, alpha=alpha, lw=lw*0.7,
                    ls='--')
            col_fill
            x_poly = np.concatenate([xo, xi[::-1], [xo[0]]])
            y_poly = np.concatenate([yo, yi[::-1], [yo[0]]])
            ax.fill(x_poly, y_poly, color=col_fill, alpha=fill_alpha)
 
        if label_rings:
            xp0, yp0 = project_ring(r_as, pa, inc, cx, cy, dx_arcsec, n_phi=4)
            ax.text(xp0[0]+1, yp0[0]+1, f"{r_as:.1f}\"",
                    color=col, fontsize=6, alpha=alpha,
                    ha='left', va='bottom')
 
        # Major-axis endpoints:
        #   phi=0   → receding side  (x_disk=r, y_disk=0)
        #   phi=pi  → approaching side (x_disk=-r, y_disk=0)
        if show_major_axis:
            # Sample just 2 points: phi=0 and phi=pi
            # Use project_ring with n_phi=2; linspace(0,2pi,2,endpoint=False)
            # gives phi=0 and phi=pi exactly
            xpa, ypa = project_ring(r_as, pa, inc, cx, cy, dx_arcsec, n_phi=2)
            receding_x.append(xpa[0]);    receding_y.append(ypa[0])
            approaching_x.append(xpa[1]); approaching_y.append(ypa[1])
 
    # Draw the major-axis spines connecting all ring endpoints
    # Sort by radius to ensure the line follows the rings from inside out
    if show_major_axis and len(rings) > 1:
        # Sort by ring radius (inner → outer)
        order = np.argsort([r.radius for r in rings])
 
        rx = np.array(receding_x)[order]
        ry = np.array(receding_y)[order]
        ax = ax   # keep reference
        ax.plot(rx, ry,
                color=receding_color, alpha=spine_alpha,
                lw=spine_lw, ls='-',
                label='Receding side')
 
        ax_ref = ax
        ax_ref.plot(np.array(approaching_x)[order],
                    np.array(approaching_y)[order],
                    color=approaching_color, alpha=spine_alpha,
                    lw=spine_lw, ls='-',
                    label='Approaching side')
