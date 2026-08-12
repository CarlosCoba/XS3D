"""
plot_rings3d.py
===============
3D visualisation of tilted-ring models, plus sky-plane projection.

Functions
---------
plot_rings_3d        : 3D perspective view with configurable viewing angle
plot_sky_projection  : 2D projection of rings onto the (x_sky, y_sky) plane

Viewing angle guide
-------------------
Face-on  (z_los toward screen, x_sky / y_sky plane visible):
    elev=90,  azim=0

Edge-on  (looking along y_sky axis):
    elev=0,   azim=-90

Side view  (looking along x_sky axis):
    elev=0,   azim=0

Oblique  (default, shows 3D depth):
    elev=28,  azim=-60
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.patheffects as pe

from .colormaps_CLC import vel_map
cmap = vel_map()
#cmap = 'plasma'
cmap='RdYlBu'

# ── geometry ──────────────────────────────────────────────────────────────

def ring_surface_3d(ring, n_phi=120):
    """
    Return 3D surface arrays for one thick tilted ring in sky coordinates.

    The ring is an annular cylinder in the disk frame:
      inner radius = ring.radius - ring.width/2
      outer radius = ring.radius + ring.width/2
      half-height  = 2 × ring.z_scale  (visual thickness)

    Then rotated to sky frame via inclination + PA.

    Returns
    -------
    surfaces : list of (X, Y, Z) tuples   for ax.plot_surface
    midline  : (xm, ym, zm)               midplane circle at z_disk=0
    """
    r      = ring.radius
    dr     = ring.width / 2.
    r_in   = max(r - dr, 0.)
    r_out  = r + dr
    z_half = getattr(ring, 'z_scale', 0.) * 2.0
    if z_half <= 0:
        z_half = max(dr * 0.3, 0.05)

    phi   = np.linspace(0., 2.*np.pi, n_phi, endpoint=True)
    inc_r = np.radians(ring.inc)
    pa_r  = np.radians(ring.pa)

    def disk_to_sky(xd, yd, zd):
        # Inclination: tilt around disk x-axis
        yi = yd * np.cos(inc_r) - zd * np.sin(inc_r)
        zi = yd * np.sin(inc_r) + zd * np.cos(inc_r)
        xi = xd
        # PA rotation in the sky plane
        xs = -xi * np.sin(pa_r) - yi * np.cos(pa_r)
        ys =  xi * np.cos(pa_r) - yi * np.sin(pa_r)
        return xs, ys, zi   # zi = z_los (depth along LOS)

    surfaces = []
    for z_sign in [+1, -1]:           # top and bottom annular faces
        r_arr = np.array([r_in, r_out])
        PP, RR = np.meshgrid(phi, r_arr)
        Xd = RR * np.cos(PP); Yd = RR * np.sin(PP)
        Zd = np.full_like(Xd, z_sign * z_half)
        surfaces.append(disk_to_sky(Xd, Yd, Zd))

    for r_wall in [r_out, r_in]:      # outer and inner cylindrical walls
        ZZ = np.array([-z_half, +z_half])
        PP, ZZg = np.meshgrid(phi, ZZ)
        Xd = r_wall * np.cos(PP); Yd = r_wall * np.sin(PP)
        surfaces.append(disk_to_sky(Xd, Yd, ZZg))

    midline = disk_to_sky(r*np.cos(phi), r*np.sin(phi), np.zeros_like(phi))
    return surfaces, midline


def ring_sky_ellipses(ring, n_phi=360):
    """
    Return (x_sky, y_sky) arrays for the projected ellipses of one ring
    in the sky plane — the outer edge, inner edge, and midline.

    These are the 2D projections obtained by setting z_disk=0 and
    projecting through inclination and PA.

    Returns
    -------
    outer : (x, y)   outer ellipse edge
    inner : (x, y)   inner ellipse edge
    mid   : (x, y)   midline ellipse (r = ring.radius)
    """
    phi   = np.linspace(0., 2.*np.pi, n_phi, endpoint=True)
    inc_r = np.radians(ring.inc)
    pa_r  = np.radians(ring.pa)
    dr    = ring.width / 2.

    def project(r_disk):
        xd = r_disk * np.cos(phi)
        yd = r_disk * np.sin(phi)
        # Inclination (z_disk=0 so zi=0)
        yi = yd * np.cos(inc_r)
        xi = xd
        # PA rotation
        xs = -xi * np.sin(pa_r) - yi * np.cos(pa_r)
        ys =  xi * np.cos(pa_r) - yi * np.sin(pa_r)
        return xs, ys

    outer = project(ring.radius + dr)
    inner = project(max(ring.radius - dr, 0.))
    mid   = project(ring.radius)
    return outer, inner, mid


# ── 3D plot ───────────────────────────────────────────────────────────────

def plot_rings_3d(rings,
                  title='Tilted rings (3D)',
                  color_by='radius',
                  show_midline=True,
                  show_surface=True,
                  show_sky_shadow=True,
                  alpha_surface=0.25,
                  alpha_shadow=0.7,
                  cmap_name=cmap,
                  figsize=(12, 9),
                  elev=28.,
                  azim=-60.,
                  unit='arcsec',
                  ax=None):
    """
    Plot Ring objects as thick 3D annuli in sky coordinates.

    The three axes are:
      x_sky   East-West on the sky (RA direction, rotated by PA)
      y_sky   North-South on the sky (Dec direction, rotated by PA)
      z_los   Physical depth along the line of sight
              (does NOT appear on the detector — it maps to velocity
               via Doppler: v_los = v_rot×cos(φ)×sin(inc))

    VIEWING ANGLE GUIDE
    -------------------
    Face-on  (z_los toward screen, sky plane fully visible):
        elev=90, azim=0

    Edge-on  (looking along y_sky, LOS in the plane):
        elev=0,  azim=-90

    Side view  (looking along x_sky):
        elev=0,  azim=0

    Oblique default  (shows depth):
        elev=28, azim=-60

    Parameters
    ----------
    rings            : list of Ring
    title            : str
    color_by         : 'radius' | 'vrot' | 'pa' | 'inc' | 'vdisp'
    show_midline     : bool   draw midplane ring outline
    show_surface     : bool   draw thick-ring 3D surface
    show_sky_shadow  : bool   project midline rings onto z_los=z_min plane
                              (shows what the detector actually sees)
    alpha_surface    : float  3D surface transparency
    alpha_shadow     : float  sky-plane shadow transparency
    cmap_name        : str    matplotlib colourmap
    figsize          : tuple
    elev             : float  elevation viewing angle [degrees]
    azim             : float  azimuthal viewing angle [degrees]
    unit             : str    spatial axis label
    ax               : Axes3D or None

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    radii = np.array([r.radius for r in rings])
    r_max = radii.max()

    _fields = {
        'radius': (f'radius [{unit}]', [r.radius  for r in rings]),
        'vrot':   ('v_rot [km/s]',     [r.v_rot   for r in rings]),
        'pa':     ('PA [°]',           [r.pa      for r in rings]),
        'inc':    ('inc [°]',          [r.inc     for r in rings]),
        'vdisp':  ('v_disp [km/s]',    [r.v_disp  for r in rings]),
    }
    clabel, cvals_list = _fields.get(color_by, _fields['radius'])
    cvals  = np.array(cvals_list, dtype=float)
    cmap   = cm.get_cmap(cmap_name)
    norm   = plt.Normalize(cvals.min(), cvals.max())
    colors = cmap(norm(cvals))

    # z floor for the sky-plane shadow
    z_floor = 0#-r_max * 1.55
    off = r_max*10
        
    for ring, color in zip(rings, colors):
        surfaces, (xm, ym, zm) = ring_surface_3d(ring)

        if show_surface:
            for (Xs, Ys, Zs) in surfaces:
                ax.plot_surface(Xs, Ys, Zs+off,
                                color=color, alpha=alpha_surface,
                                linewidth=0, antialiased=True)

        if show_midline:
            ax.plot(xm, ym, zm+off, color=color, lw=1.8, alpha=0.95, zorder=5)

        if show_sky_shadow:
            # Project midline onto the z=z_floor plane (sky-plane shadow)
            ax.plot(xm, ym, np.full_like(zm+off, z_floor),
                    color=color, lw=1.2, alpha=alpha_shadow,
                    linestyle='--', zorder=2)

    # Grey label for the shadow plane
    if show_sky_shadow:
        ax.text(r_max*0.6, r_max*0.6, z_floor,
                'sky plane\n(x$_{sky}$, y$_{sky}$)',
                fontsize=12, color='black', ha='center', va='bottom',
                alpha=0.7)

    # Coordinate axis arrows in lower-left front corner
    ax_len = r_max * 0.30
    corner = np.array([-r_max*1.05, -r_max*1.05, z_floor])
    axis_specs = [
        (np.array([ax_len, 0,      0     ]), 'x$_{sky}$ (RA)',    'crimson'),
        (np.array([0,      ax_len, 0     ]), 'y$_{sky}$ (Dec)',   'forestgreen'),
        (np.array([0,      0,      ax_len]), 'z$_{los}$ (depth)', 'steelblue'),
    ]
    for dvec, lbl, col in axis_specs:
        tip = corner + dvec
        ax.quiver(*corner, *dvec, color=col, lw=1.5,
                  arrow_length_ratio=0.20, normalize=False)
        ax.text(*(tip + dvec*0.25), lbl,
                fontsize=8, color=col, ha='center', va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=col, alpha=0.75, linewidth=0.8))

    # Faint midplane reference disk
    gp = np.linspace(0, 2*np.pi, 60)
    gr = np.linspace(0, r_max*1.05, 2)
    GR, GP = np.meshgrid(gr, gp)
    ax.plot_wireframe(GR*np.cos(GP), GR*np.sin(GP), np.zeros_like(GR),
                      color='gray', alpha=1, lw=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    label_tmp = {'radius':'radius (arcsec)', 'vrot': 'Vrot (km/s)', 'pa': 'PA (deg)', 'inc': 'inc (deg)', 'vdisp': 'sigma (km/s)'}
    cbar=plt.colorbar(sm, ax=ax, shrink=0.35, pad=0.10, orientation='horizontal')
    cbar.ax.tick_params(labelsize=18)  # Sets the size of the tick numbers
    cbar.set_label(label_tmp[color_by], size=18)  # Sets the text and its font size


    ax.set_xlabel(f'x$_{{sky}}$ ({unit})', fontsize=18, labelpad=6)
    ax.set_ylabel(f'y$_{{sky}}$ ({unit})', fontsize=18, labelpad=6)
    ax.set_zlabel(f'z$_{{los}}$ ({unit})', fontsize=18, labelpad=-6)
    ax.zaxis.set_ticklabels([])
    #ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.view_init(elev=elev, azim=azim)
    #ax.set_box_aspect([1, 1, 0.55])

    return fig, ax


# ── 2D sky-plane projection ───────────────────────────────────────────────

def plot_sky_projection(rings,
                        title='Sky-plane projection (x$_{sky}$, y$_{sky}$)',
                        color_by='radius',
                        show_midline=True,
                        show_width=True,
                        show_major_axis=True,
                        cmap_name='plasma',
                        figsize=(7, 7),
                        unit='arcsec',
                        ax=None):
    """
    Plot the 2D projection of tilted rings onto the sky plane (x_sky, y_sky).

    This is exactly what the detector sees spatially — the projection
    of each ring through inclination and PA onto the plane of the sky.
    The z_los (depth) information is lost in this projection, just as it
    is on the detector.

    For each ring the projection is an ellipse with:
      semi-major axis = ring.radius
      semi-minor axis = ring.radius × cos(inc)
      position angle  = ring.pa  (CCW from North)

    Parameters
    ----------
    rings           : list of Ring
    title           : str
    color_by        : 'radius' | 'vrot' | 'pa' | 'inc' | 'vdisp'
    show_midline    : bool  draw the projected midline ellipse
    show_width      : bool  shade the projected ring width (inner/outer edge)
    show_major_axis : bool  draw the projected major axis line
    cmap_name       : str   matplotlib colourmap
    figsize         : tuple
    unit            : str   axis label
    ax              : Axes or None

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    radii = np.array([r.radius for r in rings])
    r_max = radii.max()

    _fields = {
        'radius': (f'radius [{unit}]', [r.radius  for r in rings]),
        'vrot':   ('v_rot [km/s]',     [r.v_rot   for r in rings]),
        'pa':     ('PA [°]',           [r.pa      for r in rings]),
        'inc':    ('inc [°]',          [r.inc     for r in rings]),
        'vdisp':  ('v_disp [km/s]',    [r.v_disp  for r in rings]),
    }
    clabel, cvals_list = _fields.get(color_by, _fields['radius'])
    cvals  = np.array(cvals_list, dtype=float)
    cmap   = cm.get_cmap(cmap_name)
    norm   = plt.Normalize(cvals.min(), cvals.max())
    colors = cmap(norm(cvals))

    for ring, color in zip(rings, colors):
        outer, inner, mid = ring_sky_ellipses(ring)

        if show_width:
            # Fill the annular region between inner and outer ellipse
            from matplotlib.patches import PathPatch
            from matplotlib.path import Path
            # Outer boundary (CCW) then inner boundary (CW = reversed)
            verts_out = list(zip(outer[0], outer[1]))
            verts_in  = list(zip(inner[0][::-1], inner[1][::-1]))
            verts     = verts_out + verts_in + [verts_out[0]]
            codes     = ([Path.MOVETO] +
                         [Path.LINETO]*(len(verts_out)-1) +
                         [Path.LINETO]*len(verts_in) +
                         [Path.CLOSEPOLY])
            path  = Path(verts, codes)
            patch = PathPatch(path, facecolor=color, edgecolor='none',
                              alpha=0.25, zorder=2)
            ax.add_patch(patch)

        if show_midline:
            ax.plot(*mid, color=color, lw=1.8, alpha=0.9, zorder=3)

        if show_major_axis:
            # Major axis: two points at phi=0 and phi=pi on midline
            # phi=0: x_disk=r, y_disk=0 → projected end of receding major axis
            inc_r = np.radians(ring.inc); pa_r = np.radians(ring.pa)
            r     = ring.radius
            def proj(xd, yd):
                yi = yd*np.cos(inc_r); xi = xd
                xs = -xi*np.sin(pa_r) - yi*np.cos(pa_r)
                ys =  xi*np.cos(pa_r) - yi*np.sin(pa_r)
                return xs, ys
            xr, yr = proj( r, 0.)  # receding end
            xa, ya = proj(-r, 0.)  # approaching end
            ax.plot([xa, xr], [ya, yr], color=color, lw=0.8,
                    alpha=1, linestyle=':', zorder=1)

    # Cross-hairs at kinematic centre
    cx = np.mean([r.x_center for r in rings])
    cy = np.mean([r.y_center for r in rings])
    ax.axhline(0, color='k', lw=0.6, alpha=0.3)
    ax.axvline(0, color='k', lw=0.6, alpha=0.3)
    ax.plot(0, 0, '+k', ms=8, mew=1.5, zorder=10)

    # Axis labels with physical meaning
    ax.set_xlabel(f'x$_{{sky}}$  (RA direction)  [{unit}]', fontsize=10)
    ax.set_ylabel(f'y$_{{sky}}$  (Dec direction)  [{unit}]', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)

    lim = r_max * 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # Note about what is NOT shown
    ax.text(0.02, 0.02,
            'z$_{los}$ (depth) not shown — maps to velocity on detector',
            transform=ax.transAxes, fontsize=7, color='gray',
            va='bottom', style='italic')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=clabel, shrink=0.5, pad=0.02)

    return fig, ax
