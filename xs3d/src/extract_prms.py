import numpy as np


def extractp(best_rings,vmode='circular'):
	
	best = {}
			
	scalar_fields = ["radius","v_rot", "v_rad", "v_disp", "v_sys", "inc", "pa",
					 "x_center", "y_center", "z_scale",
					 "v_2r", "v_2t", "phi_bar"]

	if 'hrm' in vmode:

		scalar_fields = ["radius","v_rot", "v_disp", "v_sys", "inc", "pa",
					 "x_center", "y_center", "z_scale", "phi_bar"
						]
					 	
	for f in scalar_fields:
			vals = np.array([getattr(r, f) for r in best_rings])
			best[f] = vals                  
                      
                 
	return best
	

def extract_harmonics(best_rings):
    """
    Extract harmonic velocity coefficients from a list of best-fit rings.

    Returns a dict with keys 'radii', 'c_m1', 's_m1', 'c_m2', 's_m2', etc.
    for every harmonic order present in any ring.

    Parameters
    ----------
    best_rings : list of Ring
        Best-fit rings returned by fit_rings().

    Returns
    -------
    result : dict
        'radii'  : np.ndarray of ring radii
        'c_mN'   : np.ndarray of cosine coefficients for order N
        's_mN'   : np.ndarray of sine   coefficients for order N
    """
    radii = np.array([r.radius for r in best_rings])

    # Collect all harmonic orders present across all rings
    all_orders = set()
    for ring in best_rings:
        if ring.harmonics:
            all_orders.update(ring.harmonics.keys())
    all_orders = sorted(all_orders)

    result = {'radii': radii}

    for m in all_orders:
        c_vals = np.array([ring.harmonics.get(m, (0.0, 0.0))[0]
                           for ring in best_rings])
        s_vals = np.array([ring.harmonics.get(m, (0.0, 0.0))[1]
                           for ring in best_rings])
        result[f'c_m{m}'] = c_vals
        result[f's_m{m}'] = s_vals

    return result
