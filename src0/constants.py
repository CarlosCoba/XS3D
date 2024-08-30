import numpy as np

# speed of light in km/s
__c__ = 299792.458
# speed of light in Angstrom/s
__c__Angs = 2.99792458e18
#speed of light in mu/s
__c__mu = 2.99792458e14

# convert std to FWHM
__sigma_2_FWHM__ = 2*(2*np.log(2))**0.5
# convert FWHM to sigma
__FWHM_2_sigma__ = 1/__sigma_2_FWHM__


# Cosmological parameters
__Hubble_constant__ = 71  # km/(s Mpc)
__Omega_matter__ = 0.27
__Omega_Lambda__ = 0.73

__Ha_central_wl__ = 6562.68

