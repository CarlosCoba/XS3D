

*******************************************
XookSuut3D (XS3D)
*******************************************

|logo|



====

:Authors: Carlos Lopez Coba
:Contact: carlos.lopezcoba@gmail.com




Description
===========
XookSuut3D or XS3D for short, is a python tool developed to model circular and non-circular motions on 3D data cubes like those obtained
with Integral-Field-Spectroscopy, ALMA, VLA, etc. XS3D models a spectral-line cube obsevation, while corrects for the 
the observational Beam/PSF and spectral broadening (LSF). For this purpose
it makes use of the FFTW library via pyFFTW. 
XS3D is  particularly designed for spectral-line observations on frequency and wavelenght domain, which makes it suitable to 
model a wide variety of emission-lines from CO, HI, ionized gas and IR lines etc.
XS3D adopts the same minimization technique as its 2D version, XookSuut, but extended to datacubes.
XS3D is able to model circular rotation models, axisymmetric radial flows, bisymmetric flows, vertical flows, and a general harmonic decomposition of the LOSV.
To derive the best set of parameters on each kinematic model XS3D uses all the information from the datacube. Therefore,
large dimension cubes could take large CPU time to derive the best model.
XS3D is designed to take advantage of multicores, so using them through the XS3D configuration file is advisable.
 
 
Dependencies
===========

            * ::
            
                Python >= 3.8


Installation
===========

1. Go to the XS3D directory
cd /XS3D/

2.  pip install -e .
-e stands for editable, meaning that you will be able to work on the script and invoke the latest version without need to reinstall.

3. Try it. Go to any folder and type XS3D

you must get the following ::

USE: XS3D name cube.fits [mask2D] [PA] [INC] [X0] [Y0] [VSYS] vary_PA vary_INC vary_X0 vary_Y0 vary_VSYS ring_space [delta] Rstart,Rfinal cover kin_model [R_bar_min,R_bar_max] [config_file] [prefix]



Uninstall
===========

pip uninstall XS3D


Use
===========

XS3D is designed to run in command line, although you can easely set-up a python script so you can run it as a script.
This could be usefull in case you want to analyse several objects in parallel.
Please read the run_example.txt file to see how to run XS3D.
XS3D requires as input a 3D cube free of continuum emission.

Example on a low redshift galaxy with the MUSE/VLT spectrograph
===========
Following are some of the outputs you will obtain from running XS3D on a MUSE datacube for the Halpha line (lambda0=6562.68 AA).
FWHM=100 km/s.

Moment maps obtained from the observed and model datacubes.
|mom_muse|

Intrinsic velocity and velocity dispersion of the gas.
|disp_muse|

Position velocity diagram from the model cube along the major and minor axes
|pvd_muse|


Example on a high redshift galaxy (z=7.30) with ALMA
===========
This example is a high redshift object REBELS-25 at z=7.30, observed with ALMA  [CII]  (158mu=1900.537GHz). 

Observed and model moment maps from the cubes. Beam shape: BMAJ=0.134arcsec, BMIN=0.121arcsec, BPA=82deg.
|mommaps_highz|

The intrinsic circular velocity and velocity dispersion.
|disp_rebels|


Example on a protoplanetary disk of astronomical unit scales observed with  ALMA
===========

Observed and model moment maps.
|mommaps_proto|

Position velocity diagram
|pvd_proto|

Channel maps taken from the datacube and model cube
|channel_proto|

Referencing XookSuut3D
=================
 
If you are using XS3D in your work, please cite the XS release paper https://ui.adsabs.harvard.edu/abs/2024RMxAA..60...19L/abstract.
A version for 3D is ongoing.
Also, if you use the XS colormap (red-black-blue) in a different context, I would appreciate it, if you include XS in the acknowledgment section.


.. |logo| image:: logo.png
    :scale: 10 %
    :target: https://github.com/CarlosCoba/XS3D

.. |mom_muse| image:: /figures/mommaps_circular_model_NGC3351.png
    :scale: 10 %
    :target: https://github.com/CarlosCoba/XS3D

.. |disp_muse| image::  /figures/kin_circular_disp_NGC3351.png
    :scale: 10 %
    :target: https://github.com/CarlosCoba/XS3D

.. |pvd_muse| image:: /figures/pvd_circular_model_NGC3351.png
    :scale: 10 %
    :target: https://github.com/CarlosCoba/XS3D

.. |mommaps_proto| image:: /figures/mommaps_circular_model_HD163296_v2.png
    :scale: 5 %
    :target: https://github.com/CarlosCoba/XS3D

.. |pvd_proto| image:: /figures/pvd_circular_model_HD163296_v2.png
    :scale: 5 %
    :target: https://github.com/CarlosCoba/XS3D

.. |mommaps_highz| image:: /figures/mommaps_circular_model_rebels.png
   :scale: 5 %
   :target: https://github.com/CarlosCoba/XS3D

.. |disp_rebels| image::  /figures/kin_circular_disp_rebels.png
   :scale: 5 %
   :target: https://github.com/CarlosCoba/XS3D

.. |channel_proto| image::  /figures/channels_cube_circular_model_HD163296_v2.png
   :scale: 5 %
   :target: https://github.com/CarlosCoba/XS3D




