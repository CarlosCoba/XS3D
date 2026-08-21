#!/usr/bin/env python3
import sys
nargs=len(sys.argv)
import configparser
import os
this_dir, this_filename = os.path.split(__file__)
#first read the config file
CONFIG_PATH = os.path.join(this_dir,"config_file", "xs_conf.ini")
osi = ["-", ".", "#","%", "&", "","None"]
try:
	config_file=sys.argv[20] if sys.argv[20] not in osi else CONFIG_PATH
except(IndexError,IOError):
	config_file=CONFIG_PATH

input_config = configparser.ConfigParser(
			# allow a variables be set without value
			allow_no_value=True,
			# allows duplicated keys in different sections
			strict=False,
			# deals with variables inside configuratio file
			interpolation=configparser.ExtendedInterpolation())
input_config.read(config_file)
# Shortcuts to the different configuration sections variables.
config_general = input_config['general']
nthreads=config_general.getint('nthreads',1)

from .utils import *
set_threads(nthreads)

import numpy as np
from .initialize_XS3D_main import XS_out
from .pixel_params import eps_2_inc
from .man import print_manual
"""
#################################################
# 				XookSuut3D (XS3D)				#
# 				C. Lopez-Coba					#
#################################################

"""


class input_params:
	def __init__(self):
		if nargs == 2:
			if sys.argv[1] in ['-help','--help', '-man', '-h']:
				print_manual()
				quit()
		if (nargs < 19 or nargs > 22):
			print('')
			print (" USE: XS3D name cube.fits [mask] [pa] [inc] [xc] [yc] [vsys] vary_pa vary_inc vary_xc vary_yc vary_vsys ring_space [delta] rstart,rfinal cover kin_model [r_nc_min,r_nc_max] config_file [prefix]" )
			print(' ~~~~~~~~~~')
			print(' Copy the config file located at  '
				'xs3d/config_file/xs_conf.ini and place it in your working directory.')
			print(' ~~~~~~~~~~')
			exit()

		#object name
		galaxy = sys.argv[1]

		#FITS information
		datacube	= sys.argv[2]
		mask		= sys.argv[3]

		# Geometrical parameters
		pa_0	= sys.argv[4]
		inc_0	= sys.argv[5]
		xc_0	= sys.argv[6]
		yc_0	= sys.argv[7]
		vsys_0	= sys.argv[8]
		vary_pa = int(sys.argv[9])
		vary_inc = int(sys.argv[10])
		vary_xc = int(sys.argv[11])
		vary_yc = int(sys.argv[12])
		vary_vsys = int(sys.argv[13])
		vary_phib = 1

		# Rings configuration
		ring_space	= float(sys.argv[14])
		delta		= sys.argv[15]
		rstart_rfinal =  sys.argv[16]
		rstart_rfinal = rstart_rfinal.split(',')
		rstart,rfinal =	rstart_rfinal
		rstart = eval(rstart)
		frac_pixel = eval(sys.argv[17])

		# Kinematic model, minimization method and iterations
		vmode = sys.argv[18]

		#valid optional-string-inputs (osi):
		osi = ["-", ".", "#","%", "&", "","None"]

		r_bar_min_max,config_file,prefix = "","",""
		C, G = "C", "G"
		try:
			if sys.argv[19] not in osi: r_bar_min_max =  eval(sys.argv[19])
			if sys.argv[20] not in osi: config_file = sys.argv[20]
			if sys.argv[21] not in osi: prefix = sys.argv[21]
		except(IndexError): pass

		if config_file in osi:
			config_file = CONFIG_PATH
			print("XookSuut: No config file has been passed. Using default configuration file ..")


		if delta in osi:
			delta_tmp = ring_space/2.
		else:
			delta_tmp = float(delta)		
		delta	= np.clip(delta_tmp,0,ring_space)

		if r_bar_min_max in osi: r_bar_min_max = np.inf
		if vmode not in ["circular","radial","bisymmetric","vertical", "ff", 'mock'] and "hrm_" not in vmode: print("XookSuut: choose a proper kinematic model !"); quit()


		if type(r_bar_min_max)  == tuple:
			bar_min_max = [r_bar_min_max[0], r_bar_min_max[1] ]
		else:
			bar_min_max = [rstart, r_bar_min_max ]

		if prefix != "": galaxy = "%s-%s"%(galaxy,prefix)


		input_config = configparser.ConfigParser(
			# allow a variables be set without value
			allow_no_value=True,
			# allows duplicated keys in different sections
			strict=False,
			# deals with variables inside configuratio file
			interpolation=configparser.ExtendedInterpolation())
		input_config.read(config_file)

		# Shortcuts to the different configuration sections variables.
		config_const	= input_config['fitting']
		config_general	= input_config['general']

		n_it	= config_const.getint("n_it", 0)
		phi_b	= 45
		config	= input_config


		if vmode == 'mock':
			from .mock_cube import mock
			mock(vsys_0, pa_0, inc_0, vary_pa, vary_inc,
			vary_xc, vary_yc, vary_vsys,
			delta, rstart, rfinal, ring_space, vmode, config
			)
			
		v_center = False # <-- depreciated variable
			
		x	= XS_out(galaxy, datacube, mask, vsys_0, pa_0, inc_0, xc_0, yc_0, phi_b, n_it,
		vary_pa, vary_inc, vary_xc, vary_yc, vary_vsys, vary_phib, delta, rstart, rfinal,
		ring_space, frac_pixel, v_center, bar_min_max, vmode, config, prefix, osi)
		
		out_xs = x.results()

	def __str__(self,txt):
		print(txt)

if __name__ == "__main__":
	init = input_params('Bye !')
