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

from src0.utils import *
set_threads(nthreads)

import numpy as np
from src0.initialize_XS3D_main import XS_out
from src0.pixel_params import eps_2_inc
"""
#################################################
# 				XookSuut3D (XS3D)					#
# 				C. Lopez-Coba					#
#################################################

"""


class input_params:
	def __init__(self):
		if (nargs < 19 or nargs > 22):

			print ("USE: XS3D name cube.fits [mask2D] [PA] [INC] [X0] [Y0] [VSYS] vary_PA vary_INC vary_X0 vary_Y0 vary_VSYS ring_space [delta] Rstart,Rfinal cover kin_model [R_bar_min,R_bar_max] [config_file] [prefix]" )

			exit()

		#object name
		galaxy = sys.argv[1]

		#FITS information
		vel_map = sys.argv[2]
		mask2D = sys.argv[3]
		#pixel_scale = float(sys.argv[4])

		# Geometrical parameters
		PA = sys.argv[4]
		INC = sys.argv[5]
		#if 0<INC<1: INC=eps_2_inc(INC)*180/np.pi
		X0 = sys.argv[6]
		Y0 = sys.argv[7]
		VSYS = sys.argv[8]
		vary_PA = bool(float(sys.argv[9]))
		vary_INC = bool(float(sys.argv[10]))
		vary_XC = bool(float(sys.argv[11]))
		vary_YC = bool(float(sys.argv[12]))
		vary_VSYS = bool(float(sys.argv[13]))
		vary_PHIB = 1

		# Rings configuration
		ring_space = float(sys.argv[14])
		delta = sys.argv[15]
		rstart, rfinal =  eval(sys.argv[16])
		frac_pixel = eval(sys.argv[17])

		# Kinematic model, minimization method and iterations
		vmode = sys.argv[18]
		#fit_method = sys.argv[20]
		#n_it = int(sys.argv[21])


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
			delta = ring_space/2. 
		else:
			delta = float(delta)

		if r_bar_min_max in osi: r_bar_min_max = np.inf
		if vmode not in ["circular","radial","bisymmetric","vertical"] and "hrm_" not in vmode: print("XookSuut: choose a proper kinematic model !"); quit()
	

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
		config_const = input_config['constant_params']
		config_general = input_config['general']

		PA = config_const.get('PA', PA)
		INC = config_const.get('INC', INC)
		X0 = config_const.get('X0', X0)
		Y0 = config_const.get('Y0', Y0)
		VSYS = config_const.get('VSYS', VSYS)
		PHI_BAR = config_const.getfloat('PHI_BAR', 45)

		vary_PA = config_const.getboolean('FIT_PA', vary_PA)
		vary_INC = config_const.getboolean('FIT_INC', vary_INC)
		vary_XC = config_const.getboolean('FIT_X0', vary_XC)
		vary_YC = config_const.getboolean('FIT_Y0', vary_YC)
		vary_VSYS = config_const.getboolean('FIT_VSYS', vary_VSYS)
		vary_PHIB = config_const.getboolean('FIT_PHI_BAR', vary_PHIB)

		n_it=config_general.getint("n_it", 1)
		v_center = config_general.get("v_center", 0)  
		survey = config_general.get("dataset", "-")
		
		
		
		try:
			v_center = float(v_center)
		except (ValueError): pass

		if type(v_center) == str and v_center != "extrapolate":
			print("XookSuut: v_center: %s, did you mean ´extrapolate´ ?"%v_center)
			quit()


		config = input_config		

		x = XS_out(galaxy, vel_map, mask2D, VSYS, PA, INC, X0, Y0, PHI_BAR, n_it, vary_PA, vary_INC, vary_XC, vary_YC, vary_VSYS, vary_PHIB, delta, rstart, rfinal, ring_space, frac_pixel, v_center, bar_min_max, vmode, survey, config, prefix, osi  )
		out_xs = x()

if __name__ == "__main__":
	init = input_params()


