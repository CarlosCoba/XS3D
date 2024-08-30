import os
from os import path

current_directory = "."


def direc_out(config):

	config_gen = config['general']
	out_dir = config_gen.get('output_directory', "./")

	if not out_dir.endswith('/'):
		out_dir += '/'

	main_dir = "%sXS3D/"%out_dir
	path_models = "%smodels/"%main_dir
	path_plots = "%sfigures/"%main_dir

	if path.exists(main_dir) == False:
		os.mkdir(main_dir)

	if path.exists(path_plots) == False:
		os.mkdir(path_plots)

	if path.exists(path_models) == False:
		os.mkdir(path_models)

	return main_dir



