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

	'''
	if path.exists(main_dir) == False:
		os.mkdir(main_dir)

	if path.exists(path_plots) == False:
		os.mkdir(path_plots)

	if path.exists(path_models) == False:
		os.mkdir(path_models)
	'''

	# An FileExistsError is raised if xs3d is run in parallel,
	# since multiple jobs try to simultaneously create the
	# directories.
	# Implementing Christian suggestion 08/12/2026.
	# Setting exist_ok=True ignores FileExistsError if the
	# target directory already.
	os.makedirs(main_dir, exist_ok=True)
	os.makedirs(path_plots, exist_ok=True)
	os.makedirs(path_models, exist_ok=True)		

	return main_dir



