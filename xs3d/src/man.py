from textwrap import fill

def print_manual():

	print()
	print("USE:")
	print("  XS3D name cube.fits [mask] [pa] [inc] [xc] [yc] [vsys] "
		  "vary_pa vary_inc vary_xc vary_yc vary_vsys ring_space "
		  "[delta] rstart,rfinal cover kin_model "
		  "[r_nc_min,r_nc_max] config_file [prefix]")

	print()
	print("Variables in brackets are optional but must still be declared.")
	print("Use `None` or `-` for optional parameters that are omitted.")
	print()

	p_width = 18
	o_width = 10
	d_width = 52

	header = (
		f"{'Parameter':<{p_width}}"
		f"{'Optional':<{o_width}}"
		f"{'Description'}"
	)

	print(header)
	print("-" * (p_width + o_width + d_width))

	rows = [
		("name", "N",
		 "Name used for all saved products."),

		("cube.fits", "N",
		 "Input data cube in FITS format."),

		("mask", "Y",
		 "Mask in FITS format. It may be either a 2D or 3D mask."),

		("pa", "Y",
		 "Initial estimate of the kinematic position angle (degrees)."),

		("inc", "Y",
		 "Initial estimate of the disk inclination (degrees)."),

		("xc", "Y",
		 "Pixel coordinate of the kinematic center."),

		("yc", "Y",
		 "Pixel coordinate of the kinematic center."),

		("vsys", "Y",
		 "Initial estimate of the systemic velocity (km/s)."),

		("vary_pa", "N",
		 "Whether the position angle is varied during the fit. "
		 "Options: 0 = fixed, 1 = variable but constant with radius, "
		 "2 = variable at every ring."),

		("vary_inc", "N",
		 "Same options as vary_pa."),

		("vary_xc", "N",
		 "Same options as vary_pa."),

		("vary_yc", "N",
		 "Same options as vary_pa."),

		("vary_vsys", "N",
		 "Same options as vary_pa."),

		("ring_space", "N",
		 "Spacing between rings (arcseconds)."),

		("delta", "Y",
		 "Width of the sub-rings used for interpolation (arcseconds). "
		 "Default: ring_space/2. All fitted quantities are interpolated "
		 "every delta arcseconds."),

		("rstart,rfinal", "N",
		 "Radius of the first and last ring (arcseconds)."),

		("cover", "N",
		 "Minimum fraction of valid pixels required for a ring to be "
		 "included in the analysis. Valid values range from 0 to 1, "
		 "where 1 requires complete coverage. A value of 0.5 is often "
		 "a good choice."),

		("kin_model", "N",
		 "Kinematic model. Choose from 'circular', 'radial', or "
		 "'bisymmetric'. Harmonic decomposition models use the form "
		 "'hrm_m', where m is the harmonic order."),

		("r_nc_min,r_nc_max", "Y",
		 "Minimum and maximum radii over which non-circular motions are "
		 "computed. If only one value is supplied, it is interpreted as "
		 "r_nc_max."),

		("config_file", "N",
		 "Configuration file containing information about the emission "
		 "line, beam smearing, and other parameters. The file must be "
		 "located in the current working directory. A template is "
		 "provided in xs3d/src/config_file/xs_conf.ini."),

		("prefix", "Y",
		 "Prefix added to all saved products.")
	]

	for par, opt, desc in rows:
		wrapped = fill(desc, width=d_width).split("\n")
		print(f"{par:<{p_width}}{opt:<{o_width}}{wrapped[0]}")
		for line in wrapped[1:]:
			print(f"{'':<{p_width}}{'':<{o_width}}{line}")

	print()
	print("Example")
	print("-------")
	print("XS3D NGC1087 NGC1087.fits None - - - - -  "
		  "1 1 1 1 1 4 - 0,100 1/2 radial 0,50 n1087.ini")
	print("-------")
	print()			  
  
