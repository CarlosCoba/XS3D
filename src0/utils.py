import os
	
def set_threads(ncores=None):
	if ncores is None:
		ncores=1
	ncores=int(ncores)
	os.environ["MKL_NUM_THREADS"] = str(ncores)
	os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) 
	os.environ["OMP_NUM_THREADS"] = str(ncores)	

