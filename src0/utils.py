import os
import numpy as np
	
def set_threads(ncores=None):
	if ncores is None:
		ncores=1
	ncores=int(ncores)
	os.environ["MKL_NUM_THREADS"] = str(ncores)
	os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) 
	os.environ["OMP_NUM_THREADS"] = str(ncores)	


def parabola(x,y):
	#(x1,y1),(x2,y2),(x3,y3)=p1,p2,p3
	x1,x2,x3=x[:]
	y1,y2,y3=y[:]
			
	# f(x)=A*x**2 + B*x +c	
	denom = (x1 - x2)*(x1 - x3)*(x2 - x3)
	A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
	B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
	C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

	# the vertex:
	xmax = -B/(2*A)
	ymax =  C-(B**2)/(4*A)

	#print(A,B,C)	
	return xmax, ymax

