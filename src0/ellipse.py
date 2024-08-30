import numpy as np

def drawellipse(x0,y0,bmajor,pa_deg,bminor=None,eps=None):
	t=np.linspace(-2*np.pi,2*np.pi,100)
	A=bmajor
	if bminor is not None:
		B=bminor
	if eps is not None:
		B=A*(1-eps)
	
	pa_deg+=90
	pa=pa_deg*np.pi/180
	x=x0+A*np.cos(pa)*np.cos(t)-B*np.sin(pa)*np.sin(t)
	y=y0+A*np.sin(pa)*np.cos(t)+B*np.cos(pa)*np.sin(t)
	return x,y	
