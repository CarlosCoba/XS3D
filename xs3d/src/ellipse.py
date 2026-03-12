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

def drawrectangle(x0,y0,bmajor,pa_deg,bminor = None,eps = None):
	A = bmajor
	if bminor is not None:
		B = bminor
	if eps is not None:
		B = A*(1-eps)

	hh = A
	hw = B 
	corners_unrotated = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])

	# Create a rotation matrix

	t = pa_deg % 180
	cos_theta = np.cos(t*np.pi/180)
	sin_theta = np.sin(t*np.pi/180)
	R = np.array([[cos_theta, -sin_theta],
		[sin_theta,  cos_theta]])

	# Rotate the corners
	corners_rotated = np.dot(corners_unrotated, R.T)

	# Translate the rotated corners to the actual center
	corners_final = corners_rotated + np.array([x0, y0])

	# Extract x and y coordinates for plotting
	x_coords = np.append(corners_final[:, 0], corners_final[0, 0])  # Close the rectangle
	y_coords = np.append(corners_final[:, 1], corners_final[0, 1])

	return x_coords, y_coords
