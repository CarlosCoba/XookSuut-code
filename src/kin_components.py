import numpy as np
from src.weights_interp import weigths_w
from src.pixel_params import Rings


def myatan(x,y):
	m = np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
         -0.25*(2+np.sign(x))*np.sign(y))\
         -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))

	m = np.pi/2. - m
	for i,j in enumerate(m):
		if m[i] < -np.pi : m[i] = m[i] + 2*np.pi
	return m


def myatan2D(x,y):
	m = np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
         -0.25*(2+np.sign(x))*np.sign(y))\
         -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))

	m = np.pi/2. - m
	mask = m < -np.pi
	m[mask] = m[mask] +  2*np.pi
	return m

def AZIMUTHAL_ANGLE(shape,pa,inc,x0,y0):

	[ny,nx] = shape
	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	xy_mesh = np.meshgrid(X,Y)
	(x,y) = xy_mesh

	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	theta = myatan2D(sin_theta,cos_theta)
	return theta


def CIRC_MODEL(xy_mesh,Vrot,pa,inc,x0,y0):
	(x,y) = xy_mesh
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	vlos = np.sin(inc)*(Vrot*cos_theta)
	return np.ravel(vlos)


def RADIAL_MODEL(xy_mesh,Vrot,Vr2,pa,inc,x0,y0):
	(x,y) = xy_mesh
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	vlos = np.sin(inc)*(Vrot*cos_theta + Vr2*sin_theta)
	return np.ravel(vlos)


def BISYM_MODEL(xy_mesh,Vrot,Vrad,pa,inc,x0,y0,Vtan,phi_b):
	(x,y) = xy_mesh
	pa,inc,phi_b=(pa)*np.pi/180,inc*np.pi/180,phi_b*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)

	m = 2
	theta = np.arctan(sin_theta/cos_theta)
	theta = myatan(sin_theta,cos_theta)
	theta_b = theta - phi_b
	
	vlos = np.sin(inc)*((Vrot*cos_theta) - Vtan*np.cos(m*theta_b)*cos_theta - Vrad*np.sin(m*theta_b)*sin_theta)

	return np.ravel(vlos)



def HARMONIC_MODEL(xy_mesh,c_k,s_k,pa,inc,x0,y0,m_hrm):
	(x,y) = xy_mesh
	pa,inc=pa*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_theta = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_theta = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)

	theta_c = np.arccos(cos_theta)
	theta_s = np.arcsin(sin_theta)


	#C_k = np.array(c_k)
	#S_k = np.array(s_k)

	v_k = 0
	for k in range(1,m_hrm+1):
		v_k = v_k + c_k[k-1]*np.sin(inc)*np.cos(k*theta_c) + s_k[k-1]*np.sin(inc)*np.sin(k*theta_s)
		#v_k = v_k + C_k*np.sin(inc)*np.cos(k*theta_c) + S_k*np.sin(inc)*np.sin(k*theta_s)

	return np.ravel(v_k)






