import numpy as np
import numpy as np
import matplotlib.pylab as plt
import sys
import lmfit
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec


 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	(x,y) = xy_mesh
	#X = -(x-x0)*np.sin(PA)+(y-y0)*np.cos(PA)
	#Y = ((x-x0)*np.cos(PA)+(y-y0)*np.sin(PA))/np.cos(inc)

	X = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))
	Y = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))



	R= np.sqrt(X**2+(Y/np.cos(inc))**2)
	return R*pixel_scale

def Rings0(xy_mesh,pa,inc,x0,y0):
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	(x,y) = xy_mesh
	X = -(x-x0)*np.sin(PA)+(y-y0)*np.cos(PA)
	Y = ((x-x0)*np.cos(PA)+(y-y0)*np.sin(PA))/np.cos(inc)
	R= np.sqrt(X**2+Y**2)
	return R



def Rings_r_1(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
		#mask = R==0
	else:
		#ring = ring +1
		ring = ring +delta
		mask = (R>=ring) & (R <= ring+delta)		
	
	s = M*mask
	return s



def Rings_r_2(R,vel,ring,delta=0,pixel_scale=1):
	R = R*pixel_scale
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
		#mask = R==0
	else:
		#ring = ring +1
		ring = ring*delta +delta
		mask = (R>=ring) & (R <= ring+delta)		
	
	s = M*mask
	#plt.imshow(s)
	#plt.show()
	return s


def Rings_r_3(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
	else:
		mask = (R>=ring) & (R <= ring+delta)		
	
	s = M*mask
	return s


def Rings_r_4(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
	else:
		mask = (R>=ring-0.5*delta) & (R <= ring+0.5*delta)		
	
	s = M*mask
	#plt.imshow(s)
	#plt.show()
	return s


def Ring_arcsec(R,vel,ring,delta=0):
	R = R
	if delta == 0:
		R = R.astype(int)


	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if delta == 0:
		R = R.astype(int)
		mask = R== ring

	else:

		mask = (R>=ring-delta) & (R < ring+delta)



	s = M*mask

	return s



def Vlos_BISYM(xy_mesh,Vrot,Vr2,pa,inc,x0,y0,Vsys):
	(x,y) = xy_mesh
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	sin_tetha = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))/(np.cos(inc)*R)
	#vlos =  Vrot*cos_tetha*np.sin(inc)+Vsys
	#m = 2
	#vlos = Vsys+np.sin(inc)*((Vrot*cos_tetha)-Vt2*np.cos(m*theta_b*np.pi/180)*cos_tetha - Vr2*np.sin(m*theta_b*np.pi/180)*sin_tetha)
	vlos = Vsys+np.sin(inc)*(Vrot*cos_tetha + Vr2*sin_tetha)
	return np.ravel(vlos)


def Vlos_ROT(xy_mesh,Vrot,pa,inc,x0,y0,Vsys):#,Vt2=0,Vr2=0,theta_b=0):
	(x,y) = xy_mesh
	R  = Rings0(xy_mesh,pa,inc,x0,y0)
	#print "R=",R
	PA,inc=(pa-90*0)*np.pi/180,inc*np.pi/180
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	sin_tetha = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))/(np.cos(inc)*R)
	vlos =  Vrot*cos_tetha*np.sin(inc)+Vsys
	return np.ravel(vlos)



def Vrot_MODEL(xy_mesh,vlos,pa,inc,x0,y0,Vsys):#,Vt2=0,Vr2=0,theta_b=0):
	(x,y) = xy_mesh
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	PA,inc=(pa-90*0)*np.pi/180,inc*np.pi/180
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	sin_tetha = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))/(np.cos(inc)*R)
	vrot = (vlos - Vsys)/cos_tetha*np.sin(inc)
	return vrot


def Weight(xy_mesh,Vrot,pa,inc,x0,y0,Vsys):
	(x,y) = xy_mesh
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings0(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	abs_cos = abs(cos_tetha) 
	return np.ravel(abs_cos)




#######################################################3


def ring_pixels(xy_mesh,pa,inc,x0,y0,ring,delta,pixel_scale):

	r_n = Rings(xy_mesh,pa,inc,x0,y0,pixel_scale)
	a_k = ring


	mask = np.where( (r_n >= a_k - delta) & (r_n < a_k + delta) ) 
	r_n = r_n[mask]

	return mask



 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))
	Y = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)
	return R*pixel_scale



def pixels(shape,vel,pa,inc,x0,y0,ring, delta=1,pixel_scale = 1):

	[ny,nx] = shape


	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)

	XY_mesh = np.meshgrid(x,y,sparse=True)
	r_pixels_mask = ring_pixels(XY_mesh,pa,inc,x0,y0,ring,delta,pixel_scale)
	mask = r_pixels_mask

	indices = np.indices((ny,nx))
	pix_y=  indices[0]
	pix_x=  indices[1]


	indices = np.indices((ny,nx))


	pix_y=  indices[0]
	pix_x=  indices[1]


	pix_x = pix_x[mask]
	pix_y = pix_y[mask]



	vel_pixesl = vel[mask]	
	pix_y =  np.asarray(pix_y)
	pix_x = np.asarray(pix_x)
	npix_exp = len(pix_x)



	mask = np.isfinite(vel_pixesl) == True
	vel_val = vel_pixesl[mask]
	len_vel = len(vel_val)


	"""
	plt.imshow(vel, origin = "lower")
	plt.plot(pix_x,pix_y,"ko")
	plt.show()
	"""
	if npix_exp >0 and len_vel >0 :
		f_pixel = len(vel_val)/(1.0*npix_exp)
	else:
		f_pixel = 0


	return f_pixel







