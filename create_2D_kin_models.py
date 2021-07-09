import numpy as np
import numpy as np
import matplotlib.pylab as plt
import scipy
import sys
from weights_interp import weigths_w
 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R*pixel_scale



def bidi_models(vmode,shape,V, pa, inc, x0, y0, Vsys, rings_pos, ring_space,pixel_scale, phi_b = None):


	Vrot, Vrad, Vtan = np.array(V[0]), np.array(V[1]), np.array(V[2])
	mask_zeros = Vrad != 0

	Vrad = Vrad[mask_zeros]
	Vtan = Vtan[mask_zeros]

	nrings = len(rings_pos)
	n_annulus = nrings - 1  
	[ny,nx] = shape

	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(X,Y)

	r_n = Rings(XY_mesh,pa*np.pi/180,inc*np.pi/180,x0,y0,pixel_scale)


	def evaluate_model(vmode,xy_mesh, Vrot_i, r_0, r_space = ring_space):

		modl = Vrot_i*np.ones(len(xy_mesh[0]),)*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale)
		return modl,Vsys



	def interpolation(V_k):

		nrings = len(V_k)
		n_annulus = nrings - 1 

		interp_model = np.zeros((ny,nx))
		for N in range(n_annulus):

			mdl_ev = 0
			mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
			x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
			XY = (x,y)


			# We need two velocities to describe the interpolation in a ring
			for kk in range(2):
				Vrot_i = V_k[N + kk]
				Vxy,Vsys = evaluate_model(vmode, XY, Vrot_i,  r_0 = rings_pos[N])

				mdl_ev = mdl_ev + Vxy[kk]


				if N == 0 and kk == 0:
						
					mask1 = np.where( (r_n < rings_pos[0] ) )
					x1,y1 = XY_mesh[0][mask1], XY_mesh[1][mask1] 
					XY1 = (x1,y1)


				#
				#
				# inner interpolation
				#
				#
						
				#(a) velocity rise linearly from zero

					r_space_0 =  rings_pos[0]
					Vxy,Vsys = evaluate_model(vmode, XY1, Vrot_i, r_0 = 0, r_space = r_space_0)
					
					interp_model[mask1] = Vxy[1] 

			interp_model[mask] = mdl_ev
			interp_model[interp_model == 0] = np.nan 

		return 	interp_model
		



	if vmode == "circular":

		Vcirc_2D = interpolation(Vrot)
		Vrad_2D = 0
		Vtan_2D = 0


	if vmode == "radial":

		Vcirc_2D = interpolation(Vrot)
		Vrad_2D = interpolation(Vrad)
		Vtan_2D = 0


	if vmode == "bisymmetric":

		Vcirc_2D = interpolation(Vrot)
		Vrad_2D = interpolation(Vrad)
		Vtan_2D = interpolation(Vtan)




	return Vcirc_2D, Vrad_2D, Vtan_2D, r_n




