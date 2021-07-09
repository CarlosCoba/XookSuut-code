import numpy as np
import numpy as np
import matplotlib.pylab as plt
import scipy
import sys
import lmfit
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec
from weights_interp import weigths_w
 
def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R*pixel_scale




from kin_components import CIRC_MODEL
from kin_components import RADIAL_MODEL
from kin_components import BISYM_MODEL

def best_2d_model(vmode,shape,V, pa, inc, x0, y0, Vsys, rings_pos, ring_space,pixel_scale, phi_b = None):


	Vrot, Vrad, Vtan = V[0], V[1], V[2]
	nrings = len(rings_pos)
	n_annulus = nrings - 1  
	[ny,nx] = shape

	X = np.arange(0, nx, 1)
	Y = np.arange(0, ny, 1)
	XY_mesh = np.meshgrid(X,Y)

	r_n = Rings(XY_mesh,pa*np.pi/180,inc*np.pi/180,x0,y0,pixel_scale)
	interp_model = np.zeros((ny,nx))



	def evaluate_model(vmode,xy_mesh, Vrot_i, r_0, r_space = ring_space,Vrad_i = None, Vtan_i = None):
		if vmode == "circular":
			modl = (CIRC_MODEL(xy_mesh,Vrot_i,pa,inc,x0,y0))*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale)
			return modl,Vsys

		if vmode == "radial":
			modl = (RADIAL_MODEL(xy_mesh,Vrot_i,Vrad_i,pa,inc,x0,y0))*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale)
			return modl,Vsys

		if vmode == "bisymmetric":
			modl = (BISYM_MODEL(xy_mesh,Vrot_i,Vrad_i,pa,inc,x0,y0,Vtan_i,phi_b))*weigths_w(xy_mesh,shape,pa,inc,x0,y0,r_0,r_space,pixel_scale)
			return modl,Vsys



	if vmode == "circular":


			for N in range(n_annulus):

				mdl_ev = 0
				mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
				x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
				XY = (x,y)


				# We need two velocities to describe the interpolation in a ring
				for kk in range(2):
					Vrot_i = Vrot[N + kk]
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
					
						interp_model[mask1] = Vxy[1] + Vsys


						#(b)

				interp_model[mask] = mdl_ev + Vsys



			interp_model[interp_model == 0] = np.nan

			return interp_model



	if vmode == "radial":


			for N in range(n_annulus):

				mdl_ev = 0
				mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
				x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
				XY = (x,y)


				# We need two velocities to describe the interpolation in a ring
				for kk in range(2):
					Vrot_i = Vrot[N + kk]
					Vrad_i = Vrad[N + kk]
					Vxy,Vsys = evaluate_model(vmode, XY, r_0 = rings_pos[N], Vrot_i = Vrot_i, Vrad_i = Vrad_i)

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
						Vxy,Vsys = evaluate_model(vmode, XY1, r_0 = 0, r_space = r_space_0, Vrot_i = Vrot_i, Vrad_i = Vrad_i)
					
						interp_model[mask1] = Vxy[1] + Vsys


						#(b)

				interp_model[mask] = mdl_ev + Vsys



			interp_model[interp_model == 0] = np.nan

			return interp_model



	if vmode == "bisymmetric":


			for N in range(n_annulus):

				mdl_ev = 0
				mask = np.where( (r_n >= rings_pos[N] ) & (r_n < rings_pos[N+1]) )
				x,y = XY_mesh[0][mask], XY_mesh[1][mask] 
				XY = (x,y)


				# We need two velocities to describe the interpolation in a ring
				for kk in range(2):
					Vrot_i = Vrot[N + kk]
					Vrad_i = Vrad[N + kk]
					Vtan_i = Vtan[N + kk]
					Vxy,Vsys = evaluate_model(vmode, XY, r_0 = rings_pos[N], Vrot_i = Vrot_i, Vrad_i = Vrad_i, Vtan_i = Vtan_i)

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
						Vxy,Vsys = evaluate_model(vmode, XY1, r_0 = 0, r_space = r_space_0, Vrot_i = Vrot_i, Vrad_i = Vrad_i, Vtan_i = Vtan_i)
					
						interp_model[mask1] = Vxy[1] + Vsys


						#(b)

				interp_model[mask] = mdl_ev + Vsys



			interp_model[interp_model == 0] = np.nan

			return interp_model


