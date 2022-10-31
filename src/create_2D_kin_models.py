import numpy as np
import scipy
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
from kin_components import HARMONIC_MODEL

class bidi_models:
	""" This class creates the 2D maps of the different kinematic componets. I performs a 1D linear interpolation between the best fit values """
	def __init__(self,vmode,shape,V_k, pa, inc, x0, y0, Vsys, ring_pos, ring_space,pixel_scale, inner_interp, m_hrm = 1, phi_b = None):


		self.vmode  =  vmode
		self.shape = shape
		self.pa, self.inc, self.x0, self.y0, self.Vsys =pa, inc, x0, y0, Vsys
		self.rings_pos = ring_pos
		self.ring_space = ring_space
		self.nrings = len(self.rings_pos)
		self.n_annulus = self.nrings - 1
		self.pixel_scale = pixel_scale
		self.phi_b = phi_b
		self.V = V_k
		self.m_hrm = m_hrm
		self.inner_interp = inner_interp

		if "hrm" not in self.vmode:
			self.Vrot, self.Vrad, self.Vtan = V_k
		else:
			self.m_hrm = int(m_hrm)
			self.m2_hrm = int(2*m_hrm)
			self.C_k, self.S_k = [ V_k[k] for k in range(self.m_hrm) ], [ V_k[k] for k in range(self.m_hrm,self.m2_hrm) ]

		[self.ny, self.nx] = shape
		X = np.arange(0, self.nx, 1)
		Y = np.arange(0, self.ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)
		self.r_n = Rings(self.XY_mesh,self.pa*np.pi/180,self.inc*np.pi/180,self.x0,self.y0,pixel_scale)



	def evaluate_model(self,xy_mesh,V_i,r_0,r_space):
			modl = V_i*np.ones(len(xy_mesh[0]),)*weigths_w(xy_mesh,self.pa,self.inc,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
			return modl,self.Vsys


	def v_interp(self,r, r2, r1, v2, v1 ):
		m = (v2 - v1) / (r2 - r1)
		v0 = m*(r-r1) + v1
		return v0


	def mdl(self,V_i, v_t = False):
			"""Calculate total residual for fits of VMODELS to several data sets."""
			interp_model = np.zeros((self.ny,self.nx))
			# make residual per data set
			for Nring in range(self.n_annulus):
				# For r1 > rings_pos[0]
				mdl_ev = 0
				r_space_k = self.rings_pos[Nring+1] - self.rings_pos[Nring] 
				mask = np.where( (self.r_n >= self.rings_pos[Nring] ) & (self.r_n < self.rings_pos[Nring+1]) )
				x,y = self.XY_mesh[0][mask], self.XY_mesh[1][mask] 

				# For r1 < rings_pos[0]
				mdl_ev0 = 0
				mask_inner = np.where( (self.r_n < self.rings_pos[0] ) )
				x_r0,y_r0 = self.XY_mesh[0][mask_inner], self.XY_mesh[1][mask_inner] 
				r_space_0 =  self.rings_pos[0]

				# interpolation between rings requieres two velocities: v1 and v2
				#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
				for kring in np.arange(2,0,-1):
					r2_1 = self.rings_pos[Nring + kring -1 ]
					v1_2_index = Nring + 2 - kring

					# For r > rings_pos[0]:					
					Vrot_i = V_i[Nring + 2 - kring]
					Vxy,Vsys = self.evaluate_model( (x,y), Vrot_i, r_0 = r2_1, r_space = r_space_k )
					mdl_ev = mdl_ev + Vxy[2-kring]

					# For r < rings_pos[0]:					
					# Inner interpolation
					#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
					if self.inner_interp == False and Nring == 0 and kring == 2:
						Vxy,Vsys = self.evaluate_model((x_r0,y_r0), V_i[0], r_0 = 0, r_space = r_space_0)
						mdl_ev0 = Vxy[1]
						interp_model[mask_inner] = mdl_ev0 
					#(b) Extrapolate at the origin: r1 = 0, v1 != 0
					if self.inner_interp != False and Nring == 0:
						# we need to add a velocity at r1 = 0
						r2, r1, v2, v1 = self.rings_pos[1],self.rings_pos[0], V_i[1], V_i[0]
						# This is the extrapolated velocity 
						v_int = self.v_interp(0, r2, r1, v2, v1 )
						if self.inner_interp != True and v_t == True:
							v_int = self.inner_interp
						Vrot_inner = [v_int, V_i[0]]
						r2_1 = [r_space_0, 0]

						Vxy,Vsys = self.evaluate_model( (x_r0,y_r0), Vrot_inner[2-kring], r_0 = r2_1[2-kring], r_space = r_space_0)
						mdl_ev0 = mdl_ev0 + Vxy[2-kring]
						interp_model[mask_inner] = mdl_ev0 



				interp_model[mask] = mdl_ev 

			interp_model[interp_model == 0]  = np.nan
			return interp_model




	def interp(self):
			if self.vmode == "circular":

				Vcirc_2D = self.mdl(self.Vrot, v_t = True)
				Vrad_2D = 0
				Vtan_2D = 0


			if self.vmode == "radial":

				Vcirc_2D = self.mdl(self.Vrot, v_t = True)
				Vrad_2D = self.mdl(self.Vrad)
				Vtan_2D = 0


			if self.vmode == "bisymmetric":

				Vcirc_2D = self.mdl(self.Vrot, v_t = True)
				Vrad_2D = self.mdl(self.Vrad)
				Vtan_2D = self.mdl(self.Vtan)


			if "hrm" in self.vmode:
				C = [self.mdl(self.C_k[k]) for k in range(self.m_hrm)]
				S = [self.mdl(self.S_k[k]) for k in range(self.m_hrm)]

			if "hrm" not in self.vmode:
				return Vcirc_2D, Vrad_2D, Vtan_2D, self.r_n
			else:
				return C, S, self.r_n








