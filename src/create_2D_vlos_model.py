import numpy as np
import matplotlib.pylab as plt
import scipy
from weights_interp import weigths_w


def Rings(xy_mesh,pa,inc,x0,y0,pixel_scale):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))

	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R*pixel_scale


def v_interp(r, r2, r1, v2, v1 ):
	m = (v2 - v1) / (r2 - r1)
	v0 = m*(r-r1) + v1
	return v0



from kin_components import CIRC_MODEL
from kin_components import RADIAL_MODEL
from kin_components import BISYM_MODEL
from kin_components import HARMONIC_MODEL

class best_2d_model:
	def __init__(self,vmode,shape,V_k, pa, inc, x0, y0, Vsys, ring_pos, ring_space, pixel_scale, inner_interp, m_hrm = 1, phi_b = None):
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
			self.Vrot, self.Vrad, self.Vtan = np.append(self.Vrot, -1e4), np.append(self.Vrad, -1e4), np.append(self.Vtan, -1e4)
		else:
			self.m_hrm = int(m_hrm)
			self.m2_hrm = int(2*m_hrm)
			self.C_k, self.S_k = [ V_k[k] for k in range(self.m_hrm) ], [ V_k[k] for k in range(self.m_hrm,self.m2_hrm) ]



		[ny, nx] = shape
		X = np.arange(0, nx, 1)
		Y = np.arange(0, ny, 1)
		self.XY_mesh = np.meshgrid(X,Y)
		self.r_n = Rings(self.XY_mesh,self.pa*np.pi/180,self.inc*np.pi/180,self.x0,self.y0,pixel_scale)
		self.interp_model = np.zeros((ny,nx))

	def vmodel_dataset(self, i, xy_mesh, r_space, r_0 = None):
		# For inner interpolation
		r1, r2 = self.rings_pos[0], self.rings_pos[1]
		if "hrm" not in self.vmode and self.inner_interp != False and i == -1:
			v1, v2 = self.Vrot[0], self.Vrot[1]
			v_int =  v_interp(0, r2, r1, v2, v1 )
			# This only applies to circular rotation
			if self.inner_interp != True :
				v_int = self.inner_interp
			# I made up this index.
			self.Vrot[-1] = v_int
			if self.vmode == "radial" or self.vmode == "bisymmetric":
				v1, v2 = self.Vrad[0], self.Vrad[1]
				v_int =  v_interp(0, r2, r1, v2, v1 )
				self.Vrad[-1] = v_int
			if self.vmode == "bisymmetric":
				v1, v2 = self.Vtan[0], self.Vtan[1]
				v_int =  v_interp(0, r2, r1, v2, v1 )
				self.Vtan[-1] = v_int

		if "hrm" not in self.vmode:
			Vrot = self.Vrot[i]

		if self.vmode == "circular":
			modl = (CIRC_MODEL(xy_mesh,Vrot,self.pa,self.inc,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.inc,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
		if self.vmode == "radial":
			Vrad = self.Vrad[i]
			modl = (RADIAL_MODEL(xy_mesh,Vrot,Vrad,self.pa,self.inc,self.x0,self.y0))*weigths_w(xy_mesh,self.pa,self.inc,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
		if self.vmode == "bisymmetric":
			Vrad = self.Vrad[i]
			Vtan = self.Vtan[i]
			if Vrad != 0 and Vtan != 0:
				phi_b = self.phi_b
				modl = (BISYM_MODEL(xy_mesh,Vrot,Vrad,self.pa,self.inc,self.x0,self.y0,Vtan,phi_b))*weigths_w(xy_mesh,self.pa,self.inc,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
			else:
				modl = (BISYM_MODEL(xy_mesh,Vrot,0,self.pa,self.inc,self.x0,self.y0,0,0))*weigths_w(xy_mesh,self.pa,self.inc,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)
		if "hrm" in self.vmode:
			c_k, s_k  = [self.C_k[k][i] for k in range(self.m_hrm)] , [self.S_k[k][i] for k in range(self.m_hrm)]
			modl = (HARMONIC_MODEL(xy_mesh,c_k, s_k,self.pa,self.inc,self.x0,self.y0, self.m_hrm))*weigths_w(xy_mesh,self.pa,self.inc,self.x0,self.y0,r_0,r_space,pixel_scale=self.pixel_scale)

		return modl,self.Vsys


	def model2D(self):
			"""Calculate total residual for fits of VMODELS to several data sets."""

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
				r_space_0 = self.rings_pos[0]

				 
				# interpolation between rings requieres two velocities: v1 and v2
				#v_new = v1*(r2-r)/(r2-r1) + v2*(r-r1)/(r2-r1)
				for kring in np.arange(2,0,-1):
					r2_1 = self.rings_pos[Nring + kring -1 ]
					v1_2_index = Nring + 2 - kring

					# For r > rings_pos[0]:					
					Vxy,Vsys = self.vmodel_dataset(v1_2_index, (x,y), r_0 = r2_1, r_space = r_space_k )
					mdl_ev = mdl_ev + Vxy[2-kring]

					# For r < rings_pos[0]:					
					# Inner interpolation
					#(a) velocity rises linearly from zero: r1 = 0, v1 = 0
					if self.inner_interp == False and Nring == 0 and kring == 2:
						Vxy,Vsys = self.vmodel_dataset(0, (x_r0,y_r0), r_0 = 0, r_space = r_space_0)
						mdl_ev0 = Vxy[1]
						self.interp_model[mask_inner] = mdl_ev0
					#(b) Extrapolate at the origin: r1 = 0, v1 != 0
					if self.inner_interp != False and Nring == 0:
						# we need to add a velocity at r1 = 0
						# index -1 is not in the velocity dictonary. I made up !
						r2_1 = [r_space_0,0]
						v1_2_index = [-1,0]

						Vxy,Vsys = self.vmodel_dataset(v1_2_index[2-kring], (x_r0,y_r0), r_0 = r2_1[2-kring], r_space = r_space_0 )
						mdl_ev0 = mdl_ev0 + Vxy[2-kring]
						self.interp_model[mask_inner] = mdl_ev0


				self.interp_model[mask] = mdl_ev

			self.interp_model[self.interp_model == 0]  = np.nan
			self.interp_model = self.interp_model + Vsys
			return self.interp_model



