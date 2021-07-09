import numpy as np
from pixel_params import pixels
from M_tabulated import M_tab

def tab_mod_vels(rings, vel, evel, pa,inc,x0,y0,vsys,theta_b,delta,pixel_scale,vmode,shape, frac_pixel,r_bar_min, r_bar_max):


				vrot_tab,vrad_tab,vtan_tab = np.asarray([]),np.asarray([]),np.asarray([])
				nrings = len(rings)			
				R_pos = np.asarray([])
				index = 0
				for ring in rings:


					fpix = pixels(shape,vel,pa,inc,x0,y0,ring, delta=delta,pixel_scale = pixel_scale)

					if fpix > frac_pixel:
						if vmode == "bisymmetric":

							if ring >=  r_bar_min and ring <= r_bar_max:

									# Create bisymetric model
								try:

									v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = vmode)

									vrot_tab = np.append(vrot_tab,v_rot_k)
									vrad_tab = np.append(vrad_tab,v_rad_k)
									vtan_tab = np.append(vtan_tab,v_tan_k)
									R_pos = np.append(R_pos,ring)


								except(np.linalg.LinAlgError):

										vrot_tab,vrad_tab,vtan_tab = np.append(vrot_tab,100),np.append(vrad_tab,10),np.append(vtan_tab,10)							
										R_pos = np.append(R_pos,ring)
							else:

									# Create ciruclar model
									v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = "circular")
									vrot_tab = np.append(vrot_tab,v_rot_k)
									vrad_tab = np.append(vrad_tab,0)
									vtan_tab = np.append(vtan_tab,0)
									R_pos = np.append(R_pos,ring)
					#return vrot_tab, vrad_tab, vtan_tab, R_pos




						if vmode == "radial":


							v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = vmode)

							vrot_tab = np.append(vrot_tab,v_rot_k)
							vrad_tab = np.append(vrad_tab,v_rad_k)
							vtan_tab = 0
							R_pos = np.append(R_pos,ring)



						if vmode == "circular":


							v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = vmode)

							vrot_tab = np.append(vrot_tab,v_rot_k)
							vrad_tab = 0
							vtan_tab = 0
							R_pos = np.append(R_pos,ring)


				return vrot_tab, vrad_tab, vtan_tab, R_pos


						
