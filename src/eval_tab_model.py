import numpy as np
from src.pixel_params import pixels
from src.M_tabulated import M_tab

def tab_mod_vels(rings, vel, evel, pa, inc, x0, y0, vsys,theta_b, delta, pixel_scale, vmode, shape, frac_pixel, r_bar_min, r_bar_max, m_hrm = 1):

				vrot_tab,vrad_tab,vtan_tab = np.asarray([]),np.asarray([]),np.asarray([])
				c1_tab, c3_tab, s1_tab, s3_tab = np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])

				#C, S = [], []
				for j in range(1,m_hrm+1):
					globals()['C%s_tab' % (j)] = np.asarray([])
					globals()['S%s_tab' % (j)] = np.asarray([])

				nrings = len(rings)			
				R_pos = np.asarray([])
				index = 0
				for ring in rings:


					fpix = pixels(shape,vel,pa,inc,x0,y0,ring, delta=delta,pixel_scale = pixel_scale)
					# If the last ring, increase data points
					if ring == rings[-1]: delta = 2*delta
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


							if ring >=  r_bar_min and ring <= r_bar_max:
								# Create radial model
								v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = vmode)

								vrot_tab = np.append(vrot_tab,v_rot_k)
								vrad_tab = np.append(vrad_tab,v_rad_k)
								vtan_tab = 0
								R_pos = np.append(R_pos,ring)
							else:
								# Create ciruclar model
								v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = "circular")

								vrot_tab = np.append(vrot_tab,v_rot_k)
								vrad_tab = np.append(vrad_tab,0)
								vtan_tab = np.append(vtan_tab,0)
								R_pos = np.append(R_pos,ring)



						if vmode == "circular":


							v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = vmode)

							vrot_tab = np.append(vrot_tab,v_rot_k)
							vrad_tab = 0
							vtan_tab = 0
							R_pos = np.append(R_pos,ring)



						if "hrm" in vmode:


							if ring >=  r_bar_min and ring <= r_bar_max:
								# Create m2 model
								c_k, s_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = vmode, m_hrm = m_hrm )
								n = len(c_k)
								for j in range(1,m_hrm+1):
									globals()['C%s_tab' % (j)] = np.append( globals()['C%s_tab' % (j)], c_k[j-1] )
									globals()['S%s_tab' % (j)] = np.append( globals()['S%s_tab' % (j)], s_k[j-1] )
								R_pos = np.append(R_pos,ring)
	

							else:

								v_rot_k,v_rad_k,v_tan_k = M_tab(pa,inc,x0,y0,theta_b,ring, delta,index, shape, vel-vsys, evel, pixel_scale=pixel_scale,vmode = "circular")

								globals()['C1_tab'] = np.append(globals()['C1_tab'],v_rot_k)

								k = 1
								for j in range(1,m_hrm+1):
									if k != m_hrm:
										globals()['C%s_tab' % (k+1)] = np.append( globals()['C%s_tab' % (k+1)], 0 )
									globals()['S%s_tab' % (j)] = np.append( globals()['S%s_tab' % (j)], 0 )
									k = k + 1


								R_pos = np.append(R_pos,ring)


				if "hrm" not in vmode:
					return vrot_tab, vrad_tab, vtan_tab, R_pos
				else:
					return [globals()['C%s_tab' % (j)] for j in range(1,m_hrm+1)], [globals()['S%s_tab' % (j)] for j in range(1,m_hrm+1)], R_pos


						
