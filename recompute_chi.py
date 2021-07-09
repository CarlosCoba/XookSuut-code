import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pylab as plt
np.warnings.filterwarnings('ignore')
from scipy import interp, arange, exp


 
def Rings(xy_mesh,pa,inc,x0,y0):
	(x,y) = xy_mesh

	X = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))
	Y = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))



	R= np.sqrt(X**2+(Y/np.cos(inc))**2)

	return R



def result(vmode,data,xy_mesh,sigma,Vlist,R_v,pa,inc,x0,y0,Vsys,N_free,phi_b = 0):

	#print(pa, inc, Vsys, x0,y0)
	vrot, vrad, vtan  = Vlist
	pa,inc,phi_b=pa*np.pi/180,inc*np.pi/180,phi_b*np.pi/180


	R_v = np.asarray(R_v) / 1





	# We need to include R = 0, and V = 0

	if vmode == "circular":
		Vrot_list = vrot.copy()
		#Vrot_list.append(0)

	if vmode == "radial":
		Vrot_list = vrot.copy()
		Vrad_list = vrad.copy()
		Vrot_list.append(0)
		Vrad_list.append(0)

	if vmode == "bisymmetric":

		Vtan_list = vtan.copy()
		Vrot_list = vrot.copy()
		Vrad_list = vrad.copy()


		Vrot_list.append(0)
		Vrad_list.append(0)
		Vtan_list.append(0)


	R_list = R_v.tolist()
	R_list.append(0)




	vlos_model  = []
	k = 0

	M = np.zeros((50,50))
	for I,J in xy_mesh:
		K = 0
		(x,y) = I,J
		#print(k, x,y)


		#print(pa, inc, Vsys, x0,y0)
		R  = Rings((x,y),pa,inc,x0,y0)
		#print(R)


		cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
		sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
		theta = np.arctan(sin_tetha/cos_tetha)



		max_R = np.nanmax(R_list)
		if vmode == "circular":
			f0 = interp1d(np.asarray(R_list), np.asarray(Vrot_list), fill_value = "extrapolate")

			Vrot_new = f0(R)
			#Vrot_new = Vrot_list[k]
			k = k+1

			vlos = Vsys+np.sin(inc)*Vrot_new*cos_tetha
			vlos_model.append(vlos)


			for i,j in zip(I,J):
				M[j][i] = vlos[K]
				K = K +1 
			
		if vmode == "radial":

			f0 = interp1d(R_list, Vrot_list, fill_value = "extrapolate")
			f1 = interp1d(R_list, Vrad_list, fill_value = "extrapolate")

			Vrot_new = f0(R)
			Vrad_new = f1(R)
			vlos = Vsys+np.sin(inc)*(Vrot_new*cos_tetha + Vrad_new*sin_tetha)
			vlos_model.append(vlos)

		if vmode == "bisymmetric":
			f0 = interp1d(R_list, Vrot_list, fill_value = "extrapolate")
			f1 = interp1d(R_list, Vrad_list, fill_value = "extrapolate")
			Vrot_new = f0(R)
			Vrad_new = f1(R)
			f2 = interp1d(R_list, Vtan_list, fill_value = "extrapolate")
			Vtan_new = f2(R)


			m = 2
			theta_b = theta - phi_b
			vlos = Vsys+np.sin(inc)*((Vrot_new*cos_tetha) - Vtan_new*np.cos(m*theta_b)*cos_tetha - Vrad_new*np.sin(m*theta_b)*sin_tetha)
			vlos_model.append(vlos)

			
	

	vlos_model = np.asarray(vlos_model)


	sigma = np.asarray(sigma)
	residual = (vlos_model - data )/ sigma
	residual = residual**2
	
	#print(vlos_model[0],"model")
	#print(data[0],"data")
	#print(sigma[0],"sigma")
	#print(residual[0],"res")

	res = np.concatenate([X.ravel() for X in residual])
	#print( np.nansum(res), len(res), np.nanmax(res))

	chi = np.nansum(res)
	chi_red = chi

	#M[M==0] = np.nan
	#plt.imshow(M, origin = "l")
	#plt.show()

	#print(vrot)
	return chi_red


