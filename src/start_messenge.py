from src.read_config import config_file

def start(galaxy,guess,vmode,config):

		PA,INC,X0,Y0,VSYS,PHI_B = 	guess[0],guess[1],guess[2],guess[3], guess[4], guess[5] 

		print("############################")
		print("#### Running Xook-Suut #####")
		print("Guess values for %s"%galaxy)
		print("pa:\t\t %s"%PA)
		print("inc:\t\t %s"%INC)
		print("x0,y0:\t\t %s,%s"%(round(X0,2),round(Y0,2)))
		print("vsys:\t\t %s"%round(VSYS,2))
		if vmode == "bisymmetric" :
			print("phi_bar:\t %s"%PHI_B)			
		print("model:\t\t %s"%vmode)
		print("############################")


