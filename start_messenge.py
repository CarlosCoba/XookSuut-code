from read_config import config_file

def start(galaxy,guess,vmode,config):

	PA,INC,X0,Y0,VSYS = 	guess[2],guess[3],guess[4],guess[5], guess[6] 
	if config == "":

		print("############################")
		print("#### Running Xook-Suut #####")
		print("Guess values for %s"%galaxy)
		print("pa:\t %s"%PA)
		print("inc:\t %s"%INC)
		print("x0,y0:\t %s,%s"%(X0,Y0))
		print("vsys:\t %s"%VSYS)
		print("MODEL:\t %s"%vmode)
		print("############################")

	else:

		print("############################")
		print("#### Running Xook-Suut #####")
		print("Guess values for %s"%galaxy)
		print("\t","fit","\t","value","\t","min","\t","max")
		for res in config_file(config):
				param, fit, val, vmin, vmax = str(res["param"]), bool(float(res["fit"])), round(eval(res["val"]),1), round(eval(res["min"]),1), round(eval(res["max"]),1) 
				print(param,"\t",fit,"\t",val,"\t",vmin,"\t",vmax)

		print("MODEL:\t %s"%vmode)
		print("############################")


