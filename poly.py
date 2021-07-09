import numpy as np


def legendre(x,y,n=6):
	x,y = np.insert(x, 0, 0),np.insert(y, 0, 0)
	legfit = np.polynomial.legendre.legfit(x,y, 5)
	legval = np.polynomial.legendre.legval(x,legfit)
	return legval[1:]
