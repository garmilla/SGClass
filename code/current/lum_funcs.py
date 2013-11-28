import numpy as np

def Phi_gal(M, band='r'):
    h = 0.6932
    if band == 'u': 
        Mstar = -17.47 + 5.0*np.log10(h)
	ph1 = 1.97*1.0e-2*h**3
	a1 = 0.45
	ph2 = 2.10*1.0e-2*h**3
	a2 = -1.45
    elif band == 'g':
        Mstar = -19.22 + 5.0*np.log10(h)
	ph1 = 1.29*1.0e-2*h**3
	a1 = 0.14
	ph2 = 1.18*1.0e-2*h**3
	a2 = -1.40
    elif band == 'r':
        Mstar = -20.04 + 5.0*np.log10(h)
	ph1 = 1.56*1.0e-2*h**3
	a1 = -0.17
	ph2 = 0.62*1.0e-2*h**3
	a2 = -1.52
    elif band == 'i':
        Mstar = -20.47 + 5.0*np.log10(h)
	ph1 = 1.35*1.0e-2*h**3
	a1 = -0.18 
	ph2 = 0.59*1.0e-2*h**3
	a2 = -1.49
    elif band == 'z':
        Mstar = -20.68 + 5.0*np.log10(h)
	ph1 = 1.50*1.0e-2*h**3
	a1 = -0.34
	ph2 = 0.38*1.0e-2*h**3
	a2 = -1.60
    Phi = 0.4*np.log(10.0)*np.exp(-np.power(10.0,-0.4*(Mr-Mstar)))*\
          (ph1*np.power(10.0,-0.4*(Mr-Mstar)*(a1+1))+\
           ph2*np.power(10.0,-0.4*(Mr-Mstar)*(a2+1)))
    return Phi

def Phi_qso(M, z, band='i'):
    zref = 2.45
    Mstar = -26.0
    if z <= 2.4:
        A1 = 0.84
	A2 = 0.0
	B1 = 1.43
	B2 = 36.63
	B3 = 34.39
	Phistar = np.power(10.0, -5.7)
    else:
        A1 = 0.83
	A2 = -0.11
	B1 = 1.43
	B2 = 36.63
	B3 = 34.39
	Phistar = np.power(10.0, -5.7)
    xi = np.log10((1+z)/(1+zref))
    mu = M - (Mstar + B1*xi + B2*xi**2 + B3*xi**3)
    Phi = Phistar*np.power(10.0, mu*(A1+A2*(z-2.45)))
    return Phi
