import re
import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import griddata

Lsun = 3.839e33
Lvega = 40.12*Lsun
pc = 3.08567758e18 # Parsec in cm

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
    Phi = 0.4*np.log(10.0)*np.exp(-np.power(10.0,-0.4*(M-Mstar)))*\
          (ph1*np.power(10.0,-0.4*(M-Mstar)*(a1+1))+\
           ph2*np.power(10.0,-0.4*(M-Mstar)*(a2+1)))
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

def juric_lum_func(sed_file, mags, filt_names):
    ridx = filt_names.index('r')
    iidx = filt_names.index('i')
    ri = mags[ridx] - mags[iidx]
    Mr = 4.0 + 11.86*ri - 10.74*ri**2 + 5.99*ri**3 - 1.20*ri**4
    C =  np.power(10.0, -0.4*(Mr - mags[ridx]))
    sed = np.loadtxt(sed_file)
    L = 4*np.pi*(10*pc)**2*C*cumtrapz(sed[:,1], sed[:,0])[-1]
    return L

def load_pickles_dic():
    table2 = np.loadtxt('pickles_lum.txt', dtype='string')
    pickles_dic = {}
    for row in table2:
        sptype = row[4][1:-1].upper()
	Mbol = float(row[14])
	L = np.power(10.0, -2.0/5*Mbol)*Lvega
        pickles_dic[sptype] = L
    return pickles_dic

def pickles_lum(sed_file, pickles_dic=None):
    if pickles_dic == None:
        pickles_dic = load_pickles_dic()
    p = re.compile(r'.*PICKLES/')
    m = p.match(sed_file)
    sptype = sed_file[m.end():-4].upper()
    L = pickles_dic[sptype]
    return L

def wd_lum(sed_file):
    if 'g191b2b' in sed_file:
        Mb = 2.932
    elif 'gd153' in sed_file:
        Mb = 4.830
    elif 'gd71' in sed_file:
        Mb = 6.169
    elif 'hz43' in sed_file:
        Mb = 4.676
    L = np.power(10.0, -2.5*Mb)*Lvega
    return L

def load_allard_isochrones(isochrones_file):
    isochrones = np.loadtxt(isochrones_file)
    Teff = isochrones[:,1]
    logL = isochrones[:,2]
    logg = isochrones[:,3]
    grid = np.empty((len(Teff),2))
    grid[:,0] = Teff
    grid[:,1] = logg
    return grid, logL

def interpolate_allard_isochrones(isochrones_file, Teff_i, logg_i):
    xi = np.empty((len(Teff_i),2))
    xi[:,0] = Teff_i
    xi[:,1] = logg_i
    grid, logL = load_allard_isochrones(isochrones_file)
    L_i = np.power(10.0, griddata(grid, logL, xi, method="cubic"))*Lsun
    indexes = np.where(np.isnan(L_i))
    for i in indexes:
        L_i[i] = np.power(10.0, griddata(grid, logL, xi[i], method="nearest"))*Lsun
    return L_i

def load_bd_luminosities(sed_files):
    Teff_dusty = [] ; logg_dusty = [] ; index_dusty = []
    Teff_cond = [] ; logg_cond = [] ; index_cond = []
    Teff_NextGen = [] ; logg_NextGen = [] ; index_NextGen = []
    for (i, sed_file) in enumerate(sed_files):
        if 'AMES-dusty' in sed_file:
            Teff_dusty.append(float(sed_file[-27:-25])*1.0e2)
            logg_dusty.append(float(sed_file[-24:-21]))
	    index_dusty.append(i)
        elif 'AMES-cond' in sed_file:
            Teff_cond.append(float(sed_file[-26:-24])*1.0e2)
            logg_cond.append(float(sed_file[-23:-20]))
	    index_cond.append(i)
        elif 'NextGen' in sed_file:
            Teff_NextGen.append(float(sed_file[-24:-22])*1.0e2)
            logg_NextGen.append(float(sed_file[-21:-18]))
	    index_NextGen.append(i)
    L_dusty = interpolate_allard_isochrones('model.AMES-dusty.SDSS', Teff_dusty, logg_dusty)
    L_cond = interpolate_allard_isochrones('model.AMES-Cond-2000.SDSS', Teff_cond, logg_cond)
    L_NextGen = interpolate_allard_isochrones('model.NextGen.M-0.0.SDSS', Teff_NextGen, logg_NextGen)
    L = np.concatenate((L_dusty, L_cond, L_NextGen))
    index = np.concatenate((index_dusty, index_cond, index_NextGen))
    return (index, L)
