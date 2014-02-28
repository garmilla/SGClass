# -*- python -*-

class type:
    label = None
    path = None
    z_min = None
    z_max = None 
    Nz = None
    z_ref = None
    z_median = None
    z_pow = None

star = type()
star.label = 'star'
star.path = 'sed/star'
star.z_min = 0.0
star.z_max = 0.0
star.Nz = 1

galaxy = type()
galaxy.label = 'galaxy'
galaxy.path = 'sed/galaxy'
galaxy.z_min = 1.0e-4
galaxy.z_max = 2.0
galaxy.Nz = 30
galaxy.z_ref = galaxy.z_min
galaxy.z_median = 0.3
galaxy.z_pow = 2.5

qso = type()
qso.label = 'qso'
qso.path = 'sed/qso'
qso.z_min = 1.0e-4
qso.z_max = 5.0
qso.Nz = 30
qso.z_ref = 2.0
qso.z_median = 1.0
qso.z_pow = 2.5

self.classes = [star, galaxy, qso]

self.filters = {'u':'filt/up.pb',\
                'g':'filt/gp.pb',\
		'r':'filt/rp.pb',\
                'i':'filt/ip.pb',\
		'z':'filt/zp.pb'}

self.maglims = {'r':np.array([20.0, 21.0])}

self.num = 1001

self.fuzz = 0.5

self.fuzz_fac = 0.9

self.data = 'result_patch.txt'
self.ra = 349.0
self.dec = 1.0

self.method = 2

self.fuzz = 0.5

self.fuzz_fac = 0.9
