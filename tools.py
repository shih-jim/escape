"""
Tools for ESCAPE project
"""

import astropy
import numpy as np

#### define constants ####
# All in SI units

# c = 2.99792458e8  # m/s
c = astropy.constants.c.to("m/s").value
# h = 6.62607015e-34  # J s
h = astropy.constants.h.to("J*s").value
# u = 1.6605390666e-27  # kg  -atomic mass unit intended to be in units of an atomic hydrogen mass
u = astropy.constants.u.to("kg").value
# kB = 1.380649e-23  # J/K
kB = astropy.constants.k_B.to("J/K").value
# AU = 1.49597871e11  # m
AU = astropy.constants.au.to("m").value
# pc = 3.085677581491367e+16  # m
pc = astropy.constants.pc.to("m").value
# RJ = 7.1492e7  # m
RJ = astropy.constants.R_jup.to("m").value
# RE = 6.371e6  # m
RE = astropy.constants.R_earth.to("m").value
# Rsun = 6.957e8  # m
Rsun = astropy.constants.R_sun.to("m").value
# Msun = 1.988409870698051e+30  # kg
Msun = astropy.constants.M_sun.to("kg").value
# MJ = 1.8981245973360505e+27  # kg
MJ = astropy.constants.M_jup.to("kg").value
# ME = 5.972167867791379e+24  # kg
ME = astropy.constants.M_earth.to("kg").value
# G = 6.6743e-11  # m3/kg/s2
G = astropy.constants.G.to("m**3 * kg**-1 * s**-2").value
# eV = 1.602e-19  # J
eV = astropy.constants.e.to("C").value*1.0  # energy = charge x voltage
# sigma_sb = 5.670374419e-8  # W/m2/K4
sigma_sb = astropy.constants.sigma_sb.to("W/m**2/K**4").value
