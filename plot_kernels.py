#!/usr/bin/env python3
from numpy import *
from scipy import special
import math
import cmath
import scipy.integrate as integrate
from matplotlib import pyplot as plt

from Kernels import Kernels

# Gluon splitting with finite-z
kern=Kernels([21,21,21],2)

# List of [z, dI(t,E)/dz] for different propagation length
En = 1e4 # [GeV]
t1 = 0.2 # [GeV^-1]
rate_Full1 = array([[z,kern.dIdzdL_Full(z,t1,En)] for z in logspace(-7,-1e-7,100)])
t2 = 2.0 # [GeV^-1]
rate_Full2 = array([[z,kern.dIdzdL_Full(z,t2,En)] for z in logspace(-7,-1e-7,100)])
t3 = 20. # [GeV^-1]
rate_Full3 = array([[z,kern.dIdzdL_Full(z,t3,En)] for z in logspace(-7,-1e-7,100)])


# Plot begins
print('Quick test that plots the rate.')
fig = plt.figure(figsize=(7,6))

plt.loglog(rate_Full1.T[0],rate_Full1.T[0]**1.5*rate_Full1.T[1],'-',label=r'$t=0.2$ [1/GeV]')
plt.loglog(rate_Full2.T[0],rate_Full2.T[0]**1.5*rate_Full2.T[1],'-',label=r'$t=2$ [1/GeV]')
plt.loglog(rate_Full3.T[0],rate_Full3.T[0]**1.5*rate_Full3.T[1],'-',label=r'$t=20$ [1/GeV]')

plt.ylabel(r'$z^{3/2}\frac{{\rm d}I_g}{{\rm d}z{\rm d}t}$ [GeV]', fontsize=20)
plt.xlabel(r'$z$',fontsize=20)
plt.tick_params(axis='both',which='both', right=True, top=True, bottom=True, direction='in', labelsize=20)
plt.legend(fontsize=18, ncol=1,loc='lower left');
plt.ylim(1e-6,1e-1)
plt.xlim(1e-7,1)
plt.tight_layout()
plt.subplots_adjust(hspace=.0,wspace=.0)
plt.savefig("plot_kernels.pdf", bbox_inches="tight")
plt.show()
