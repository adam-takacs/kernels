#!/usr/bin/env python3
from numpy import *
from scipy import special
import math
import cmath
import scipy.integrate as integrate

from Kernels import Kernels

class Evolution:
    '''A class that solves the evolution equation
    d/dt D(x,t) = int_x^1 dxi P(x,xi,t)*D(xi,t) - D(x,t)*int_0^1 dxi P(xi,x,t)
    where D(x,t) = w dN(E)/dw [1], x=w/E, and the splitting kernel is
    P(x,xi,t) = x/xi^2 * dI(E)/(dzdt)(z=x/xi,E=xi*p0,t) [GeV].'''
    def __init__(self, fl=[21,21,21], p0=100.):
        cases          = 2 # Soft limit (0), finite-z (2)
        running_alphas = 0 # Frozen coupling (0), running (1)
        self.kern   = Kernels(fl,cases,running_alphas)
        self.p0     = p0       # [GeV]
        self.Lmax   = 5./0.197 # [GeV-1]
        '''Initializing the grid:
           The grid in x is logarithmic and symmetric in x,1-x.
           The grid in t is linear and it is in GeV.
           (The algorithm is 1% accurate with Nx=1000 and Nt=5000 for ~5fm
           medium. By decreasing the grid to Nx=100, the precision changes
           drastically.)'''
        self.Nx     = 1000
        self.xlow   = 1e-6
        self.xval   = zeros(self.Nx+1)
        for i in range(1,self.Nx+2):
            m = (2.*(i-1.)-self.Nx)/self.Nx*log((1.-self.xlow)/self.xlow)
            self.xval[i-1] = exp(m)/(1.+exp(m))
        self.deltax = append(self.xval[1:]-self.xval[:-1], 0)
        self.Nt     = 5000
        self.qval   = linspace(1e-6,self.Lmax,self.Nt+1) #[GeV-1]
        self.dq     = self.qval[1]-self.qval[0]
        # Results
        self.f_grid = zeros([self.Nx+1,self.Nx+1])
        self.Dres   = zeros([self.Nx+1,self.Nt+1])
        self.floss  = zeros(self.Nx+1)

    def grid_int(self, f_arr):
        # Calculate int_0^1 dx f(x) of an arbitrary array using trapezoids.
        return sum((f_arr[1:]+f_arr[:-1])/2.*self.deltax[:-1])

    def init(self):
        # Initialize D(x,t) with delta(1-x), ensuring the integral = 1
        self.Dres[self.Nx,0] = 2./self.deltax[self.Nx-1]

    def f(self, x, xi, t):
        '''Definition of the emission kernel'''
        #return x/xi**2*self.kern.dIdzdL_HO(x/xi,t,xi*self.p0,True)   if x<xi else 0
        #return x/xi**2*self.kern.dIdzdL_Full(x/xi,t,xi*self.p0,True) if x<xi else 0
        return self.f0(x,xi,t)

    def f0(self, x, xi, t):
        # Soft BDMPSZ, simplified kernel, fixed qhat.
        temp = self.kern.alphas(0)*self.kern.CA/pi*sqrt(self.kern.qhat0/self.p0)
        return (temp * x*xi**(-2.5)/(x/xi*(1.-x/xi))**1.5) if x<xi else 0

    def prepare_split(self, k):
        self.f_grid = array([[self.f(xi,xj,self.qval[k]) for xj in self.xval] for xi in self.xval])
        temp        = integrate.quad(lambda xi: self.f(xi,self.xval[0],self.qval[k-1]), 0, 2.*self.xval[0]-self.xval[1], epsabs=1e-2)[0]
        self.floss  = append(temp, [integrate.quad(lambda xi: self.f(xi,x,self.qval[k-1]), 0, self.xval[0], epsabs=1e-2)[0] for x in self.xval[1:]])

    def deriv(self, y):
        dydx = []
        for i in range(1,self.Nx+2):
            gn = sum(0.5*(self.f_grid[i-1,:-1]*self.deltax[:-1]*y[:-1] + self.f_grid[i-1,1:]*self.deltax[:-1]*y[1:]))
            ls = sum(0.5*(self.f_grid[:-1,i-1]*self.deltax[:-1]        + self.f_grid[1:,i-1]*self.deltax[:-1]))
            dydx.append(gn - (ls + self.floss[i-1]) * y[i-1])
        return array(dydx)

    def next_step(self, k, integrator='RK4'): #TODO RK4, but the time interval is simplified.
        y     = self.Dres[:,k-1]
        dydx  = self.deriv(y)
        if   integrator=='Euler': dd = y + self.dq*dydx
        elif integrator=='RK4':
            y1    = y + self.dq/2. * dydx
            dydx1 = self.deriv(y1)
            y1    = y + self.dq/2. * dydx1
            dydx2 = self.deriv(y1)
            y1    = y + self.dq * dydx2
            dydx3 = dydx2 + dydx1
            dydx1 = self.deriv(y1)
            dd    = y + self.dq/6. * (dydx + dydx1 + 2.*dydx3)
        self.Dres[:,k] = array([max(i) for i in zip(zeros(self.Nx+1),dd)])

    def run(self, tmax):
        print("Evolution::run() is initializing the the evolution.")
        self.init()
        print("Evolution::run() is is running.")
        for k in range(1,self.Nt+1):
            print("Timestep: ",self.Nt," / ",k, end="\r")
            self.prepare_split(k)
            self.next_step(k)
            if self.qval[k]<tmax:
                with open('result.out', 'a') as file:
                    for item in self.Dres[:,k]: file.write(str(item)+' ')
                    file.write('\n')
            else: break
        print("The fragmentation function hss been saved in result.out.\n      \
               the read of the file is: D(x,t), where x are the rows,\n        \
               and t are the columns as defined in Evolution::xval,\n          \
               and Evolution::qval.")

#
print('Quick test that solves the evolution equation in the soft limit.')
evol = Evolution()
evol.run(0.5)
