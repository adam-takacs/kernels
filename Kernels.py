#!/usr/bin/env python3
from numpy import *
from scipy import special
import math
import cmath
import scipy.integrate as integrate

class Kernels:
    '''This class calculates the spectrum dI/dz [1], and the rate dI/dzdL [GeV]
       of a medium-induced emission in various perturbative schemes and
       approximations including running coupling, finite-z terms, all flavors.'''
    def __init__(self, flavs=[21,21,21], cases=2, alphas_run=0):
        '''Flavor dependence: [a,b,c] = a --> b(z) + c(1-z)'''
        self.CA = 3.
        self.CF = 4./3.
        self.nf = 1
        self.flavs = flavs
        # g --> g + g
        if self.flavs[0]==self.flavs[1]==self.flavs[2]==21:
            self.C0=0.5; self.C1=0.5; self.C2=0.5
        # q --> q + g
        elif self.flavs[1]==21 and 0<self.flavs[0]==self.flavs[2]<=self.nf:
            self.C0=0.5; self.C1=self.CF/self.CA-0.5; self.C2=0.5
        # g --> q + qbar
        elif self.flavs[0]==21 and 0<self.flavs[1]==-self.flavs[2]<=self.nf:
            self.C0=self.CF/self.CA-0.5; self.C1=0.5; self.C2=0.5
            if self.cases==0: print("ERROR: no g->q+qbar in the soft limit!")
        else: print("ERROR in Kernels: Wrong flavor assignment!")
        '''Different cases:
            0: soft limit
            1: minimal z-terms in Kernels.qhat()
            2: maximal z-terms in Kernels.qhat()'''
        self.cases = cases
        if self.cases == 0: self.C0=1.; self.C1=0.; self.C2=0.
        '''Medium elastic potential parameters:
           Gyulassy-Wang potential is defined as
           d2sig/d2q = 4pi*qhat0/(q^2+mu^2)^2 [1/GeV]'''
        self.alphas_run = alphas_run # Running coupling on(1), off(0)
        self.qhat0 = 0.3    # Bare quenching parameter [GeV3]
        self.mu    = 0.3    # IR scale[GeV]
        self.mu2s  = self.mu**2/4.*exp(-1.+2.*euler_gamma) # IOE parameter of GW
        self.lam   = self.mu**2/self.qhat0  # Mean free path
        self.wBH   = self.mu**2*self.lam/2. # Bethe-Heitler energy

    '''Altarelli-Parisi Splitting Functions'''
    def P(self, z):
        # Soft limit
        if self.cases==0:
            if   self.flavs[0]==21: return self.CA/(z*(1.-z))
            elif self.flavs[0]!=21: return 2.*self.CF/z
        # g --> g(z) + g(1-z)
        if self.flavs[0]==self.flavs[1]==self.flavs[2]==21:
            return self.CA*(1.-z*(1.-z))**2/(z*(1.-z))
        # q --> g(z) + q(1-z)
        if self.flavs[0]==self.flavs[2] and self.flavs[1]==21:
            return self.CF*(1.+(1.-z)**2)/z
        # g --> q(z) + qbar(1-z)
        if self.flavs[0]==21 and self.flavs[1]==-self.flavs[2]:
            return 0.5*(z**2+(1.-z)**2)

    '''Running coupling'''
    def alphas(self, kt2):
        if   self.alphas_run==0: return 0.28
        elif self.alphas_run==1:
            # 1-loop running coupling
            MZ = 91.192; aMZ = 0.1184
            b0 = (11.*self.CA-2.*self.nf)/(12.*pi)
            alpha = aMZ/(1.+b0*aMZ*log(kt2/MZ**2))
            # Freezing in running coupling
            return alpha if (0<alpha<0.28) else 0.28

    def kt2_med(self, z, p0, t):
        '''Medium kt^2 of the different scattering regions is approximated by
           the minimal possible kt of the process. With this one would overshoot
           the running coupling:
           kt^2 ~ mu^2 (BH), ~ sqrt(2w*qhat) (IOE), ~ 2w/t.'''
        w = z*(1.-z)*p0
        return max(self.mu**2, sqrt(2.*w*self.qhat(z,p0)[0]), 2.*w/t)

    '''Improved Opacity Expansion'''
    def Cba(self, z): return self.C0 + self.C1*z**2 + self.C2*(1.-z)**2

    def qhat(self, z, p0):
        if z<1e-8 or z>1.-1e-8: z=1e-8
        w = z*(1.-z)*p0
        '''The different cases are different ways to separate the potential to
           v_{HO} + \delta v.
           Case 0: pure soft limit:
              v_{HO} = \hat q_0/4 x^2\ln(Q^2/\mu_ast^2),
              \delta v = -\hat q_0/4 x^2\ln(x^2 Q^2).
           Case 1: keeping the minimal part of the z-dependence:
              v_{HO} = \hat q_0/4 x^2(C0 + C1z^2 + C2(1-z)^2)\ln(Q^2/\mu_ast^2),
              \delta v = -\hat q_0/4 x^2(C0\ln(x^2 Q^2)
                                       + C1z^2\ln(z^2 x^2 Q^2)
                                       + C2(1-z)^2\ln((1-z)^2 x^2 Q^2)).
           Case 2: including subleading z-terms:
              v_{HO} = \hat q_0/4 x^2(C0\ln(Q^2/\mu_ast^2)
                                    + C1z^2\ln(Q^2/(z^2\mu_ast^2))
                                    + C2(1-z)^2\ln(Q^2/((1-z)^2\mu_ast^2))),
              \delta v = -\hat q_0/4 x^2(C0 + C1z^2 + C1(1-z)^2)\ln(x^2 Q^2).'''
        if   self.cases==0: c = 0; b = 1.
        elif self.cases==1: c = 0; b = self.Cba(z)
        elif self.cases==2:
            c = - self.C1*z**2*log(z**2) - self.C2*(1.-z)**2*log((1.-z)**2);
            b = self.Cba(z)
        # Analytic solution of Q2 = sqrt(w*qhat0*log(Q2/mu2s))
        if w < 2.*e*self.mu2s**2*exp(2*c/b)/(self.qhat0*b):
            Q2 = self.mu2s*sqrt(exp(1.-2.*c/b)/b)
        else:
            tp = -2.*self.mu2s**2*exp(-2*c/b)/(self.qhat0*w*b)
            Q2 = sqrt(-self.qhat0*w*b/2.*special.lambertw(tp,-1)).real
        qhat = self.qhat0*(log(Q2/self.mu2s)*b+c)
        # Making sure qhat, Q2 is smooth where there is no solution
        Q2 = max(self.mu2s,Q2); qhat = max(self.qhat0,qhat)
        return [qhat, Q2]

    def dIdz_HO(self, z, t, p0=100, limit=False):
        w  = z*(1.-z)*p0
        wc = self.qhat(z,p0)[0]*t**2/2.
        x  = sqrt(2.*wc/w)
        if t==0: return 0
        alp = self.alphas(self.kt2_med(z,p0,t))
        # Cutting off the high frequency part
        if x>100: return alp/pi*self.P(z)*(log(0.25)+x)
        return alp/pi*self.P(z)*log(0.5*(cos(x)+cosh(x)))

    def dIdzdL_HO(self, z, t, p0=100, limit=False):
        w  = z*(1.-z)*p0
        wc = self.qhat(z,p0)[0]*t**2/2.
        x  = sqrt(2.*wc/w)
        if t==0: return 0
        alp = self.alphas(self.kt2_med(z,p0,t))
        #Cutting off the high frequency part
        if x>100: return alp/pi*self.P(z)*x/t
        return alp/pi*self.P(z)*x/t*(sinh(x)-sin(x))/(cos(x)+cosh(x))

    def dIdz_NHO(self, z, t, p0=100, limit=False):
        w  = z*(1.-z)*p0
        qhat, Q2 = self.qhat(z,p0)
        wc = qhat*t**2/2.
        def k2(z, p0, s, t):
            Om = (1.-1j)/2.*sqrt(self.qhat(z,p0)[0]/(z*(1.-z)*p0))
            return 0.5j*(z*(1.-z)*p0)*Om*(1./cmath.tan(Om*s)-cmath.tan(Om*(t-s)))
        if t==0: return 0.
        alp = self.alphas(self.kt2_med(z,p0,t))
        # The different cases are defined in Kernels.qhat().
        if self.cases==0:
            # Soft limit
            if limit==True: return print("Error in Kernels::dIdz_NHO: Unavailable!")
            tp = integrate.quad(lambda s: (-1./k2(z,p0,s,t)                    \
               *(euler_gamma+log(-k2(z,p0,s,t)/Q2))).real, 0, t, epsrel=1e-2)[0]
            return alp/(2.*pi)*self.P(z)*self.qhat0 * tp
        if self.cases==1:
            if limit==True: return print("Error in Kernels::dIdz_NHO: Unavailable!")
            tp = integrate.quad(lambda s: (-1./k2(z,p0,s,t)                    \
               *(self.C0*log(-k2(z,p0,s,t)/(Q2*exp(-euler_gamma)))             \
               +self.C1*z**2*log(-k2(z,p0,s,t)/(z**2*Q2*exp(-euler_gamma)))    \
               +self.C2*(1.-z)**2*log(-k2(z,p0,s,t)                            \
               /((1.-z)**2*Q2*exp(-euler_gamma))))).real, 0, t, epsrel=1e-2)[0]
            return alp/(4.*pi)*self.P(z)*self.qhat0 * tp
        if self.cases==2:
            # Including finite-z terms
            if limit==True:
                # Approximating the integral with its limits
                if w<wc/2.:
                    return alp/pi*self.P(z)*self.Cba(z)*self.qhat0/qhat        \
                    *sqrt(qhat*t**2/(4.*w))*(1.+tanh(sqrt(qhat*t**2/(4.*w)))   \
                    *(euler_gamma-1.+pi/4.+log(sqrt(w*qhat/2.)/Q2))            \
                    + sqrt(w/(qhat*t**2))*(pi**2/12.                           \
                    *tanh(sqrt(qhat*t**2/(4.*w)))-2.*log(2.)))
                else:
                    return alp/2.*self.P(z)*self.Cba(z)*self.qhat0*t**2/(2.*w) \
                    * (1. + 2./(3.*pi)*qhat*t**2/(2.*w)*(2.*euler_gamma-7./12. \
                    +log(w/(2.*t*Q2))))
            int = integrate.quad(lambda s: (-1./k2(z,p0,s,t)*(euler_gamma      \
                  +log(-k2(z,p0,s,t)/Q2))).real, 0, t, epsrel=1e-2)[0]
            return alp/(2.*pi)*self.P(z)*self.Cba(z)*self.qhat0 * int

    def dIdzdL_NHO(self, z, t, p0=100, limit=False):
        qhat, Q2 = self.qhat(z,p0)
        w  = z*(1.-z)*p0
        wc = qhat*t**2/2.
        if t<1e-8: return self.dIdzdL_NHO(z,t-1e-8,p0,limit)
        alp = self.alphas(self.kt2_med(z,p0,t))
        if self.cases==2 and limit==True:
            # Including finite-z terms and pproximating the integral
            if w<wc/2.:
                return alp/pi*self.P(z)*self.Cba(z)*self.qhat0/qhat/t          \
                *sqrt(qhat*t**2/(4.*w))/2./cosh(sqrt(qhat*t**2/(4.*w)))**2     \
                *(pi**2/12.+1.+cosh(sqrt(qhat*t**2/w))+(sqrt(qhat*t**2/w)      \
                +sinh(sqrt(qhat*t**2/w)))*(euler_gamma-1.+pi/4.                \
                +log(sqrt(w*qhat/2.)/Q2)))
            else:
                return alp/2.*self.P(z)*self.Cba(z)*self.qhat0*t/w*(1.         \
                +2./(3.*pi)*qhat*t**2/(2.*w)*(4.*euler_gamma-5./3.             \
                +2.*log(w/(2.*t*Q2))))
        return (self.dIdz_NHO(z,t+1e-8,p0,limit)-self.dIdz_NHO(z,t,p0,limit))/1e-8

    def dIdz_IOE(self, z, t, p0=100, limit=False):
        return self.dIdz_HO(z,t,p0,limit) + self.dIdz_NHO(z,t,p0,limit)

    def dIdzdL_IOE(self, z, t, p0=100, limit=False):
        # The HO+NHO far out of its validity can become negative
        return self.dIdzdL_HO(z,t,p0,limit) + max(0,self.dIdzdL_NHO(z,t,p0,limit))

    '''Opacity Expansion'''
    def dIdz_N1(self, z, t, p0=100, limit=False):
        w = z*(1.-z)*p0
        y = self.mu**2*t/(2.*w)
        if t==0: return 0
        alp = self.alphas(2.*w/t)
        res = self.C0*(euler_gamma-1.+log(y)+(pi*sin(y/2.)**2                  \
                -special.sici(y)[1]*sin(y)+cos(y)*special.sici(y)[0])/y)/y     \
              +self.C1*z**2*(euler_gamma-1.+log(z**2*y)+(pi*sin(z**2*y/2.)**2  \
                -special.sici(z**2*y)[1]*sin(z**2*y)+cos(z**2*y)               \
                *special.sici(z**2*y)[0])/(z**2*y))/(z**2*y)                   \
              +self.C2*(1.-z)**2*(euler_gamma-1.+log((1.-z)**2*y)+(pi          \
                *sin((1.-z)**2*y/2.)**2-special.sici((1.-z)**2*y)[1]           \
                *sin((1.-z)**2*y)+cos((1.-z)**2*y)                             \
                *special.sici((1.-z)**2*y)[0])/((1.-z)**2*y))/((1.-z)**2*y)
        return alp/(pi*w)*self.P(z)*self.qhat0*t**2 * res

    def dIdz_N2(self, z, t, p0=100, limit=False):
        w = z*(1.-z)*p0
        y = self.mu**2*t/(2.*w)
        if t==0: return 0
        alp = self.alphas(2.*w/t)
        if self.cases==0:
            I1 = 0.25/y**3*(4.*euler_gamma-pi*y+y**2+4*log(y)-cos(y)*(4.       \
                 *special.sici(y)[1]+y*(pi-2.*special.sici(y)[0]))+2.*sin(y)   \
                 *(pi-y*special.sici(y)[1]-2.*special.sici(y)[0]))
            def temp(p0,p):
                r1  = (1.-cos(p0))/p0**2 if p0<7.5*pi else 1./p0**2
                r2  = (1.-cos(p))/p**2   if p<7.5*pi  else 1./p**2
                res = r1-r2
                return p0/(p+y)/((p+p0+y)**2-4.*p*p0)**1.5/(p-p0)*res
            I2 = integrate.dblquad(temp,0,inf,lambda p: p, inf)[0]             \
               + integrate.dblquad(temp,0,inf,0,lambda p: p, epsrel=1e-10)[0]
            return -4*alp/pi*self.P(z)*(t/self.lam)**2*y*(I1-y*I2)
        else: print("Only available in the soft limit!")

    def dIdzdL_N1(self, z, t, p0=100, limit=False):
        w = z*(1.-z)*p0
        y = self.mu**2*t/(2.*w)
        if t==0: return 0
        alp = self.alphas(2.*w/t)
        res = self.C0*(euler_gamma-cos(y)*special.sici(y)[1]-sin(y)            \
                *special.sici(y)[0]+log(y)+pi/2.*sin(y))/y                     \
              +self.C1*z**2*(euler_gamma-cos(z**2*y)*special.sici(z**2*y)[1]   \
                -sin(z**2*y)*special.sici(z**2*y)[0]+log(z**2*y)+pi/2.         \
                *sin(z**2*y))/(z**2*y)                                         \
              +self.C2*(1.-z)**2*(euler_gamma-cos((1.-z)**2*y)                 \
                *special.sici((1.-z)**2*y)[1]-sin((1.-z)**2*y)                 \
                *special.sici((1.-z)**2*y)[0]+log((1.-z)**2*y)+pi/2.           \
                *sin((1.-z)**2*y))/((1.-z)**2*y)
        return alp/(pi*w) * self.P(z) * self.qhat0 * t * res

    def dIdzdL_N2(self, z, t, p0=100, limit=False):
        if t<1e-8:   return self.dIdzdL_N2(z,t-1e-8,p0,limit)
        return (self.dIdz_N2(z,t+1e-8,p0,limit)-self.dIdz_N2(z,t,p0,limit))/1e-8

    '''Resummed Opacity Expansion'''
    def dIdz_NR1(self, z, t, p0=100, limit=False):
        w = z*(1.-z)*p0
        y = self.mu**2*t/(2.*w)
        S = self.qhat0/self.mu**2*self.Cba(z)*t
        if t==0: return 0
        alp = self.alphas(self.mu**2)
        def intres(y,s):
            return 1./(2.*s*(s**2+y**2)**2) * (2.*y*(s**2+y**2)+exp(s)         \
              *(pi*s**3*(s-1.)+pi*y**2*s*(1.+s)-2.*y*(s**2+y**2))+pi*s         \
              *(s**2-y**2)*cos(y)+2.*s*(exp(s)*y*(s*(s-2.)+y**2)               \
              *(special.expi(-s)+log(y/s))+special.sici(y)[1]*(2.*s*y*cos(y)   \
              +(s**2-y**2)*sin(y))-s*y*sin(y)*(pi-2.*special.sici(y)[0])       \
              -(s**2-y**2)*cos(y)*special.sici(y)[0]))
        res = self.C0*intres(y,S) + self.C1*z**2*intres(z**2*y,S)              \
            + self.C2*(1.-z)**2*intres((1.-z)**2*y,S)
        return alp/(pi*w)*self.P(z)*self.qhat0*t**2*exp(-S) * res

    def dIdz_NR2(self, z, t, p0=100, limit=False):
        w   = z*(1.-z)*p0
        wOE = self.mu**2*t/2.
        y   = wOE/w
        S   = self.qhat0/self.mu**2*self.Cba(z)*t
        if t==0: return 0
        alp = self.alphas(self.mu**2)
        if self.cases==0:
            if limit==True:
                if S<1:
                    if w<wOE: return alp*self.P(z)*S**2/y
                    else:     return alp/6.*self.P(z)*S**2*y
                else:
                    if w<self.wBH: return alp*self.P(z)*S*w/self.wBH
                    else:          return alp*self.P(z)*S*self.wBH/w
            def temp(p0,p):
                r1  = 1./(p0**2+S**2)**2*(S*(p0**2+S**2)+p0**2-S**2-exp(-S)    \
                    * ((p0**2-S**2)*cos(p0)+2*S*p0*sin(p0)))
                r2  = 1./(p**2+S**2)**2*(S*(p**2+S**2)+p**2-S**2-exp(-S)       \
                    * ((p**2-S**2)*cos(p)+2*S*p*sin(p)))
                res = r1-r2
                return p0/(p+y)/((p+p0+y)**2-4.*p*p0)**1.5/(p-p0)*res
            intres = integrate.dblquad(temp,0,inf,lambda p: p, inf)[0]         \
                   + integrate.dblquad(temp,0,inf,0,lambda p: p)[0]
            return 4 * alp/pi * self.P(z) * S**2 * y**2 * intres
        else: print("Only available in the soft limit!")

    def dIdzdL_NR1(self, z, t, p0=100, limit=False):
        w = z*(1.-z)*p0
        y = self.mu**2*t/(2.*w)
        S = self.qhat0/self.mu**2*self.Cba(z)*t
        if t==0: return 0
        alp = self.alphas(self.mu**2)
        def intres(y,s):
            return 1./(2.*(s**2+y**2)) * (exp(S)*(pi*S+2*y*(special.expi(-S)   \
              -log(S/y)))-cos(y)*(2.*y*special.sici(y)[1]+S*(pi-2.             \
              *special.sici(y)[0]))+sin(y)*(-2.*s*special.sici(y)[1]+y*(pi-2.  \
              *special.sici(y)[0])))
        res = self.C0*intres(y,S) + self.C1*z**2*intres(z**2*y,S)              \
            + self.C2*(1.-z)**2*intres((1.-z)**2*y,S)
        return alp/(pi*w) * self.P(z) * self.qhat0 * t * exp(-S) * res

    def dIdzdL_NR2(self, z, t, p0=100, limit=False):
        if t<1e-8:   return self.dIdzdL_NR2(z,t-1e-8,p0,limit)
        return (self.dIdz_NR2(z,t+1e-8,p0,limit)-self.dIdz_NR2(z,t,p0,limit))/1e-8

    '''Full Interpolation'''
    def dIdz_Full(self, z, t, p0=100, limit=False, smooth=False):
        w = z*(1.-z)*p0
        if t==0: return 0.
        wcrit = min(self.wBH,self.wBH*t/self.lam)
        if smooth==True:
            # Smoothing the transition between ROE and IOE
            wmin = 0.5*wcrit; wmax=2*wcrit
            if w<wmin: return self.dIdz_NR1(z,t,p0,limit)
            elif wmin<w<wmax:
                x = (w - wmin) / (wmax - wmin)
                S = cos(pi/2.*(1.-x)) #Arbitrary function that goes from 0 to 1.
                return (1.-S)*self.dIdz_NR1(z,t,p0,limit)                      \
                       + S*self.dIdz_IOE(z,t,p0,limit)
            else:   return self.dIdz_IOE(z,t,p0,limit)
        if w<wcrit: return self.dIdz_NR1(z,t,p0,limit)
        else:       return self.dIdz_IOE(z,t,p0,limit)

    def dIdzdL_Full(self, z, t, p0=100, limit=False, smooth=False):
        w = z*(1.-z)*p0
        if t==0: return 0.
        wcrit = min(self.wBH,self.wBH*t/self.lam)
        if smooth==True:
            # Smoothing the transition between ROE and IOE
            wmin = 0.5*wcrit; wmax=2*wcrit
            if w<wmin: return self.dIdzdL_NR1(z,t,p0,limit)
            elif wmin<w<wmax:
                x = (w - wmin) / (wmax - wmin)
                S = cos(pi/2.*(1.-x)) #Arbitrary function that goes from 0 to 1.
                return (1.-S)*self.dIdzdL_NR1(z,t,p0,limit)                    \
                        + S*self.dIdzdL_IOE(z,t,p0,limit)
            else:   return self.dIdzdL_IOE(z,t,p0,limit)
        if w<wcrit: return self.dIdzdL_NR1(z,t,p0,limit)
        else:       return self.dIdzdL_IOE(z,t,p0,limit)


z0=1e-6; t0=2.; p0=1e4
kern=Kernels([21,21,21],0)
kern2=Kernels([21,21,21],2)
print("g-->gg: z = ",z0,", L = ", t0," [1/GeV], E = ", p0, " [GeV]")
print("Soft limit: dI/dz = ",kern.dIdz_Full(z0,t0,p0,0)," [1]")
print("Soft limit: dI/dzdL = ",kern.dIdzdL_Full(z0,t0,p0,0)," [GeV]")
print("Finite-z: dI/dz = ",kern2.dIdz_Full(z0,t0,p0,0)," [1]")
print("Finite-z: dI/dzdL = ",kern2.dIdzdL_Full(z0,t0,p0,0)," [GeV]")
print("")
kern=Kernels([1,21,1],0)
kern2=Kernels([1,21,1],2)
print("q-->qg: z = ",z0,", L = ", t0," [1/GeV], E = ",p0, " [GeV]")
print("Soft limit: dI/dz = ",kern.dIdz_Full(z0,t0,p0,0)," [1]")
print("Soft limit: dI/dzdL = ",kern.dIdzdL_Full(z0,t0,p0,0)," [GeV]")
print("Finite-z: dI/dz = ",kern2.dIdz_Full(z0,t0,p0,0)," [1]")
print("Finite-z: dI/dzdL = ",kern2.dIdzdL_Full(z0,t0,p0,0)," [GeV]")
