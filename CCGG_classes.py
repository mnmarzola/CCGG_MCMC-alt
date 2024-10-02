#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
import os
import emcee
import corner
from getdist import plots, gaussian_mixtures, MCSamples
import time


# # Data

# In[16]:


class Data:
    
    def __init__(self):
        #binned Pantheon SNe Ia data
        self.SNe_full_data = pd.read_csv('Pantheon_SNeIa_dataset/lcparam_DS17f.txt',sep=' ')
        self.SNe_redshift = self.SNe_full_data['zcmb']
        self.SNe_appmag = self.SNe_full_data['mb']
        self.SNe_appmag_unc = self.SNe_full_data['dmb']
        self.distmod = self.SNe_appmag - (-19.25)  #-19.25 = absolute magnitude for Type 1a SNe
        self.distmod_unc = self.SNe_appmag_unc
        
        #CC data
        self.CC_full_data = pd.read_excel('CC_dataset/CC_data.xlsx')
        self.CC_redshift = self.CC_full_data['redshift']
        self.CC_Hub = self.CC_full_data['H(z)']
        self.CC_Hub_unc = self.CC_full_data['sigma']
        
        #BAO data
        self.BAO_full_data = pd.read_excel('BAO_dataset/BAO_data.xlsx')
        self.BAO_redshift = self.BAO_full_data['redshift']
        self.BAO_Thetdeg = self.BAO_full_data['Theta [deg]']
        self.BAO_Thetdeg_unc = self.BAO_full_data['sigma [deg]']
        
        #CMB data
        self.CMB_redshift = 1089
        self.CMB_R = 1.7661
        self.CMB_la = 301.7293
        self.CMB_v_obs = np.array([self.CMB_R,self.CMB_la])
        self.CMB_C = 10**(-8)*np.array([[33483.54, -44417.15],[-44417.15, 4245661.67]]) #covariance matrix
        self.CMB_C_inv = np.linalg.inv(self.CMB_C)
        
        #BBN data (Our "data" is the hubble parameter for LambdaCDM at z=10^9)
        self.BBN_redshift = 10**9
        self.BBN_HLCDM = 67.4*np.sqrt(5*10**(-5)*(1+self.BBN_redshift)**4) #(67.4 is the average Hubble param value today)
    
    
    def plot_SNe_Data(self):
        #plot
        plt.figure()
        plt.errorbar(self.SNe_redshift, self.distmod, yerr=self.distmod_unc, capsize=3, fmt='r.', label='Pantheon data')
        plt.xlabel('$z$')
        plt.ylabel(r'$\mu$')
        plt.legend()
        plt.show()
        return
    
    
    def plot_CC_Data(self):
        #plot
        plt.figure()
        plt.errorbar(self.CC_redshift, self.CC_Hub, yerr=self.CC_Hub_unc, capsize=3, fmt='r.', label='CC data')
        plt.xlabel('$z$')
        plt.ylabel(r'$H\,\,\,[km/s/Mpc]$')
        plt.legend()
        plt.show()
        return
    
    
    def plot_BAO_Data(self):
        #change to radians
        Thet = self.BAO_Thetdeg*np.pi/180 
        Thet_unc = self.BAO_Thetdeg_unc*np.pi/180
        
        #True observable with uncertainty
        self.BAO_DA_rdrag = 1/((1+self.BAO_redshift)*Thet)
        self.BAO_DA_rdrag_unc = Thet_unc/((1+self.BAO_redshift)*Thet**2)
        
        #plot
        plt.figure()
        plt.errorbar(self.BAO_redshift, self.BAO_DA_rdrag, yerr=self.BAO_DA_rdrag_unc, capsize=3, fmt='r.', label='BAO data')
        plt.xlabel('$z$')
        plt.ylabel(r'$D_A/r_{drag}$')
        plt.legend()
        plt.show()
        return
    
    
    def print_CMB_Data(self):
        print('CMB redshift =',self.CMB_redshift)
        print('R =',self.CMB_R)
        print('la =',self.CMB_la)
        print('Covariance matrix =',self.CMB_C)
        return
    
    
    def print_BBN_Data(self):
        print('BBN redshift =',self.BBN_redshift)
        print('Hubble parameter at BBN =',self.BBN_HLCDM)
        return


# # Numerical Solution

# In[17]:


class AXIS_interval:
    
    def __init__(self, x_iv):
        self.m    = x_iv[2]+1
        self.x    = np.linspace(x_iv[0],x_iv[1],self.m)
        self.xrev = np.linspace(x_iv[1],x_iv[0],self.m)
        self.y    = np.zeros((3,self.m))
        self.ms   = 0
        self.me   = 0


# In[18]:


class AXIS:
    
    def __init__(self, x_iv):
        self.AXI = []
        #Input of xs,xe,n_sample per interval----------
        n = 0
        for i in range(len(x_iv)):
            self.AXI.append(AXIS_interval(x_iv[i]))
            self.AXI[i].ms = n
            n += self.AXI[i].m-1
            self.AXI[i].me = n
        self.niv = len(self.AXI)    #no. of intervals
        self.m = 0
        for i in range(self.niv):
            self.m += self.AXI[i].m
        self.m -= self.niv-1
        
        self.x = np.zeros(self.m)
        k = 0
        for i in range(self.niv):
            for j in range(self.AXI[i].m):
                self.x[k+j] = self.AXI[i].x[j]
            k += self.AXI[i].m-1

        
    def get_index(self, xi):
        x = self.x
        m = self.m
        mx = 0
        if   xi <= x[0]:   mx = 0
        elif xi >= x[m-1]: mx = m-1
        else:
            for i in range(self.m):
                if x[i] > xi:
                    break
                mx = i
        return mx
    
    
    def get_interval(self, index):
        for i in range(self.niv):
            ms  = self.AXI[i].ms
            me1 = self.AXI[i].me+1
            for j in range(ms,me1):
                if index == j: return i
        return -1


# In[19]:


class FLEQ:

    def __init__(self, GR, AX):
        self.GR     = GR
        self.AX     = AX
        self.eps    = 1
        self.neq    = 4 
        self.y      = np.zeros((self.neq, AX.m))
        self.yp     = np.zeros(self.neq)
        self.imin   = 0
        self.imax   = AX.m
        self.mv     = 1.e20
        self.prc    = 1.e-12
        self.Om     = 0
        self.Orad   = 0
        self.Ol     = 0
        self.Ok     = 0
        self.h      = 0
        self.Os     = 0
        self.Og     = 0
        self.s1     = 0
        self.Sg1    = 0
        self.H0     = 0
        self.switch = 0

    
    def set_prm(self, *args):
        self.Om   = Om   = args[0]
        self.Orad = Orad = args[1]
        self.Ol   = Ol   = args[2]
        self.Ok   = Ok   = args[3]
        self.h    = h    = args[4]
        self.Os   = Os   = args[5]
        self.Og   = Og   = args[6]
        self.s1   = s1   = args[7]
        self.H0   = h*2.13312e-42

        # Sigma_1
        M0        = Om/4+Ol
        Og1       = M0*(3*Om/4+Orad)/(Og-M0)
        Ot1       = 1-Om-Orad-Ol-Ok-Og1
        ds1       = (1-Om-Orad-Ol-Ok-Og1+(Os-1)*s1**2)*(Og-M0)
        ds1       = ds1+s1**2+0.5*Os*s1**2*(1+(0.5*Os-1)*s1**2-Ok)
        ds1       = 2*np.sqrt(ds1)-s1
        self.Sg1  = ds1/s1
        return

    
    def print_parameters(self):
        print('\n --- Cosmological parameters ---------------------')
        print(' Ω_m                  = %11.3e  []'%self.Om)
        print(' Ω_r                  = %11.3e  []'%self.Orad)
        print(' Ω_Λ                  = %11.3e  []'%self.Ol)
        print(' Ω_K                  = %11.3e  []'%self.Ok)
        print(' h                    = %11.3e  []'%self.h)
        print(' Ω_s                  = %11.3e  []'%self.Os)
        print(' Ω_g                  = %11.3e  []'%self.Og)
        print(' s(τ=1)               = %11.3e  []'%self.s1)
        print(' Σ(τ=1)               = %11.3e  []'%self.Sg1)
        print(' -------------------------------------------------')
        return
    
    
    #GR system of equations    
    def FL_equations_GR(self, x, y):
        yp   = self.yp

        z    = y[0]
        H    = y[1]
        if z < 0 or z > self.mv:
            return yp, 1
        
        z1 = z+1
        z2 = z1**2
        z3 = z1**3
        H2 = H**2
        M  = 0.25*self.Om*z3+self.Ol
        yp[0] = -z1*H
        yp[1] = -2*H2+0.5*self.Om*z3+2*self.Ol+self.Ok*z2
        return yp, 0


    #CCGG system of equations
    def FL_equations_CG(self, x, y):
        Om   = self.Om
        Orad = self.Orad
        Ol   = self.Ol
        Ok   = self.Ok
        Os   = self.Os
        Og   = self.Og
        Os2  = Os/2
        yp   = self.yp
        mv   = self.mv
        switch = self.switch
      
        z  = y[0]
        H  = y[1]
        s  = y[2]
        w  = y[3]
        if z < 0 or z > mv or abs(H) > mv or abs(s) > mv:
            return yp, 1
        
        z1 = z+1
        z2 = z1**2
        z3 = z1**3
        H2 = H**2
        s2 = s**2
        M  = 0.25*Om*z3+Ol
        yp[0] = -z1*H
        yp[1] = -2*H2+2*M+Ok*z2-(Os-1)*s2
        if switch == 0:
            V0 = -Om*z1-Orad*z2-Ol/z2
            ks = (H2*s2-M*(0.75*Om*z3+Orad*z1**4)
                  +Os2*s2*(H2+(Os2-1)*s2-Ok*z2)
                  +(Og-M)*(H2+V0*z2-Ok*z2+(Os-1)*s2))
            if ks > self.prc:
                yp[2] = -H*s+self.eps*2*np.sqrt(ks)
            else: 
                yp[2] = w*s
                self.switch = 1
        s2Hw = s2*(H+w)
        if abs(s2Hw) > self.prc and abs(w) < mv:
            E = 1.5*Om*H*z3-s2*(5*H-(2*Os-1)*w)
            E *= (H2-Ok*z2-2*M+(Os-1)*s2)/s2Hw
            yp[3] = -w**2-2*s2+5*H2-3*H*w+4*Og*(Os-1)-2*Ok*z2+E         
        elif switch == 1: return yp, 2
        return yp, 0


    def RK4(self, f, x, y):
        #4th order Runge/Kutta
        #Abramowitz/Stegun (1964/1972), p.896, 25.5.10
        m = len(x)         
        h = x[1]-x[0]
        err  = 0
        nerr = -1
        
        for i in range(1, m):
            yh = y[:, i-1]
            k1, err1 = f(x[i-1],     yh)
            k1 = k1*h
            k2, err2 = f(x[i-1]+h/2, yh+k1/2)
            k2 = k2*h
            k3, err3 = f(x[i-1]+h/2, yh+k2/2)
            k3 = k3*h
            k4, err4 = f(x[i-1]+h,   yh+k3)
            k4 = k4*h
            y[:,i] = y[:,i-1]+(k1+2*k2+2*k3+k4)/6

            if err1+err2+err3+err4 != 0:
                #print('RK4: ', err1,err2,err3,err4)
                err  = 1
                nerr = i
                break
        return err, nerr


    def solve_FLeq(self):
        AX  = self.AX
        m   = AX.m
        y   = self.y
        neq = self.neq
        ode_sys = self.FL_equations_CG if self.GR == 'n' else self.FL_equations_GR
        
        # Initial condition for z,H,s,w at t=1
        y0 = [0,1,self.s1,self.Sg1]
        
        imin = 0
        m1  = AX.get_index(1)
        miv = AX.get_interval(m1)
        k = m1
        y[:,k] = y0
        for i in range(miv,-1,-1):
            xi = AX.AXI[i].xrev
            yi = np.zeros((neq, len(xi)))
            yi[:,0] = y[:,k]
            err, nerr = self.RK4(ode_sys, xi, yi)
            for j in range(AX.AXI[i].m):
                self.y[:,k-j] = yi[:,j]
            if err != 0: 
                imin = k-nerr
                break
            k -= AX.AXI[i].m-1

        # Validity range of solution
        self.imin = m-1
        for i in range(m-2,imin-1,-1):
            if y[0,i] > 0 and y[0,i] < self.mv:
                self.imin = i
            else: break
        print(y[0,self.imin:self.imin+3])
        print(1/(y[0,self.imin:self.imin+3]+1))

        # reset for next run
        self.switch = 0
        return

    
    def draw_zHs(self, xmin, xmax, ymin, ymax):
        imin = self.imin
        imax = self.imax
        a = np.zeros(self.AX.m)
        for i in range(imin,imax):
            a[i] = 1/(self.y[0,i]+1)
        x = self.AX.x
        y = self.y
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel(r'$\tau$')
        plt.ylabel('$z,H,s,a$')
        plt.xscale('linear')
        ax.axvline(x[imin],color="grey", ls='dotted', zorder=-1)
        ax.axvline(1.,color="grey", ls='dotted', zorder=-1)
        ax.axhline(0.,color="grey", ls='dotted', zorder=-1)
        ax.axhline(1.,color="grey", ls='dotted', zorder=-1)
        plt.plot(x[imin:imax], y[0,imin:imax], label='z')
        plt.plot(x[imin:imax], y[1,imin:imax], label='H')
        plt.plot(x[imin:imax], y[2,imin:imax], label='s')
        plt.plot(x[imin:imax], a[imin:imax], label='a')
        plt.legend()
        plt.show()
        return


# In[20]:


# distance modulus
class MDLS:

    def __init__(self, FL, GR_asym, Ol_GR):
        self.FL      = FL
        self.GR_asym = GR_asym
        self.Ol_GR   = Ol_GR
        self.zh      = np.zeros(FL.AX.m)
        self.Hh      = np.zeros(FL.AX.m)
        self.muh     = np.zeros(FL.AX.m)
        self.rcd     = np.zeros(FL.AX.m)
        self.ztr     = 1089 # transparency
        self.rtr     = 0
        self.z       = 0
        self.Hz      = 0
        self.mz      = 0
        self.mu      = 0
        self.izmin   = 0
        self.izmax   = 0
        self.rs      = 0
        self.R       = 0
        self.la      = 0
    

    # distance modulus
    def calc_mu(self):
        FL  = self.FL
        m   = FL.AX.m
        zh  = self.zh
        Hh  = self.Hh
        muh = self.muh

        for i in range(m):
            zh[i] = FL.y[0,m-1-i]
            Hh[i] = FL.y[1,m-1-i]

        # Identify domain where z grows strictly monotonely
        self.izmin = 0
        izmax = m-FL.imin
        self.izmax = izmax
        for i in range(1,izmax):
            if zh[i] < zh[i-1]:
                self.izmax = i-1
                break
             
        muh[self.izmin] = 0
        for i in range(self.izmin+1,self.izmax):
            muh[i] = muh[i-1]+0.5*(zh[i]-zh[i-1])*(1/Hh[i]+1/Hh[i-1])/self.FL.H0
        for i in range(self.izmin+1,self.izmax):
            muh[i] = 5*np.log10((1+zh[i])*muh[i]/1.5637382e38)+25
            
        #self.izmin += 1
        self.z  = zh[self.izmin:self.izmax]
        self.Hz = Hh[self.izmin:self.izmax]
        self.mu = muh[self.izmin:self.izmax]
        self.mz = len(self.z)
        return


    # S_k(r)
    def S_k(self, Ok, r):
        H0  = self.FL.H0
        res = 0
        if abs(Ok) < self.FL.prc:
            res = r
        elif Ok < 0:
            KHC = np.sqrt(-Ok)*H0
            res = np.sin(KHC*r)/KHC
        else:
            KHC = np.sqrt(Ok)*H0
            res = np.sinh(KHC*r)/KHC
        return res
   
    
    # CMB distance priors
    def calc_R_la(self):
        z    = self.z
        Hz   = self.Hz
        H0   = self.FL.H0
        mz   = self.mz
        rcd  = self.rcd
        Om   = self.FL.Om
        Ok   = self.FL.Ok
        ztr  = self.ztr  
        H02  = 0.5/H0
       
        # co-moving distance rcd(z)
        rcd[0] = 0
        ntr = 0
        for i in range(1,mz):
            dz2 = H02*(z[i]-z[i-1])
            rcd[i] = rcd[i-1]+dz2*(1/Hz[i]+1/Hz[i-1])
            if ntr == 0 and z[i] > ztr: ntr = i-1
        if ntr == 0: return 0,0,1
        self.rtr = rcd[ntr]+(ztr-z[ntr])*(rcd[ntr+1]-rcd[ntr])/(z[ntr+1]-z[ntr])
        #self.rtr = quad_interpol(z[ntr-1:ntr+2], rcd[ntr-1:ntr+2], ztr)
        print(" ntr,z[ntr-1:ntr+2],ztr: ",ntr,z[ntr-1:ntr+2],ztr)
        print(" rcd[ntr-1:ntr+2],rtrq,rtrl: ",rcd[ntr-1:ntr+2],self.rtr)
        
        # sound horizon rs
        self.rs  = 0
        igr0 = 1/(np.sqrt(1+660/(1+z[ntr]))*Hz[ntr])
        for i in range(ntr+1,mz):
            dz2 = (z[i]-z[i-1])/2
            igr = 1/(np.sqrt(1+660/(1+z[i]))*Hz[i])
            self.rs += dz2*(igr+igr0)
            igr0 = igr
        self.rs /= (H0*np.sqrt(3))
    
        # R, la
        skr    = self.S_k(Ok, self.rtr)
        self.R = np.sqrt(Om)*H0*skr
        if abs(self.rs) < self.FL.prc: self.la = 0
        else: self.la = np.pi*skr/self.rs      
        return

  
    def print_results(self):
        RPl  = 1.7661
        laPl = 301.7293
        print('\n --- Results -------------------------------------')
        print(' zmin                 = %11.3e  []'%self.z[0])
        print(' zmax                 = %11.3e  []'%self.z[-1])
        print(' z(transparency)      = %11.3e  []'%self.ztr)
        print(' r(transparency)      = %11.3e  []'%self.rtr)
        print(' H(zmin)              = %11.3e  []'%self.Hz[0])
        print(' H(zmax)              = %11.3e  []'%self.Hz[-1])
        print(' r_sound              = %11.3e  []'%self.rs)
        print(' R                    = %11.3e  []'%self.R)
        print(' R(Planck)            = %11.3e  []'%RPl)
        print(' la                   = %11.3e  []'%self.la)
        print(' la(Planck)           = %11.3e  []'%laPl)
        print(' -------------------------------------------------')
        return


    def draw_H(self, xmin, xmax, ymin, ymax):
        z   = self.z
        Hz  = self.Hz
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel('$z$')
        plt.ylabel('$H(z)$')
        ax.axvline(z[-1], color="grey", ls='dotted', zorder=-1)
        ax.plot(z, Hz,'k')
        plt.show()
        return


    def draw_rcd(self, xmin, xmax, ymin, ymax):
        z   = self.z
        rcd = self.rcd[self.izmin:self.izmax]
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel('$z$')
        plt.ylabel(r'$r_{cd} \, (z)$')
        ax.axvline(self.ztr, color="grey", ls='dotted', zorder=-1)
        ax.plot(z, rcd, 'k', label=r'$r_{cd} \, (z)$')
        ax.plot(self.ztr, self.rtr, 'x', color='r', label=r'$r_{cd} \, (z_{tr})$')
        plt.legend()
        plt.show()
        return

    
    def draw_mu(self, xmin, xmax, ymin, ymax, SNe_redshift, SNe_distmod, SNe_distmod_unc):
        redshift = SNe_redshift
        distmod = SNe_distmod
        distmod_unc = SNe_distmod_unc
        z   = self.z
        mu  = self.mu
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.errorbar(redshift, distmod, yerr=distmod_unc, capsize=3, fmt='r.', label='Pantheon data')
        plt.plot(z, mu, color='b',label=r'$\mu$')
        plt.xlabel('$z$')
        plt.ylabel(r'$\mu \, (z)$')
        plt.legend()
        plt.show()
        return


# # MCMC

# In[ ]:




