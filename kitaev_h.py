# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:19:56 2019

@author: APREM
"""

#!/usr/bin/env python

"""Exact eigen-energy of the Kitaev model."""

__author__ = "Aprem"
__version__ = "2.0.0"

import sys
import numpy as np
import scipy.linalg as sl
#from matplotlib import cm
import matplotlib.pyplot as plt
from operator import mul,add
import kitaev_periodic as kp
import params as p
from importlib import reload
#L1 = 20 # system length along e1
#L2 = 20 # system length along e2
#M = 0 # skewness of periodic boundary
#
reload(p)
Jx = p.Jx
Jy = p.Jy
Jz = p.Jz
K = p.K3
L1 = p.L1
M=0
L2=p.L2
N_unit=p.N_unit
class Bond():
    def __init__(self):
        """Initialize bond configuration."""
        self.bond = np.array([[Jx,Jy,Jz] for i in range(L1*L2)])
        
    def create_vortex_x(self,i):
        """create vortex at plaquette labelled by left bottom corner i with gauge string along x"""
        xp,yp = kp.position(i)
        for x in range(0,xp+1):
            self.bond[kp.index(x,yp)][1] *= -1
            
    def create_vortex_y(self,i):
        """create vortex at plaquette labelled by left bottom corner i with gauge string along y"""
        xp,yp = kp.position(i)
        for y in range(0,yp+1):
            self.bond[kp.index(xp,y)][0] *= -1  
        
    def create_vortex_z(self,i):
        """create vortex at plaquette labelled by left bottom corner i with gauge string along y"""
        xp,yp = kp.position(i)
        x=xp
        y=yp+1
        while x+1<L1 and y>0:
            self.bond[kp.index(x+1,y-1)][2] *= -1
            x += +1
            y += -1
    
    def flip_x(self,i):
        self.bond[i][0] *= -1
    def flip_y(self,i):
        self.bond[i][1] *= -1
    def flip_z(self,i):
        self.bond[i][2] *= -1
def nnn_1(i):
    return kp.nn_1(i)
def nnn_2(i):
    return kp.nn_2(i)
def nnn_3(i):
    x,y = kp.position(i)
    return kp.index(x-1,y+1)
def nnn_4(i):
    x,y = kp.position(i)
    return kp.index(x-1,y)
def nnn_5(i):
    x,y = kp.position(i)
    return kp.index(x,y-1)
def nnn_6(i):
    x,y = kp.position(i)
    return kp.index(x+1,y-1)
nnn = [nnn_1,nnn_2,nnn_3,nnn_4,nnn_5,nnn_6]
def plaquette(i,a):
    """Calculate flux through plaquette i"""
    x0,y0 = kp.position(i)
    W = a[i][kp.nn_1(i)]*a[kp.nn_1(i)][kp.nn_1(i)]*a[kp.nn_1(i)][kp.nn_2(kp.nn_1(i))]*a[kp.nn_2(i)][kp.nn_2(kp.nn_1(i))]*a[kp.nn_2(i)][kp.nn_2(i)]*a[i][kp.nn_2(i)]
    return W
def check_flux(vortex_array,x_flip,y_flip,z_flip,dtype=int):
    x_p = []
    y_p = []
    W = []
    c=0
    A=vison_per(vortex_array,x_flip,y_flip,z_flip)[0]
    plt.figure()
    for i in range(L1*L2):
        x,y = kp.position(i)
        x_p.append(x)
        y_p.append(y)
        W.append(plaquette(i,A))
        c += 1
    plt.scatter(x_p,y_p,c=W)
    plt.colorbar()
    plt.show()
    return 
def dag(X):
    return np.conj(np.transpose(X))
def min_energy(bond):
    """Calculate minimum energy.
    Args:
        bond: an instance of Bond or array[L1*L2][3].
    """
    coupling = bond.bond if isinstance(bond, Bond) else bond

    # Create matrix A
    a = np.zeros((N_unit, N_unit), dtype=float)
    for i in range(N_unit):
        a[i][kp.nn_1(i)] += coupling[i][0]
        a[i][kp.nn_2(i)] += coupling[i][1]
        a[i][i] += coupling[i][2]
#    a[0][0]*=.7 #break C6 symmetry
    #print(a)
#    H11 = np.zeros((N_unit,N_unit),float)
#    H1 = np.concatenate((H11,a),axis=1)
#    H2 = np.concatenate((-np.transpose(a),H11),axis=1)
#    Ham = np.concatenate((H1,H2),axis=0)
    Haa = np.zeros((N_unit,N_unit),float)
    Hbb = np.zeros((N_unit,N_unit),float)
    for i in range(N_unit):
        Haa[i][nnn_1(i)] += -K*a[i][nnn_1(i)]*a[nnn_1(i)][nnn_1(i)]#use J=1 for this to be correct
        Haa[i][nnn_2(i)] += K*a[i][nnn_2(i)]*a[nnn_2(i)][nnn_2(i)]
        Haa[i][nnn_3(i)] += -K*a[i][nnn_2(i)]*a[nnn_3(i)][nnn_2(i)]
        Haa[i][nnn_4(i)] += K*a[nnn_4(i)][i]*a[i][i]
        Haa[i][nnn_5(i)] += -K*a[i][i]*a[nnn_5(i)][i]
        Haa[i][nnn_6(i)] += K*a[i][nnn_1(i)]*a[nnn_6(i)][nnn_1(i)]
        
        Hbb[i][nnn_1(i)] += K*a[i][i]*a[i][nnn_1(i)]
        Hbb[i][nnn_2(i)] += -K*a[i][i]*a[i][nnn_2(i)]
        Hbb[i][nnn_3(i)] += K*a[nnn_4(i)][i]*a[nnn_4(i)][nnn_3(i)]
        Hbb[i][nnn_4(i)] += -K*a[nnn_4(i)][i]*a[nnn_4(i)][nnn_4(i)]
        Hbb[i][nnn_5(i)] += K*a[nnn_5(i)][i]*a[nnn_5(i)][nnn_5(i)]
        Hbb[i][nnn_6(i)] += -K*a[nnn_5(i)][i]*a[nnn_5(i)][nnn_6(i)]
    h = a+np.transpose(a)+1j*(Haa+Hbb)
    delta = np.transpose(a)-a+1j*(Haa-Hbb)
    H1 = np.concatenate((h,delta),axis=1)
    H2 = np.concatenate((np.conj(np.transpose(delta)),-np.transpose(h)),axis=1)
    Ham = np.concatenate((H1,H2),axis=0)
#    Hm1 = np.concatenate((Haa,a),axis=1)
#    Hm2 = np.concatenate((-np.conj(np.transpose(a)),Hbb),axis=1)
#    Hm = np.concatenate((H1,H2),axis=0)
    w,v = np.linalg.eigh(Ham)   # Inv(w)=dagger(w)
    sgn = np.prod(np.sign(coupling))
    ## from boundary condition
    if (L1+L2+M*(L1-M))%2 != 0: sgn *= -1 # (-1)^theta
    ## det(Q) = det(VU)
    U = dag(v)[:kp.N_unit,:kp.N_unit]
    V = dag(v)[:kp.N_unit:,kp.N_unit:]
    #T = np.concatenate((T1,np.flipud(T2)),axis=0)
    Y = U
    X = V
    T = np.concatenate((np.concatenate((U,V),axis=1),np.concatenate((np.conj(V),np.conj(U)),axis=1)),axis=0)
#    sgn *= np.linalg.slogdet(v)[0]
#    idx = w.argsort()[::1]   
#    w = w[idx]
#    v = v[:,idx]
#    U = v[:N_unit,:N_unit]/np.sqrt(0.5)
#    V = -1j*v[N_unit:2*N_unit,:N_unit]/np.sqrt(0.5)
    return Ham,T,sgn,w,X,Y
def vortex_energy_all():
    e_vortex = []
    v = []
    for x in range(10,L1-10):
        for y in range(10,L2-10):
            i = kp.index(x,y)
            v.append(i)
            e_vortex.append(main_cir(np.array([i]))[2])
            print(i,plaquette(i,main_cir(np.array([i]))[2]))
    plt.plot(v,e_vortex)
    plt.show()
    
"""ground state energy for two flux sector as a function of separation between them"""
def vortex_int():   
    plt.figure()
    e_d=[]
    e0 = -0.5*sum(abs(vison_per([0],[0],[0],[0])[2])/2)
    for x in range(0,L1//2):
        print(x)
        v = kp.index(x,L2//2)
#        print("{0},{1}\n {2}, {3}".format(v1,plaquette(v1,main_cir(v)[0]),v2,plaquette(v2,main_cir(v)[0])))
        e_d.append([x,vison_per([v],[0],[0],[0])[2][kp.N_unit]/2])
        #plt.scatter(y,max(vison_per([v],[0],[0],[0])[1]),c="b")
    return e_d

def main_per(vortex_array,dtype = int):
    global L1, L2, Jz
    #print("# L1={0} L2={1} M={2} Jz={3} Jx=Jy=1.0".format(L1, L2, M, Jz))
    bond = Bond()
    for vortex in vortex_array:
        if vortex > 0:
            bond.create_vortex_y(vortex)
    
    return(min_energy(bond))
def vison_per(v_0,x_flip,y_flip,z_flip,dtype = int):
    #print("# L1={0} L2={1} M={2} Jz={3} Jx=Jy=1.0".format(L1, L2, M, Jz))
    bond = Bond()
    for vison in v_0:
        if vison > 0:
            bond.create_vortex_x(vison)   #create the initial vison at v_0 using a y-string
    for x in x_flip:
        if x>0:
            bond.flip_x(x)
    for y in y_flip:
        if y>0:
            bond.flip_y(y)
    for z in z_flip:
        if z>0:
            bond.flip_z(z)
    return(min_energy(bond))

#%%

