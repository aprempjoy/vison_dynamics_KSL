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
import params as p
#L1 = 20 # system length along e1
#L2 = 20 # system length along e2
#M = 0 # skewness of periodic boundary
#
Jx = 2*p.Jx
Jy = 2*p.Jy
Jz = 2*p.Jz
L1 = p.L1
u = p.u
M=p.M
L2=p.L2
N_unit=p.N_unit
class Bond():
    def __init__(self):
        """Initialize bond configuration."""
        self.bond = np.array([[u,u,u] for i in range(L1*L2)])
        
    def create_vortex_x(self,i):
        """create vortex at plaquette labelled by left bottom corner i with gauge string along x"""
        xp,yp = position(i)
        for x in range(0,xp+1):
            self.bond[index(x,yp)][1] *= -1
            
    def create_vortex_y(self,i):
        """create vortex at plaquette labelled by left bottom corner i with gauge string along y"""
        xp,yp = position(i)
        for y in range(0,yp+1):
            self.bond[index(xp,y)][0] *= -1  
        
    def create_vortex_z(self,i):
        """create vortex at plaquette labelled by left bottom corner i with gauge string along y"""
        xp,yp = position(i)
        x=xp
        y=yp+1
        while x+1<L1 and y>0:
            self.bond[index(x+1,y-1)][2] *= -1
            x += +1
            y += -1
    
    def flip_x(self,i):
        self.bond[i][0] *= -1
    def flip_y(self,i):
        self.bond[i][1] *= -1
    def flip_z(self,i):
        self.bond[i][2] *= -1

#def index(x, y, dr=(0,0)):
#    """Return the index of a unit cell."""
#    x0 = x+dr[0]
#    y0 = y+dr[1]
#    x1 = x0%L1
#    y1 = y0%L2
#    return x1+L1*y1
#
#
#def position(i, dr=(0,0)):
#    """Return the coordinate of a unit cell."""
#    x0 = (i%L1)+dr[0]
#    y0 = (i//L1)+dr[1]
#    return (x0%L1, y0%L2)

def index(x, y, dr=(0,0)):
    """Return the index of a unit cell."""
    x0 = x+dr[0]
    y0 = y+dr[1]
    x1 = (x0-M*(y0//L2))%L1
    y1 = y0%L2
    return x1+L1*y1


def position(i, dr=(0,0)):
    """Return the coordinate of a unit cell."""
    x0 = (i%L1)+dr[0]
    y0 = (i//L1)+dr[1]
    return ((x0-M*(y0//L2))%L1, y0%L2)

def nn_1(i):
    """Return the index of the nearest neighbor unit cell along e1."""
    x, y = position(i)
    return index(x+1,y)


def nn_2(i):
    """Return the index of the nearest neighbor unit cell along e2."""
    x, y = position(i)
    return index(x,y+1)
def plaquette(i,a):
    """Calculate flux through plaquette i"""
    x0,y0 = position(i)
    W = a[i][nn_1(i)]*a[nn_1(i)][nn_1(i)]*a[nn_1(i)][nn_2(nn_1(i))]*a[nn_2(i)][nn_2(nn_1(i))]*a[nn_2(i)][nn_2(i)]*a[i][nn_2(i)]
    return W
def check_flux(vortex_array,x_flip,y_flip,z_flip,dtype=int):
    x_p = []
    y_p = []
    W = []
    c=0
    A=vison_per(vortex_array,x_flip,y_flip,z_flip)[0]
    plt.figure()
    for i in range(L1*L2):
        x,y = position(i)
        x_p.append(x)
        y_p.append(y)
        W.append(plaquette(i,A))
        c += 1
    plt.scatter(x_p,y_p,c=W)
    plt.colorbar()
    plt.show()
    return
def min_energy(bond):
    """Calculate minimum energy.
    Args:
        bond: an instance of Bond or array[L1*L2][3].
    """
    N_unit = L1*L2
    coupling = bond.bond if isinstance(bond, Bond) else bond

    # Create matrix A where +K in spin model becomes -Ku_{ij} c_i c_j in the majorana basis
    a = np.zeros((N_unit, N_unit), dtype=float)
    for i in range(N_unit):
        a[i][nn_1(i)] += -Jx*coupling[i][0]
        a[i][nn_2(i)] += -Jy*coupling[i][1]
        a[i][i] += -Jz*coupling[i][2]
    #print(a)
    u,s,vt = sl.svd(a)
    U=u
    V=np.transpose(vt)
    sgn = -np.prod(np.sign(coupling))
    ## from boundary condition
    #if (L1+L2+M*(L1-M))%2 != 0: sgn *= -1 # (-1)^theta
    ## det(Q) = det(VU)
    sgn *= ((-1)**(L1+L2+M*(L1-M)))*np.linalg.slogdet(U)[0]*np.linalg.slogdet(V)[0]
    return a,sgn,s,U,V
def vortex_energy_all():
    e_vortex = []
    v = []
    for x in range(10,L1-10):
        for y in range(10,L2-10):
            i = index(x,y)
            v.append(i)
            e_vortex.append(main_cir(np.array([i]))[1])
            print(i,plaquette(i,main_cir(np.array([i]))[1]))
    plt.plot(v,e_vortex)
    plt.show()
    
"""ground state energy for two flux sector as a function of separation between them"""
def vortex_int():   
    plt.figure()
    e0 = -0.5*sum(main_per([0])[1])
    for y in range(0,L2):
        x0 = L1//2
        v = index(x0,y)
#        print("{0},{1}\n {2}, {3}".format(v1,plaquette(v1,main_cir(v)[0]),v2,plaquette(v2,main_cir(v)[0])))
        plt.scatter(y,-0.5*sum(main_per([v])[1])-e0,c="r")
        plt.scatter(y,max(main_per([v])[1]),c="b")
    plt.show()
    return

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
#A,E_per,U_per,V_per = main_per([0])
#A0,E_per0,U_per0,V_per0 = main_per([0])
##%%
#x_A =[]
#x_B=[]
#y_A=[]
#y_B=[]
#L = np.arange(L1*L2)
#for m in L:
#        x_A.append((position(m)[0]+position(m)[1])*0.5)
#        y_A.append((position(m)[0]-position(m)[1])*np.sqrt(3)/2)
#        x_B.append((position(m)[0]+position(m)[1])*0.5)
#        y_B.append((position(m)[0]-position(m)[1])*np.sqrt(3)/2 + 1/np.sqrt(3))
#m=899
#fig,Axes = plt.subplots()
#Axes.set_aspect('equal')
#plt.scatter(x_A,y_A,c=(U_per0[:,m]),s=5)
#plt.scatter(x_B,y_B,c=(V_per0[:,m]),s=5)
#plt.colorbar()
##%%
#q1 = tuple(2*np.pi*np.array([1,1/np.sqrt(3)]))
#q2 = tuple(2*np.pi*np.array([-1,1/np.sqrt(3)]))
#n1 = (1/2,np.sqrt(3)/2)
#n2 = (-1/2,np.sqrt(3)/2)
#
#def E_kitaev(kx,ky):
#    k=(kx,ky)
#    return(abs(Jz+Jx*np.exp(1j*np.dot(k,n1))+Jy*np.exp(1j*np.dot(k,n2))))
##%%
#    k_x=[]
#    k_y=[]
#    Elevels=[]
#    for i in range(L1):
#        for j in range(L2):
#            kx,ky = tuple(map(add,tuple(map(mul,q1,(i/L1,i/L1))),tuple(map(mul,q2,(j/L2,j/L2)))))
#            Elevels.append(E_kitaev(kx,ky))
#            k_x.append(kx)
#            k_y.append(ky)
##%%
#fig = plt.figure()
#plt.scatter(range(len(E_per)),(E_per),s=5)
#plt.scatter(range(len(E_per0)),(E_per0),s=5)
#plt.scatter(range(len(Elevels)),np.sort(Elevels))
##%%
#xxA = np.ravel(x_A); yyA = np.ravel(y_A) ; zzA = np.ravel(U_an)
#xxB = np.ravel(x_B); yyB = np.ravel(y_B) ; zzB = np.ravel(V_an)
#plt.scatter(xxA,yyA,c=zzA,cmap=cm.Reds)
#plt.scatter(xxB,yyB,c=zzB,cmap=cm.Reds)
#plt.colorbar()
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_trisurf(x_A,y_A,U[:,m], cmap = cm.jet)
#ax.plot_trisurf(x_B,y_B,V[:,m], cmap = cm.jet)
#plt.colorbar()
