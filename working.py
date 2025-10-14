#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:38:27 2022

@author: apjoy
"""

import kitaev_h as kh
import kitaev_open as ko
import kitaevOpen_h as koh
import kitaev_periodic as kp
from importlib import reload
from functions import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import pfaff
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)
"x0=0,K3=.1,L1=70,L2=18"
#%%
reload(kh)
reload(kp)
x0,y0=0,kp.L2//2+2
v0=kp.index(x0,y0)
y_loop=[kp.index(1,i) for i in range(kp.L2)]
H0,T0,p0,E0,U0,V0 = kh.vison_per([kp.index(kp.L1-1,1)],[],[],[v0+6])
#%%
H1,T1,p1,E1,U1,V1 = kh.vison_per([kp.index(kp.L1-1,1)],[],[v0],[])
H2,T2,p2,E2,U2,V2 = kh.vison_per([kp.index(kp.L1-1,1)],[kp.index(x0-1,y0+1)],[],[])
H3,T3,p3,E3,U3,V3 = kh.vison_per([kp.index(kp.L1-1,1)],[],[],[kp.index(x0-1,y0+1)])
H4,T4,p4,E4,U4,V4 = kh.vison_per([kp.index(kp.L1-1,1)],[],[kp.index(x0-1,y0)],[])
H5,T5,p5,E5,U5,V5 = kh.vison_per([kp.index(kp.L1-1,1)],[kp.index(x0-1,y0)],[],[])
H6,T6,p6,E6,U6,V6 = kh.vison_per([kp.index(kp.L1-1,1)],[],[],[v0])
#%%
H2p,T2p,p2p,E2p,U2p,V2p = kh.vison_per([kp.index(kp.L1-1,1)],[kp.index(x0,y0+1)],[v0,kp.index(x0,y0+1)],[])
H3p,T3p,p3p,E3p,U3p,V3p = kh.vison_per([kp.index(kp.L1-1,1)],[kp.index(x0-1,y0+1),kp.index(x0-2,y0+2)],[],[kp.index(x0-1,y0+2)])
H4p,T4p,p4p,E4p,U4p,V4p = kh.vison_per([kp.index(kp.L1-1,1)],[],[kp.index(x0-2,y0+1)],[kp.index(x0-1,y0+1),kp.index(x0-2,y0+1)])
H5p,T5p,p5p,E5p,U5p,V5p = kh.vison_per([kp.index(kp.L1-1,1)],[kp.index(x0-2,y0)],[kp.index(x0-1,y0),kp.index(x0-1,y0-1)],[])
H6p,T6p,p6p,E6p,U6p,V6p = kh.vison_per([kp.index(kp.L1-1,1)],[kp.index(x0-1,y0),kp.index(x0,y0-1)],[],[kp.index(x0,y0-1)])
H1p,T1p,p1p,E1p,U1p,V1p = kh.vison_per([kp.index(kp.L1-1,1)],[],[kp.index(x0+1,y0-1)],[v0,v0+1])
#%%
X_1,Y_1,F_1,t_1 = Overlap2(U0,U1,V0,V1)
X_2,Y_2,F_2,t_2 = Overlap2(U0,U2,V0,V2)
X_3,Y_3,F_3,t_3 = Overlap2(U0,U3,V0,V3)
X_4,Y_4,F_4,t_4 = Overlap2(U0,U4,V0,V4)
X_5,Y_5,F_5,t_5 = Overlap2(U0,U5,V0,V5)
X_6,Y_6,F_6,t_6 = Overlap2(U0,U6,V0,V6)
X_1p,Y_1p,F_1p,t_1p = Overlap2(U0,U1p,V0,V1p)
X_2p,Y_2p,F_2p,t_2p = Overlap2(U0,U2p,V0,V2p)
X_3p,Y_3p,F_3p,t_3p = Overlap2(U0,U3p,V0,V3p)
X_4p,Y_4p,F_4p,t_4p = Overlap2(U0,U4p,V0,V4p)
X_5p,Y_5p,F_5p,t_5p = Overlap2(U0,U5p,V0,V5p)
X_6p,Y_6p,F_6p,t_6p = Overlap2(U0,U6p,V0,V6p)
#%%
X_21,Y_21,F_21,t_21 = Overlap2(U1,U2,V1,V2)
X_32,Y_32,F_32,t_32 = Overlap2(U2,U3,V2,V3)
X_43,Y_43,F_43,t_43 = Overlap2(U3,U4,V3,V4)
X_54,Y_54,F_54,t_54 = Overlap2(U4,U5,V4,V5)
X_65,Y_65,F_65,t_65 = Overlap2(U5,U6,V5,V6)
X_16,Y_16,F_16,t_16 = Overlap2(U6,U1,V6,V1)
X_2p1,Y_2p1,F_2p1,t_2p1 = Overlap2(U1,U2p,V1,V2p)
X_3p2,Y_3p2,F_3p2,t_3p2 = Overlap2(U2,U3p,V2,V3p)
X_4p3,Y_4p3,F_4p3,t_4p3 = Overlap2(U3,U4p,V3,V4p)
X_5p4,Y_5p4,F_5p4,t_5p4 = Overlap2(U4,U5p,V4,V5p)
X_6p5,Y_6p5,F_6p5,t_6p5 = Overlap2(U5,U6p,V5,V6p)
X_1p6,Y_1p6,F_1p6,t_1p6 = Overlap2(U6,U1p,V6,V1p)
print(t_21,t_32,t_43,t_54,t_65,t_16)
print(t_2p1,t_3p2,t_4p3,t_5p4,t_6p5,t_1p6)

#%%
tpf21 = pfaff(F_1,F_2,t_1,t_2)
tpf32 = pfaff(F_2,F_3,t_2,t_3)
tpf43 = pfaff(F_3,F_4,t_3,t_4)
tpf54 = pfaff(F_4,F_5,t_4,t_5)
tpf65 = pfaff(F_5,F_6,t_5,t_6)
tpf16 = pfaff(F_6,F_1,t_6,t_1)

print(tpf21)
#%%
tpf2p1 = pfaff(F_1,F_2p,t_1,t_2p)
tpf3p2 = pfaff(F_2,F_3p,t_2,t_3p)
tpf4p3 = pfaff(F_3,F_4p,t_3,t_4p)
tpf5p4 = pfaff(F_4,F_5p,t_4,t_5p)
tpf6p5 = pfaff(F_5,F_6p,t_5,t_6p)
tpf1p6 = pfaff(F_6,F_1p,t_6,t_1p)
#%%
thop_21=odd_odd_overlap(-1,-1,X_21,Y_21,F_21)*tpf21*1j+odd_odd_overlap(-1,-1,X_2p1,Y_2p1,F_2p1)*tpf2p1*1j
print(abs(thop_21))
thop_32=odd_odd_overlap(-1,-1,X_32,Y_32,F_32)*tpf32*1j+odd_odd_overlap(-1,-1,X_3p2,Y_3p2,F_3p2)*tpf3p2*1j
print(abs(thop_32))
thop_43=odd_odd_overlap(-1,-1,X_43,Y_43,F_43)*tpf43*1j+odd_odd_overlap(-1,-1,X_4p3,Y_4p3,F_4p3)*tpf4p3*1j
print(abs(thop_43))
thop_54=odd_odd_overlap(-1,-1,X_54,Y_54,F_54)*tpf54*1j+odd_odd_overlap(-1,-1,X_5p4,Y_5p4,F_5p4)*tpf5p4*1j
print(abs(thop_54))
thop_65=odd_odd_overlap(-1,-1,X_65,Y_65,F_65)*tpf65*1j+odd_odd_overlap(-1,-1,X_6p5,Y_6p5,F_6p5)*tpf6p5*1j
print(abs(thop_65))
thop_16=odd_odd_overlap(-1,-1,X_16,Y_16,F_16)*tpf16*1j+odd_odd_overlap(-1,-1,X_1p6,Y_1p6,F_1p6)*tpf1p6*1j
print(abs(thop_16))
#%%
phi_1=np.angle(thop_21*thop_32*thop_43*thop_54*thop_65*thop_16)
print(phi_1)
phi_1_list.append([kh.K,phi_1])
#%%
plt.plot(np.array(sorted(phi_1_list)).T[0],np.array(sorted(phi_1_list)).T[1]+(2),'o-')
#%%
#with open('phi_1_listPBC.txt', 'w') as f:
#    csv.writer(f, delimiter=',').writerows(phi_1_list)
#%%
"vison creation"
H,T,p,E,U,V = kh.vison_per([kp.index(kp.L1-1,1)],[],[],[])
#%%
X_0,Y_0,F_0,t_0 = Overlap2(U0,U,V0,V)
X_1,Y_1,F_1,t_1 = Overlap2(U0,U1,V0,V1)
X_2,Y_2,F_2,t_2 = Overlap2(U0,U2,V0,V2)
X_3,Y_3,F_3,t_3 = Overlap2(U0,U3,V0,V3)
X_4,Y_4,F_4,t_4 = Overlap2(U0,U4,V0,V4)
X_5,Y_5,F_5,t_5 = Overlap2(U0,U5,V0,V5)
X_6,Y_6,F_6,t_6 = Overlap2(U0,U6,V0,V6)

X_10,Y_10,F_10,t_10 = Overlap2(U,U1,V,V1)
X_20,Y_20,F_20,t_20 = Overlap2(U,U2,V,V2)
X_30,Y_30,F_30,t_30 = Overlap2(U,U3,V,V3)
X_40,Y_40,F_40,t_40 = Overlap2(U,U4,V,V4)
X_50,Y_50,F_50,t_50 = Overlap2(U,U5,V,V5)
X_60,Y_60,F_60,t_60 = Overlap2(U,U6,V,V6)

print(t_10,t_20,t_30,t_40,t_50,t_60)
#%%
tpf10 = pfaff(F_0,F_1,t_0,t_1)
tpf20 = pfaff(F_0,F_2,t_0,t_2)
tpf30 = pfaff(F_0,F_3,t_0,t_3)
tpf40 = pfaff(F_0,F_4,t_0,t_4)
tpf50 = pfaff(F_0,F_5,t_0,t_5)
tpf60 = pfaff(F_0,F_6,t_0,t_6)

print(tpf10,tpf20,tpf30,tpf40,tpf50,tpf60)
#%%
thop1=(odd_overlapNA_a(v0,-1,U,V,X_10)[1]-
   odd_overlapNA_b(kp.index(x0,y0+1),-1,U,V,X_10)[1])*tpf10*1j
print(abs(thop1))
thop2=(odd_overlapNA_a(kp.index(x0-1,y0+1),-1,U,V,X_20)[1]-
   odd_overlapNA_b(kp.index(x0,y0+1),-1,U,V,X_20)[1])*tpf20*1j
print(abs(thop2))
thop3=(odd_overlapNA_a(kp.index(x0-1,y0+1),-1,U,V,X_30)[1]-
   odd_overlapNA_b(kp.index(x0-1,y0+1),-1,U,V,X_30)[1])*tpf30*1j
print(abs(thop3))
thop4=(odd_overlapNA_a(kp.index(x0-1,y0),-1,U,V,X_40)[1]-
   odd_overlapNA_b(kp.index(x0-1,y0+1),-1,U,V,X_40)[1])*tpf40*1j
print(abs(thop4))
thop5=(odd_overlapNA_a(kp.index(x0-1,y0),-1,U,V,X_50)[1]-
   odd_overlapNA_b(kp.index(x0,y0),-1,U,V,X_50)[1])*tpf50*1j
print(abs(thop5))
thop6=(odd_overlapNA_a(kp.index(x0,y0),-1,U,V,X_60)[1]-
   odd_overlapNA_b(kp.index(x0,y0),-1,U,V,X_60)[1])*tpf60*1j
print(abs(thop6))
#%%
flux_012=np.angle(thop1*thop_21*np.conj(thop2))
flux_023=np.angle(thop2*thop_32*np.conj(thop3))
flux_034=np.angle(thop3*thop_43*np.conj(thop4))
flux_045=np.angle(thop4*thop_54*np.conj(thop5))
flux_056=np.angle(thop5*thop_65*np.conj(thop6))
flux_061=np.angle(thop6*thop_16*np.conj(thop1))
print(flux_012,flux_023,flux_034,flux_045,flux_056,flux_061)
#%%
thop_vac_list.append([kh.K,thop_21])
bp_vac_list.append([kh.K,np.angle(berry_phase)/np.pi])
##%%
#plt.figure()
#plt.plot(np.array(bp_vac_list).T[0],np.array(bp_vac_list).T[1],'.')
###plt.plot(np.array(thop_vac_list).T[0],np.angle(np.array(thop_vac_list).T[1])/np.pi,'.-')
#plt.figure()
#plt.plot(np.array(thop_vac_list).T[0],abs(np.array(thop_vac_list).T[1]),'.')
#%%
"nn-nn by Gamma hopping vacuum sector"
reload(kh)
reload(kp)
x0,y0=kp.L1//2-1,kp.L2//2
v0=kp.index(x0,y0)
Href,Tref,pref,Eref,Uref,Vref = kh.vison_per([0],[0],[0],[v0+6])
#%%
H1,T1,p1,E1,U1,V1 = kh.vison_per([],[],[v0],[])
H2,T2,p2,E2,U2,V2 = kh.vison_per([],[],[],[kp.index(x0-1,y0+1)])
#%%
X_1,Y_1,F_1,t_1 = Overlap2(U0,U1,V0,V1)
X_2,Y_2,F_2,t_2 = Overlap2(U0,U2,V0,V2)
X_21,Y_21,F_21,t_21 = Overlap2(U1,U2,V1,V2)
tpf21 = pfaff(F_1,F_2,t_1,t_2)
thop_21=odd_odd_overlap(-1,-1,X_21,Y_21,F_21)*tpf21*1j+odd_odd_overlap(-1,-1,X_2p1,Y_2p1,F_2p1)*tpf2p1*1j
print(abs(thop_21))
