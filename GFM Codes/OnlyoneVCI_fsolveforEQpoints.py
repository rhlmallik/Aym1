# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 19:04:20 2021


We will have the following three inverters
1. Current COntrolled Inverters (INV1)
- gamma_d
- gamma_q


2. Voltage Controlled Inverters (INV2,3)
- theta
- voltage (maybe?)

3. Line dynamics
- Id
- Iq

Controller design :
    Current controller: Kp, Ki based on plant inversion
    The droop constants for the voltage is same for VCI 1,2
    The droop constants for the current controller is different
@author: rhlma
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig
#Number of module in series
Ns =1;

# Inverter paramertres
S_rated = 1000;
V_nom =  30*np.sqrt(2);
V_grid = V_nom*Ns;

I_nom = 2*S_rated/(3*V_grid);
omega_nom = 2*60*np.pi;
Rf = 0.4;
Lf = 1.2e-3;
Kdv = 0.45;
kp_v = 34;
E = V_grid;
Pref1 = 1000;
phi = np.arctan(Lf*omega_nom/Rf);
Zf = np.sqrt(Rf**2+(Lf*omega_nom)**2);



# x = [0=id, 1=iq, 2=delta, 3=V, 4=omega]

def f(x,param):
    
    V_nom , omega_nom, Lf, Rf, Pref1, kp_v, Kdv= param;

    Id = x[0];
    Iq = x[1];
    theta = x[2];
   
    dx_dt = [(1/Lf)*(-Rf*Id + Lf*omega_nom*Iq + (V_nom + kp_v*Pref1)*np.cos(theta)/(1+1.5*kp_v*(Id*np.cos(theta)+Iq*np.sin(theta) )) - V_nom),
            (1/Lf)*(-Rf*Iq - Lf*omega_nom*Id + (V_nom + kp_v*Pref1)*np.sin(theta)/(1+1.5*kp_v*(Id*np.cos(theta)+Iq*np.sin(theta) )) ),
            - Kdv*np.arctan2(Id*np.sin(theta)-Iq*np.cos(theta),Id*np.cos(theta)+Iq*np.sin(theta))]
    
    return dx_dt

# select four starting condtion from solving the equations by hand
x_init = np.array([[13.8589, -4.0005,0],
                [ 24.3632, 80.8617,0],
                [-42.7536, 60.0560,0],
                [-46.1541, 32.5128,0]]);


for i in range(0,len(x_init)):
    param = [V_nom , omega_nom, Lf, Rf, Pref1, kp_v, Kdv]
    print("We are solving for the following Id,Iq:",x_init[i,:])
    equilibrium, infodict,ie,msg = fsolve(f, x_init[i,:], param, full_output=1)

    Idsol = equilibrium[0];
    Iqsol = equilibrium[1];
    thetasol = equilibrium[2];

    Vsol = V_nom*np.sin(thetasol+phi)/np.sin(phi)
    P = 1.5*(Vsol**2*np.cos(phi)/Zf)-1.5*Vsol*V_nom*np.cos(thetasol+phi)/Zf
    Pcheck = ((V_nom+kp_v*Pref1)-Vsol)/kp_v
    Q = 1.5*Vsol**2*np.sin(phi)/Zf - 1.5*Vsol*V_nom*np.sin(thetasol+phi)/Zf

    print("P:",P,",  Q:",Q,",   Id:",Idsol,",  Iq:",Iqsol,",    V:",Vsol,",     theta",thetasol)

    [eig_val,eig_vect] = eig(infodict["fjac"]);
    eig_real = [ele.real for ele in eig_val];
    eig_imag = [ele.imag for ele in eig_val];

    plt.figure(1)
    plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
    plt.grid()

    print("Success ?",ie)

