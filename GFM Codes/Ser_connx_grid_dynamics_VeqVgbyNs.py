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
    
Modification :
    Removed the Voltage dynamics, V = Vg/Ns
@author: rhlma
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig
#Number of module in series
Ns =3;

# Inverter paramertres
S_rated = 2000;
V_nom =  30*np.sqrt(2);
V_grid = V_nom*Ns;

I_nom = 2*S_rated/(3*V_grid);
omega_nom = 2*60*np.pi;
Rf = 0.4;
Lf = 1.5e-3;
Rg = 0.8;
Lg = 2.4e-3;
Kdv = 4.5;
Kdi = 1;
kpi = Lg*1000;
kii = Rg*1000;
E = V_grid;
Pref1 = 1000;
Pref2 = 500;
Pref3 = 250;
phi_ref = 0;

# dVOC parameters
eta = 10000;
mu  = 0.1;

# x = [0=id, 1=iq, 2=delta, 3=V, 4=omega]

def f(x,param):
    
    V_nom , omega_nom, eta, mu, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, Kdi, kpi, kii, Kdv= param;

    gamma_d1 = x[0];
    gamma_q1 = x[1];
    th2 = x[2];
    th3 = x[3];
    Id1  = x[4];
    Iq1  = x[5];
    

    # algebraic substitutions
    th1 = 0-phi_ref; # we assume th_pll = 0

    Id2 = Id1*np.cos(th2-th1)+Iq1*np.sin(th2-th1);
    Iq2 = Iq1*np.cos(th2-th1)-Id1*np.sin(th2-th1);
    
    Id3 = Id1*np.cos(th3-th1)+Iq1*np.sin(th3-th1);
    Iq3 = Iq1*np.cos(th3-th1)-Id1*np.sin(th3-th1);
    
    Vq1 = kpi*(0 - Iq1) + kii*gamma_q1;
    Vd1 = (1/(1+1.5*kpi*Kdi*Id1))*(kpi*Kdi*(Pref1 - 1.5*Vq1*Iq1) + kii*gamma_d1); #Iqref1 = 0
  
    Idref1 = Kdi*(Pref1 - 1.5*(Vd1*Id1+Vq1*Iq1));
    Iqref1 = 0;
    
    V2 = E/Ns;
    V3 = E/Ns;
    
    Vd2 = V2*np.cos(th2-th1);
    Vq2 = V2*np.sin(th2-th1);
    
    Vd3 = V3*np.cos(th3-th1);
    Vq3 = V3*np.sin(th3-th1);  
    
    P2 = 1.5*(Vd2*Id2+Vq2*Iq2);
    Q2 = 1.5*(Vq2*Id2-Vd2*Iq2);
    
    P3 = 1.5*(Vd3*Id3+Vq3*Iq3);
    Q3 = 1.5*(Vq3*Id3-Vd3*Iq3);
    
   
    dx_dt = [ Idref1 - Id1,
              Iqref1 - Iq1,
              omega_nom - Kdv*np.arctan2(Q2,P2),
              omega_nom - Kdv*np.arctan2(Q3,P3),
              (1/Lg)*(-Rg*Id1 + Vd1 + Vd2 + Vd3 - E*np.cos(phi_ref)),
              (1/Lg)*(-Rg*Iq1 + Vq1 + Vq2 + Vq3 - E*np.sin(phi_ref))]
  
    return dx_dt

x_init = [0,0,0,0,0,0];

param = [V_nom , omega_nom, eta, mu, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, Kdi, kpi, kii, Kdv]
equilibrium, infodict,ie,msg = fsolve(f, x_init, param, full_output=1)
        
[eig_val,eig_vect] = eig(infodict["fjac"]);
eig_real = [ele.real for ele in eig_val];
eig_imag = [ele.imag for ele in eig_val];

plt.figure(1)
plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
plt.grid()

print("Success ?",ie)



