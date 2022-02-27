# -*- coding: utf-8 -*-

"""
Created on Mon Aug 16 19:04:20 2021


We will have the following three inverters
1. Current COntrolled Inverters (INV1)
- gamma_d
- gamma_q
- power controller integrator

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

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig
#Number of module in series
Ns =3;

# Inverter paramertres
S_rated = 1000;
V_nom =  50*np.sqrt(2);
V_grid = V_nom;

I_nom = 2*S_rated/(3*V_grid);
omega_nom = 2*60*np.pi;
Rg = 4;
Lg = 2.4e-3;


E = V_grid;
Pref1 = 1000;
phi_ref = 0;
Idref = 0;

# curret control
BW_cc = 2*np.pi*500;

kpi = Lg*BW_cc;
kii = Rg*BW_cc;

# power control for CCI 
kp_pwr = (BW_cc/10)/(1.5*V_nom);
ki_pwr = (BW_cc/10)*kp_pwr;

# x = [0=id, 1=iq, 2=delta, 3=V, 4=omega]
def Participation_Factor_Eval(A,epsi):

# Returns the partciipation factor of a matrix with a thresholding 

    [D,W] = eig(A); # we refer to W as the right eigen vectors (Machowski pp 530)

    U = np.transpose(np.linalg.inv(W));  # left eigen vectors

    PF  = np.zeros((len(A),len(A)));
    
    for i in range(0,len(A)):
        for j in range(0,len(A)):
            aa = np.abs(U[i][j]*W[i][j]);
            if(aa<epsi):
                 PF[i][j] = 0;
            else:
                 PF[i][j] = aa;

    #VV = np.diag(D); # displays the eigen values
    print('\n Participation Factor Code:');
    print('\n Eigen values ',D);

    print(PF) # displays the participation factors
    plt.figure(1)
   # A= np.diag(np.array([1,2,3,4,5]));
    [m,n] = np.shape(PF);
    plt.imshow(PF, alpha=0.8, cmap ="Greens")
    plt.xticks(np.arange(n))
    plt.yticks(np.arange(m))
    plt.xlabel('Eigen value')
    plt.ylabel('State variable')
    plt.title('Color Maps: Partcipation factor')
    plt.show


def f(x,param):
    
    V_nom, E , omega_nom, Rg, Lg, Pref1, E,kp_pwr, ki_pwr, kpi, kii= param;

    Id, Iq, gamma_d, gamma_q, gamma_pwr = x;

    # algebraic substitutions
    
    Vq = kpi*(0-Iq)+kii*gamma_q;
    Vd = (1/((1/kpi)+kp_pwr*1.5*Id))*(kp_pwr*(Pref1-1.5*Vq*Iq)+ki_pwr*gamma_pwr+(kii*gamma_d)/kpi);

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))+ki_pwr*gamma_pwr;
    
    dx_dt = [ (1/Lg)*(-Rg*Id + Vd - E),
             (1/Lg)*(-Rg*Iq + Vq),
              Idref - Id,
              0 - Iq,
              Pref1 - 1.5*(Vd*Id+Vq*Iq)]
    return dx_dt

x_init = [0,0,0,0,0];

param = [V_nom, E , omega_nom, Rg, Lg, Pref1, E,kp_pwr, ki_pwr, kpi, kii]
equilibrium, infodict,ie,msg = fsolve(f, x_init, param, full_output=1)
        
[eig_val,eig_vect] = eig(infodict["fjac"]);
Participation_Factor_Eval(infodict["fjac"],0.001);
eig_real = [ele.real for ele in eig_val];
eig_imag = [ele.imag for ele in eig_val];

plt.figure(2)
plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
plt.grid()

print("Success ?",ie)



