# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 19:04:20 2021


We will have the following three inverters
1. Current COntrolled Inverters (INV1)
- gamma_d
- gamma_q
- gamma_pwr

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

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
        
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
   
    np.set_printoptions(precision=3)
    print('\n\n Eigen values ');
    print(D.T)
    print('\n\n Participation Factor Code:');
    matprint(PF)
    # displays the participation factors
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
Rg = 0.4;
Lg = 2.4e-3;
Kdv = 4.5;

Kp_pwr = 4.5;
Ki_pwr = 100;

kpi = Lg*1000;
kii = Rg*1000;

E = V_grid;
Pref1 = 100;
Pref2 = 100;
Pref3 = 100;
phi_ref = 0;

# dVOC parameters
eta = 1;
mu  = 1;

# x = [0=id, 1=iq, 2=delta, 3=V, 4=omega]

def f(x,param):
    
    V_nom , omega_nom, eta, mu, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, Kp_pwr, Ki_pwr, kpi, kii, Kdv= param;

    Id1, Iq1, gamma_d1, gamma_q1, gamma_pwr, V2, th2, V3, th3 = x;

    # algebraic substitutions
    th1 = 0-phi_ref; # we assume th_pll = 0

    Id2 = Id1*np.cos(th2-th1)+Iq1*np.sin(th2-th1);
    Iq2 = Iq1*np.cos(th2-th1)-Id1*np.sin(th2-th1);
    
    Id3 = Id1*np.cos(th3-th1)+Iq1*np.sin(th3-th1);
    Iq3 = Iq1*np.cos(th3-th1)-Id1*np.sin(th3-th1);
    
    Vq1 = kpi*(0-Iq1)+kii*gamma_q1;
    Vd1 = (1/((1/kpi)+Kp_pwr*1.5*Id1))*(Kp_pwr*(Pref1-1.5*Vq1*Iq1)+Ki_pwr*gamma_pwr+(kii*gamma_d1)/kpi);

    Idref1 = Kp_pwr*(Pref1 - 1.5*(Vd1*Id1+Vq1*Iq1))+Ki_pwr*gamma_pwr;
    Iqref1 = 0;
    
    Vd2 = V2*np.cos(th2-th1);
    Vq2 = V2*np.sin(th2-th1);
    
    Vd3 = V3*np.cos(th3-th1);
    Vq3 = V3*np.sin(th3-th1);  
    
    P2 = 1.5*(Vd2*Id2+Vq2*Iq2);
    Q2 = 1.5*(Vq2*Id2-Vd2*Iq2);
    
    P3 = 1.5*(Vd3*Id3+Vq3*Iq3);
    Q3 = 1.5*(Vq3*Id3-Vd3*Iq3);
    
   
    dx_dt = [ (1/Lg)*(-Rg*Id1 + Vd1 + Vd2 + Vd3 - E*np.cos(phi_ref)),
              (1/Lg)*(-Rg*Iq1 + Vq1 + Vq2 + Vq3 - E*np.sin(phi_ref)),
              Idref1 - Id1,
              Iqref1 - Iq1,
              Pref1 - 1.5*(Vd1*Id1+Vq1*Iq1),
              mu*V2*(V_nom**2-V2**2) - (2*eta/(3*V2))*(P2-Pref2),
              omega_nom - Kdv*np.arctan2(Q2,P2),
              mu*V3*(V_nom**2-V3**2) - (2*eta/(3*V3))*(P3-Pref3),
              omega_nom - Kdv*np.arctan2(Q3,P3)]
  
    return dx_dt

x_init = [Pref1/(1.5*V_nom),0,0,0,0,V_nom,0,V_nom,0];

param = [V_nom , omega_nom, eta, mu, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, Kp_pwr, Ki_pwr, kpi, kii, Kdv]
equilibrium, infodict,ie,msg = fsolve(f, x_init, param, full_output=1)
        
[eig_val,eig_vect] = eig(infodict["fjac"]);
Participation_Factor_Eval(infodict["fjac"],0.001);
eig_real = [ele.real for ele in eig_val];
eig_imag = [ele.imag for ele in eig_val];

plt.figure(2)
plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
plt.grid()

print("Success ?",ie)

