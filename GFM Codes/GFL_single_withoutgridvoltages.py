# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 20:06:07 2022

Summmary : This will have only one CCI with grid and PLL dynamics And LCL filter

We will have the following three inverters
1. Current COntrolled Inverters (INV1)
- gamma_d
- gamma_q
- power controller integrator

3. Line dynamics
- Id
- Iq

We do not have any PLL dynamis
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
from scipy.integrate import odeint
from numpy.linalg import eig

#Number of module in series
Ns =1;

# Inverter paramertres
S_rated = 500;
V_nom =  30*np.sqrt(2);
V_grid = V_nom;

I_nom = 2*S_rated/(3*V_grid);
omega_nom = 2*60*np.pi;
Rf = 1;
Lf = 1.2e-3;

Pref1 = 100;
phi_ref = 0;

# curret control
BW_cc = 2*np.pi*500;

kpi = Lf*BW_cc;
kii = Rf*BW_cc;

# power control for CCI 
kp_pwr = (BW_cc/10)/(1.5*V_nom);
ki_pwr = (BW_cc/100)*kp_pwr;

# x = [0=id, 1=iq, 2=delta, 3=V, 4=omega]
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


def f(x,param):
    
    V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref = param;

    Id, Iq, gamma_d, gamma_q, gamma_pwr = x;

    # algebraic substitutions
   
    Vq = kpi*(0-Iq) + gamma_q + Lf*omega_nom*Id;
    Vd = (1/(1+kpi*kp_pwr*1.5*Id))*(kp_pwr*kpi*(Pref1-1.5*Vq*Iq) + kpi*(gamma_pwr-Id) + gamma_d - Lf*omega_nom*Iq);

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))+ gamma_pwr;
 
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_nom*Iq + Vd - V_grid), # Id
              (1/Lf)*(-Rf*Iq -Lf*omega_nom*Id + Vq ),          # Iq
              kii*(Idref - Id),                                # gamma_d
              kii*(0 - Iq),                                    # gamma_q
              ki_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))]             # gamma_pwr
    return dx_dt

x_init = [0,0,0,0,0];

param = [V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref]
equilibrium, infodict,ie,msg = fsolve(f, x_init, param, full_output=1 )
        
[eig_val,eig_vect] = eig(infodict["fjac"]);
Participation_Factor_Eval(infodict["fjac"],0.001);
eig_real = [ele.real for ele in eig_val];
eig_imag = [ele.imag for ele in eig_val];

plt.figure(2)
plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
plt.grid()

print("Success ?",ie)


#############################################################################

# SImulate the system
# Comment out this part if you do not want to simulate

def diff(x, t, V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref):
 
    Id, Iq, gamma_d, gamma_q, gamma_pwr = x;

    # algebraic substitutions
   
    Vq = kpi*(0-Iq) + gamma_q + Lf*omega_nom*Id;
    Vd = (1/(1+kpi*kp_pwr*1.5*Id))*(kp_pwr*kpi*(Pref1-1.5*Vq*Iq) + kpi*(gamma_pwr-Id) + gamma_d - Lf*omega_nom*Iq);

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))+ gamma_pwr;
 
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_nom*Iq + Vd - V_grid), # Id
              (1/Lf)*(-Rf*Iq -Lf*omega_nom*Id + Vq ),          # Iq
              kii*(Idref - Id),                                # gamma_d
              kii*(0 - Iq),                                    # gamma_q
              ki_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))]             # gamma_pwr
    return dx_dt

tstart = 0
tstop = 0.01
increment = 0.0001
tstart_plot = 0#int((10/100)*tstop/increment)
t = np.arange(tstart,tstop,increment)
x_init = [0,0,0,0,0];
res = odeint(diff, x_init, t, args=(V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref))
Id_disp = res[tstart_plot:-1,0]
Iq_disp = res[tstart_plot:-1,1]
gamma_pwr_disp = res[tstart_plot:-1,4]
gamma_d1_disp = res[tstart_plot:-1,2]
gamma_q1_disp = res[tstart_plot:-1,3]
t = t[tstart_plot:-1];
# Plot the Results
plt.figure(3)
plt.plot(t,Id_disp)
plt.plot(t,Iq_disp)
plt.title('Currents')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["Id", "Iq"],loc='upper right')
plt.show()
# plt.figure(4)
# Vd_disp = 
# P = 1.5*()
# plt.plot(t,Vcd_disp)
# plt.plot(t,Vcq_disp)
# plt.title('Voltages')
# plt.xlabel('t')
# plt.ylabel('x(t)')
# plt.grid()
# #plt.axis([-1, 1, -1.5, 1.5])
# plt.legend(["Vcd", "Vcq"],loc='upper right')
# plt.show()
plt.figure(5)
plt.plot(t,gamma_pwr_disp)
plt.plot(t,gamma_d1_disp)
plt.plot(t,gamma_q1_disp)
plt.title('Integration variables')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.legend(["gamma_pwr", "gamma_d","gamma_Q"],loc='upper right')
plt.show()



