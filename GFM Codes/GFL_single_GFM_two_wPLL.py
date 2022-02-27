# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:31:22 2022


We will have the following three inverters
1. Current COntrolled Inverters (INV1)
- gamma_d
- gamma_q
- power controller integrator
- pll dynamics

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
Ns = 3;

# Inverter paramertres
S_rated = 500;
V_nom =  30*np.sqrt(2);
V_grid = Ns*V_nom;

I_nom = 2*S_rated/(3*V_grid);
omega_nom = 2*60*np.pi;
Rf = 1;
Lf = 1.2e-3;

Pref1 = 500;
Pref2 = 200;
Pref3 = 100;
phi_ref = 0;

# curret control
BW_cc = 2*np.pi*100;
kpi = Lf*BW_cc;
kii = Rf*BW_cc;

# power control for CCI 
kp_pwr = (BW_cc/10)/(1.5*V_nom);
ki_pwr = (BW_cc/100)*kp_pwr;

# PLL
BW_pll = 2*np.pi*30;
PM     = 80;# in degrees # condition is tan(PM)^2>> 1??
kp_pll =  BW_pll;
Ti_pll = np.tan(PM*np.pi/180)/kp_pll;
ki_pll = kp_pll/Ti_pll;

# GFM controller
Kdv = 4.5;
# dVOC parameters
eta = 100;
mu  = 0.001;

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
    
    V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3, kp_pll, ki_pll = param;

    Id, Iq, gamma_d, gamma_q, gamma_pwr, th2, V2, th3, V3, gamma_vq, th_pll, theta_g = x;

    # algebraic substitutions
   
    Vq = kpi*(0-Iq) + gamma_q + Lf*omega_nom*Id;
    Vd = (1/(1+kpi*kp_pwr*1.5*Id))*(kp_pwr*kpi*(Pref1-1.5*Vq*Iq) + kpi*(gamma_pwr-Id) + gamma_d - Lf*omega_nom*Iq);

    th1 = np.arctan2(Vq,Vd); # this is the angle the inverter 1 makes with respect to the grid rotating frame

    Id2 = Id*np.cos(th2-th1)+Iq*np.sin(th2-th1);
    Iq2 = Iq*np.cos(th2-th1)-Id*np.sin(th2-th1);
    
    Id3 = Id*np.cos(th3-th1)+Iq*np.sin(th3-th1);
    Iq3 = Iq*np.cos(th3-th1)-Id*np.sin(th3-th1);

    Vd2 = V2*np.cos(th2-th1);
    Vq2 = V2*np.sin(th2-th1);
    
    Vd3 = V3*np.cos(th3-th1);
    Vq3 = V3*np.sin(th3-th1); 
    
    P2 = 1.5*(Vd2*Id2+Vq2*Iq2);
    Q2 = 1.5*(Vq2*Id2-Vd2*Iq2);
    
    P3 = 1.5*(Vd3*Id3+Vq3*Iq3);
    Q3 = 1.5*(Vq3*Id3-Vd3*Iq3);    

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq)) + gamma_pwr;
    
    Vga = V_grid*np.cos(theta_g);
    Vgb = V_grid*np.cos(theta_g-(2*np.pi/3));
    Vgc = V_grid*np.cos(theta_g+(2*np.pi/3));
    
    Vgq =(-Vga*np.sin(th_pll)-Vgb*np.sin(th_pll-(2*np.pi/3))-Vgc*np.sin(th_pll+(2*np.pi/3)));
    
    Vgd = (Vga*np.cos(th_pll)+Vgb*np.cos(th_pll-(2*np.pi/3))+Vgc*np.cos(th_pll+(2*np.pi/3)));
    
    V_grid_d = Vgd*np.cos(th_pll-th1)+Vgq*np.sin(th_pll-th1);
    V_grid_q = Vgq*np.cos(th_pll-th1)-Vgd*np.sin(th_pll-th1);
    
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_nom*Iq + Vd + Vd2 + Vd3 - V_grid_d), # Id, eqn in GFL frame of ref
              (1/Lf)*(-Rf*Iq - Lf*omega_nom*Id + Vq + Vq2 + Vq3 - V_grid_q),          # Iq
              kii*(Idref - Id),                                # gamma_d
              kii*(0 - Iq),                                    # gamma_q
              ki_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq)),                # gamma_pwr
              - Kdv*np.arctan2(Q2,P2),
              mu*V2*(V_nom**2-V2**2) - (2*eta/(3*V2))*(P2-Pref2),
              - Kdv*np.arctan2(Q3,P3),
              mu*V3*(V_nom**2-V3**2) - (2*eta/(3*V3))*(P3-Pref3),
              ki_pll*Vgq/V_grid,                                     # gamma_vq 
              kp_pll*Vgq/V_grid + gamma_vq,              # theta1 
              omega_nom]            
    return dx_dt

x_init = [0,0,0,0,0,0,V_nom,0,V_nom, 0 , 0 , 0];



#############################################################################

# SImulate the system
# You then analyze stability with initial condition derived from steady state of the simulation.

def diff(x, t, V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3, kp_pll, ki_pll):
 
    Id, Iq, gamma_d, gamma_q, gamma_pwr, th2, V2, th3, V3, gamma_vq, th_pll, theta_g = x;

    # algebraic substitutions
   
    Vq = kpi*(0-Iq) + gamma_q + Lf*omega_nom*Id;
    Vd = (1/(1+kpi*kp_pwr*1.5*Id))*(kp_pwr*kpi*(Pref1-1.5*Vq*Iq) + kpi*(gamma_pwr-Id) + gamma_d - Lf*omega_nom*Iq);

    th1 = np.arctan2(Vq,Vd); # this is the angle the inverter 1 makes with respect to the grid rotating frame

    Id2 = Id*np.cos(th2-th1)+Iq*np.sin(th2-th1);
    Iq2 = Iq*np.cos(th2-th1)-Id*np.sin(th2-th1);
    
    Id3 = Id*np.cos(th3-th1)+Iq*np.sin(th3-th1);
    Iq3 = Iq*np.cos(th3-th1)-Id*np.sin(th3-th1);

    Vd2 = V2*np.cos(th2-th1);
    Vq2 = V2*np.sin(th2-th1);
    
    Vd3 = V3*np.cos(th3-th1);
    Vq3 = V3*np.sin(th3-th1); 
    
    P2 = 1.5*(Vd2*Id2+Vq2*Iq2);
    Q2 = 1.5*(Vq2*Id2-Vd2*Iq2);
    
    P3 = 1.5*(Vd3*Id3+Vq3*Iq3);
    Q3 = 1.5*(Vq3*Id3-Vd3*Iq3);    

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq)) + gamma_pwr;
    
    Vga = V_grid*np.cos(theta_g);
    Vgb = V_grid*np.cos(theta_g-(2*np.pi/3));
    Vgc = V_grid*np.cos(theta_g+(2*np.pi/3));
    
    Vgq =(-Vga*np.sin(th_pll)-Vgb*np.sin(th_pll-(2*np.pi/3))-Vgc*np.sin(th_pll+(2*np.pi/3)));
    
    Vgd = (Vga*np.cos(th_pll)+Vgb*np.cos(th_pll-(2*np.pi/3))+Vgc*np.cos(th_pll+(2*np.pi/3)));
    
    V_grid_d = Vgd*np.cos(th_pll-th1)+Vgq*np.sin(th_pll-th1);
    V_grid_q = Vgq*np.cos(th_pll-th1)-Vgd*np.sin(th_pll-th1);
    
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_nom*Iq + Vd + Vd2 + Vd3 - V_grid_d), # Id, eqn in GFL frame of ref
              (1/Lf)*(-Rf*Iq - Lf*omega_nom*Id + Vq + Vq2 + Vq3 - V_grid_q),          # Iq
              kii*(Idref - Id),                                # gamma_d
              kii*(0 - Iq),                                    # gamma_q
              ki_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq)),                # gamma_pwr
              - Kdv*np.arctan2(Q2,P2),
              mu*V2*(V_nom**2-V2**2) - (2*eta/(3*V2))*(P2-Pref2),
              - Kdv*np.arctan2(Q3,P3),
              mu*V3*(V_nom**2-V3**2) - (2*eta/(3*V3))*(P3-Pref3),
              ki_pll*Vgq/V_grid,                                     # gamma_vq 
              kp_pll*Vgq/V_grid + gamma_vq,              # theta1 
              omega_nom]            
    return dx_dt

tstart = 0
tstop = 1
increment = 0.0001
tstart_plot = 0#int((10/100)*tstop/increment)
t = np.arange(tstart,tstop,increment)
res = odeint(diff, x_init, t, args=(V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3, kp_pll, ki_pll))
Id_disp = res[tstart_plot:-1,0]
Iq_disp = res[tstart_plot:-1,1]
gamma_d1_disp = res[tstart_plot:-1,2]
gamma_q1_disp = res[tstart_plot:-1,3]
gamma_pwr_disp = res[tstart_plot:-1,4]
th2_disp = res[tstart_plot:-1,5]
V2_disp = res[tstart_plot:-1,6]
th3_disp = res[tstart_plot:-1,7]
V3_disp = res[tstart_plot:-1,8]
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
plt.figure(6)
plt.plot(t,th2_disp,"+")
plt.plot(t,th3_disp,'-')
plt.title('Angles')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["th_2","th_3"],loc='upper right')
plt.show()
plt.figure(7)
plt.plot(t,V2_disp)
plt.plot(t,V3_disp)
plt.title('Voltages')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["V_2","V_3"],loc='upper right')
plt.show()

x_init = res[-1,:]
param = [V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3, kp_pll, ki_pll]
equilibrium, infodict,ie,msg = fsolve(f, x_init, param, full_output=1 )
        
[eig_val,eig_vect] = eig(infodict["fjac"]);
Participation_Factor_Eval(infodict["fjac"],0.001);
eig_real = [ele.real for ele in eig_val];
eig_imag = [ele.imag for ele in eig_val];

plt.figure(2)
plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
plt.grid()

print("Success ?",ie)


