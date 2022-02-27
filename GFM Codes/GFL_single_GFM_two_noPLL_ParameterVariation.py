# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:35:16 2022


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

    In this we try to vary the parameter and obtain the set where the solution converges and is stable.
@author: rhlma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
from numpy.linalg import eig

#Number of module in series
Ns = 3;
nfig = 0;

# Inverter paramertres
S_rated = 500;
V_nom =  30*np.sqrt(2);
V_grid = Ns*V_nom;

I_nom = 2*S_rated/(3*V_grid);
omega_nom = 2*60*np.pi;
Rf = 1;
Lf = 1.2e-3;

Pref1 = 500;
Pref2 = 500;
Pref3 = 100;
phi_ref = 0;

# curret control
BW_cc = 2*np.pi*100;
kpi = Lf*BW_cc;
kii = Rf*BW_cc;

# power control for CCI 
kp_pwr = (BW_cc/10)/(1.5*V_nom);
ki_pwr = (BW_cc/100)*kp_pwr;

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
    # print('\n\n Participation Factor Code:');
    # matprint(PF)
    # displays the participation factors
    global nfig
    nfig = nfig +1;
    plt.figure(nfig)
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
    
    V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3 = param;

    Id, Iq, gamma_d, gamma_q, gamma_pwr, th2, V2, th3, V3 = x;

    # algebraic substitutions
   
    Vq = kpi*(0-Iq) + gamma_q + Lf*omega_nom*Id;
    Vd = (1/(1+kpi*kp_pwr*1.5*Id))*(kp_pwr*kpi*(Pref1-1.5*Vq*Iq) + kpi*(gamma_pwr-Id) + gamma_d - Lf*omega_nom*Iq);

    th1 = np.arctan2(Vq,Vd); # this is the angle the inverter 1 makes with respect to the grid rotating frame

# Currents are [Id,Iq] in the 1st converter frame of reference.
# In the converter i frame of reference, currents are x_convi = x_conv_1 e^(-j(theta_i-theta_1)) 

    Id2 = Id*np.cos(th2-th1)+Iq*np.sin(th2-th1);
    Iq2 = Iq*np.cos(th2-th1)-Id*np.sin(th2-th1);
    
    Id3 = Id*np.cos(th3-th1)+Iq*np.sin(th3-th1);
    Iq3 = Iq*np.cos(th3-th1)-Id*np.sin(th3-th1);

# Voltages are [Vi,0] in the ith converter frame of reference.
# In the converter 1 frame of reference, voltages are x_conv1 = x_conv_i e^(j(theta_i-theta_1)) 
    Vd2 = V2*np.cos(th2-th1);
    Vq2 = V2*np.sin(th2-th1);
    
    Vd3 = V3*np.cos(th3-th1);
    Vq3 = V3*np.sin(th3-th1); 
    
    V_grid_d = V_grid*np.cos(-th1);
    V_grid_q = V_grid*np.sin(-th1);
    
    P2 = 1.5*V2*Id2;#1.5*(Vd2*Id2+Vq2*Iq2);
    Q2 = -1.5*V2*Iq2;#1.5*(Vq2*Id2-Vd2*Iq2);
    
    P3 = 1.5*V3*Id3;#1.5*(Vd3*Id3+Vq3*Iq3);
    Q3 = -1.5*V3*Iq3;#1.5*(Vq3*Id3-Vd3*Iq3);    

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq)) + gamma_pwr;
 
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_nom*Iq + Vd + Vd2 + Vd3 - V_grid_d), # Id, eqn in GFL frame of ref
              (1/Lf)*(-Rf*Iq - Lf*omega_nom*Id + Vq + Vq2 + Vq3 - V_grid_q),          # Iq
              kii*(Idref - Id),                                # gamma_d
              kii*(0 - Iq),                                    # gamma_q
              ki_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq)),                # gamma_pwr
              - Kdv*np.arctan2(Q2,P2),
              mu*V2*(V_nom**2-V2**2) - (2*eta/(3*V2))*(P2-Pref2),
              - Kdv*np.arctan2(Q3,P3),
              mu*V3*(V_nom**2-V3**2) - (2*eta/(3*V3))*(P3-Pref3),]            
    return dx_dt

x_init = [0,0,0,0,0,0,V_nom,0,V_nom];



#############################################################################

# SImulate the system
# Comment out this part if you do not want to simulate

def diff(x, t, V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3):
 
    Id, Iq, gamma_d, gamma_q, gamma_pwr, th2, V2, th3, V3 = x;

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
    
    V_grid_d = V_grid*np.cos(-th1);
    V_grid_q = V_grid*np.sin(-th1);
    
    P2 = 1.5*V2*Id2;#1.5*(Vd2*Id2+Vq2*Iq2);
    Q2 = -1.5*V2*Iq2;#1.5*(Vq2*Id2-Vd2*Iq2);
    
    P3 = 1.5*V3*Id3;#1.5*(Vd3*Id3+Vq3*Iq3);
    Q3 = -1.5*V3*Iq3;#1.5*(Vq3*Id3-Vd3*Iq3);    

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))+ gamma_pwr;
 
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_nom*Iq + Vd + Vd2 + Vd3 - V_grid_d), # Id
              (1/Lf)*(-Rf*Iq -Lf*omega_nom*Id + Vq + Vq2 + Vq3 - V_grid_q ),          # Iq
              kii*(Idref - Id),                                # gamma_d
              kii*(0 - Iq),                                    # gamma_q
              ki_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq)),                # gamma_pwr
              - Kdv*np.arctan2(Q2,P2),
              mu*V2*(V_nom**2-V2**2) - (2*eta/(3*V2))*(P2-Pref2),
              - Kdv*np.arctan2(Q3,P3),
              mu*V3*(V_nom**2-V3**2) - (2*eta/(3*V3))*(P3-Pref3),]            
    return dx_dt


# start system simulation to get the equilibrium points


BW_cc = 2*np.pi*np.array([10,100,1000]);

for i in range(0,len(BW_cc)):

    kpi = Lf*BW_cc[i];
    kii = Rf*BW_cc[i];
    kp_pwr = (BW_cc[i]/10)/(1.5*V_nom);
    ki_pwr = (BW_cc[i]/100)*kp_pwr;
    
    tstart = 0
    tstop = 1
    increment = 0.0001
    tstart_plot = 0#int((10/100)*tstop/increment)
    t = np.arange(tstart,tstop,increment)
    res = odeint(diff, x_init, t, args=(V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3))
    
    # with the equilibrium oints, analyze teh system stability
    x_init = res[-1,:]
    param = [V_nom, V_grid , omega_nom, Lf, Rf, Pref1, kp_pwr, ki_pwr, kpi, kii, phi_ref, Kdv, mu , eta, Pref2, Pref3]
    equilibrium, infodict,ie,msg = fsolve(f, x_init, param, full_output=1 )
            
    [eig_val,eig_vect] = eig(infodict["fjac"]);
    Participation_Factor_Eval(infodict["fjac"],0.001);
    eig_real = [ele.real for ele in eig_val];
    eig_imag = [ele.imag for ele in eig_val];
    
    nfig = nfig +1;
    plt.figure(nfig)
    plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
    plt.grid()
    
    print("Success ?",ie)
    
    ### steady state solutions
    Id_eq = res[-1,0]
    Iq_eq = res[-1,1]
    gamma_d_eq = res[-1,2]
    gamma_q_eq = res[-1,3]
    gamma_pwr_eq = res[-1,4]
    th2_eq = res[-1,5]
    V2_eq = res[-1,6]
    th3_eq = res[-1,7]
    V3_eq = res[-1,8]
    
    Vq_eq = kpi*(0-Iq_eq) + gamma_q_eq + Lf*omega_nom*Id_eq;
    Vd_eq = (1/(1+kpi*kp_pwr*1.5*Id_eq))*(kp_pwr*kpi*(Pref1-1.5*Vq_eq*Iq_eq) + kpi*(gamma_pwr_eq-Id_eq) + gamma_d_eq - Lf*omega_nom*Iq_eq);
    
    th1_eq = np.arctan2(Vq_eq,Vd_eq); # this is the angle the inverter 1 makes with respect to the grid rotating frame
    
    Id2_eq = Id_eq*np.cos(th2_eq-th1_eq)+Iq_eq*np.sin(th2_eq-th1_eq);
    Iq2_eq = Iq_eq*np.cos(th2_eq-th1_eq)-Id_eq*np.sin(th2_eq-th1_eq);
    
    Id3_eq = Id_eq*np.cos(th3_eq-th1_eq)+Iq_eq*np.sin(th3_eq-th1_eq);
    Iq3_eq = Iq_eq*np.cos(th3_eq-th1_eq)-Id_eq*np.sin(th3_eq-th1_eq);
    
    P1_eq = 1.5*(Vd_eq*Id_eq + Vq_eq*Iq_eq);
    Q1_eq = 1.5*(Vq_eq*Id_eq - Vd_eq*Iq_eq);
    
    P2_eq = 1.5*V2_eq*Id2_eq;#1.5*(Vd2*Id2+Vq2*Iq2);
    Q2_eq = -1.5*V2_eq*Iq2_eq;#1.5*(Vq2*Id2-Vd2*Iq2);
    
    P3_eq = 1.5*V3_eq*Id3_eq;#1.5*(Vd3*Id3+Vq3*Iq3);
    Q3_eq = -1.5*V3_eq*Iq3_eq;#1.5*(Vq3*Id3-Vd3*Iq3);
    
    print("Active power references for conv 1,2,3 :", Pref1, Pref2, Pref3, "Watt" )
     
    print("Active power of GFM 3:",round(P1_eq,2),"Watt,      ","Reactive power of GFM 3:",round(Q1_eq,2),"Var")    
    print("Active power of GFM 2:",round(P2_eq,2),"Watt,      ","Reactive power of GFM 2:",round(Q2_eq,2),"Var")    
    print("Active power of GFM 3:",round(P3_eq,2),"Watt,      ","Reactive power of GFM 3:",round(Q3_eq,2),"Var")    
    
    del res
    
