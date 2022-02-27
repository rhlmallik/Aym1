# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:39:13 2021


Summmary : This will have only one CCI with grid and PLL dynamics And LCL filter

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
from numpy.linalg import eig

#Number of module in series
Ns =1;

# Inverter paramertres
S_rated = 1000;
V_nom =  30*np.sqrt(2);
V_grid = V_nom;

I_nom = 2*S_rated/(3*V_grid);
omega_nom = 2*60*np.pi;
Rf = 0.4;
Lf = 1.2e-3;
Cf = 20e-6;
Rg = 0.8;
Lg = 2.4e-3;

E = V_grid;
Pref1 = 100;
phi_ref = 0;

# curret control
BW_cc = 2*np.pi*500;

# PLL
BW_pll = 2*np.pi*300;
PM = 60;# in degrees
kp_pll =  BW_pll/E;
Ti_pll = np.tan(PM*180/np.pi)/BW_pll;
ki_pll = kp_pll/Ti_pll;

kpi = Lg*BW_cc;
kii = Rg*BW_cc;

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
    
    V_nom, E , omega_nom, Lf, Rf, Cf, Rg, Lg, Pref1, kp_pwr, ki_pwr, kpi, kii, kp_pll, ki_pll, phi_ref = param;

    Iod, Ioq, Vcd, Vcq, Id, Iq, gamma_vq, th_pll, gamma_d, gamma_q, gamma_pwr = x;

    # algebraic substitutions
    
    th1 = th_pll-phi_ref; 
    
    Vq = kpi*(0-Iq)+kii*gamma_q;
    Vd = (1/((1/kpi)+kp_pwr*1.5*Id))*(kp_pwr*(Pref1-1.5*Vq*Iq)+ki_pwr*gamma_pwr+(kii*gamma_d)/kpi);

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))+ki_pwr*gamma_pwr;
    
    Vgd = E*np.cos(-th1);
    Vgq = E*np.sin(-th1);
    
    dx_dt = [ (1/Lf)*(-Rf*Iod + Lf*omega_nom*Ioq + Vcd - Vgd),
              (1/Lf)*(-Rf*Ioq - Lf*omega_nom*Iod + Vcq - Vgq),
              (1/Cf)*(Cf*omega_nom*Vcq + Id - Iod),
              (1/Cf)*(-Cf*omega_nom*Vcd + Iq -Ioq),
              (1/Lg)*(-Rg*Id + Lg*omega_nom*Iq + Vd - Vcd),
              (1/Lg)*(-Rg*Iq -Lg*omega_nom*Id + Vq - Vcq),
              Vgq,
              kp_pll*Vgq + ki_pll*gamma_vq + omega_nom,             
              Idref - Id,
              0 - Iq,
              Pref1 - 1.5*(Vd*Id+Vq*Iq)]
    return dx_dt

x_init = [Pref1/(np.sqrt(2)*1.5*V_nom),0,V_nom,0,Pref1/(np.sqrt(2)*1.5*V_nom),0,0,0,0,0,0];

param = [V_nom, E , omega_nom, Lf, Rf, Cf, Rg, Lg, Pref1, kp_pwr, ki_pwr, kpi, kii, kp_pll, ki_pll, phi_ref]
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

def diff(x, t, V_nom, E , omega_nom,Lf, Rf, Cf, Rg, Lg, Pref1, kp_pwr, ki_pwr, kpi, kii, kp_pll, ki_pll, phi_ref):
 

    V_nom, E , omega_nom, Lf, Rf, Cf, Rg, Lg, Pref1, kp_pwr, ki_pwr, kpi, kii, kp_pll, ki_pll, phi_ref = param;

    Iod, Ioq, Vcd, Vcq, Id, Iq, gamma_vq, th_pll, gamma_d, gamma_q, gamma_pwr = x;

    # algebraic substitutions
    
    th1 = th_pll-phi_ref; 
    
    Vq = kpi*(0-Iq)+kii*gamma_q;
    Vd = (1/((1/kpi)+kp_pwr*1.5*Id))*(kp_pwr*(Pref1-1.5*Vq*Iq)+ki_pwr*gamma_pwr+(kii*gamma_d)/kpi);

    Idref = kp_pwr*(Pref1 - 1.5*(Vd*Id+Vq*Iq))+ki_pwr*gamma_pwr;
    
    Vgd = E*np.cos(-th1);
    Vgq = E*np.sin(-th1);
    
    dx_dt = [ (1/Lf)*(-Rf*Iod + Lf*omega_nom*Ioq + Vcd - Vgd),
              (1/Lf)*(-Rf*Ioq - Lf*omega_nom*Iod + Vcq - Vgq),
              (1/Cf)*(Cf*omega_nom*Vcq + Id - Iod),
              (1/Cf)*(-Cf*omega_nom*Vcd + Iq -Ioq),
              (1/Lg)*(-Rg*Id + Lg*omega_nom*Iq + Vd - Vcd),
              (1/Lg)*(-Rg*Iq -Lg*omega_nom*Id + Vq - Vcq),
              Vgq,
              kp_pll*Vgq + ki_pll*gamma_vq + omega_nom,             
              Idref - Id,
              0 - Iq,
              Pref1 - 1.5*(Vd*Id+Vq*Iq)]
    return dx_dt

tstart = 0
tstop = 100
increment = 0.0001
x0 = [0,0]
tstart_plot = np.int((10/100)*tstop/increment)
t = np.arange(tstart,tstop+1,increment)
res = odeint(diff, x_init, t, args=(V_nom, E , omega_nom,Lf, Rf, Cf, Rg, Lg, Pref1, kp_pwr, ki_pwr, kpi, kii, kp_pll, ki_pll, phi_ref))
Iod_disp = res[tstart_plot:-1,0]
Ioq_disp = res[tstart_plot:-1,1]
Vcd_disp = res[tstart_plot:-1,2]
Vcq_disp = res[tstart_plot:-1,3]
Id_disp = res[tstart_plot:-1,4]
Iq_disp = res[tstart_plot:-1,5]
gamma_vq_disp = res[tstart_plot:-1,6]
th_pll_disp = res[tstart_plot:-1,7]
gamma_pwr_disp = res[tstart_plot:-1,10]
gamma_d1_disp = res[tstart_plot:-1,8]
gamma_q1_disp = res[tstart_plot:-1,9]

t = t[tstart_plot:-1];
# Plot the Results
plt.figure(3)
plt.plot(t,Iod_disp)
plt.plot(t,Ioq_disp)
plt.plot(t,Id_disp)
plt.plot(t,Iq_disp)
plt.title('Currents')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["Iod", "Ioq","Id", "Iq"])
plt.show()
plt.figure(4)
plt.plot(t,Vcd_disp)
plt.plot(t,Vcq_disp)
plt.title('Voltages')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["Vcd", "Vcq"])
plt.show()
plt.figure(5)
plt.plot(t,gamma_pwr_disp)
plt.plot(t,gamma_vq_disp)
plt.plot(t,gamma_d1_disp)
plt.plot(t,gamma_q1_disp)
plt.title('Integration variables')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
plt.legend(["gamma_pwr", "gamma_Vq","gamma_d","gamma_Q"])
plt.show()
plt.figure(6)
plt.plot(t,th_pll_disp)
plt.title('Angles')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["th_pll"])
plt.show()