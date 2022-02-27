# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:45:52 2021

@author: rhlma
We will have the following three inverters
1. Current COntrolled Inverters (INV1)
- gamma_d
- gamma_q
- gamma_pwr
- gamma_Vq
- theta_pll

2. Voltage Controlled Inverters (INV2,3)
- theta
- voltage (maybe?)

3. Line dynamics
- Id
- Iq
- Iod
- Ioq
- Vcd
- Vcq

Controller design :
    Current controller: Kp, Ki based on plant inversion
    The droop constants for the voltage is same for VCI 1,2
    The droop constants for the current controller is different
@author: rhlma
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
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
Lf = 1.2e-3;
Cf = 20e-6;
Rg = 0.4;
Lg = 1.2e-3;
E = V_grid;
Pref1 = 100;
Pref2 = 100;
Pref3 = 100;
phi_ref = 0;


# curret control
BW_cc = 2*np.pi*1000;

# PLL
BW_pll = 2*np.pi*1;
PM = 60;# inn degrees
kp_pll =  BW_pll/E;
Ti_pll = np.tan(PM*180/np.pi)/BW_pll;
ki_pll = kp_pll/Ti_pll;

kpi = Lf*BW_cc;
kii = Rf*BW_cc;

# power control for CCI 
kp_pwr = (BW_cc/10)/(1.5*V_nom);
ki_pwr = (BW_cc/100)*kp_pwr;

# droop control for theta
Kdv = BW_cc/1000;


# dVOC parameters
eta = 1e3;
mu  = 1e-4;

# x = [0=id, 1=iq, 2=delta, 3=V, 4=omega]

def f(x,param):
    
    V_nom , omega_nom, eta, mu, Lf, Rf, Cf, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, kp_pwr, ki_pwr, kpi, kii, Kdv, kp_pll, ki_pll= param;

    Iod, Ioq, Vcd, Vcq, Id, Iq, gamma_vq, th_pll, gamma_pwr, gamma_d1, gamma_q1, V2, th2, V3, th3 = x;

    # algebraic substitutions
    th1 = th_pll-phi_ref; # we assume th_pll = 0

    V1q = kpi*(0-Iq)+kii*gamma_q1;
    V1d = (1/((1/kpi)+kp_pwr*1.5*Id))*(kp_pwr*(Pref1-1.5*V1q*Iq)+ki_pwr*gamma_pwr+(kii*gamma_d1)/kpi);


    V2d = V2*np.cos(th2-th1);
    V2q = V2*np.sin(th2-th1);
    
    V3d = V3*np.cos(th3-th1);
    V3q = V3*np.sin(th3-th1);  
    
    omega_g = omega_nom + kp_pll*E*np.sin(th_pll) + ki_pll*gamma_vq;
   
    dx_dt = [ (1/Lg)*(-Rg*Iod + Lg*omega_g*Ioq + Vcd - E*np.cos(th_pll)),
              (1/Lg)*(-Rg*Ioq - Lg*omega_g*Iod + Vcq - E*np.sin(th_pll)),
              (1/Cf)*(Cf*omega_g*Vcq + Id - Iod),
              (1/Cf)*(-Cf*omega_g*Vcd + Iq -Ioq),
              (1/Lf)*(-Rf*Id + Lf*omega_g*Iq +V1d + V2d + V3d - Vcd),
              (1/Lf)*(-Rf*Iq - Lf*omega_g*Id +V1q + V2q + V3q - Vcq),
              E*np.sin(th_pll),
              kp_pll*E*np.sin(th_pll) + ki_pll*gamma_vq + omega_nom,
              Pref1 - 1.5*(V1d*Id+V1q*Iq),
              kp_pwr*(Pref1 - 1.5*(V1d*Id+V1q*Iq))+ki_pwr*gamma_pwr - Id,
              0 - Iq,
              mu*V2*(V_nom**2-V2**2) - (2*eta/(3*V2))*(1.5*(V2d*Id+V2q*Iq)-Pref2),
              omega_nom - Kdv*np.arctan2(1.5*(V2q*Id-V2d*Iq),1.5*(V2d*Id+V2q*Iq)),
              mu*V3*(V_nom**2-V3**2) - (2*eta/(3*V3))*(1.5*(V3d*Id+V3q*Iq)-Pref3),
              omega_nom - Kdv*np.arctan2(1.5*(V3q*Id-V3d*Iq),1.5*(V3d*Id+V3q*Iq))]
  
    return dx_dt

x_init = [Pref1/(1.5*V_nom),0,E,0,Pref1/(1.5*V_nom),0,0,0,0,0,0,V_nom,0,V_nom,0];

# Iod, Ioq, Vcd, Vcq, Id, Iq, gamma_vq, th_pll, gamma_pwr, gamma_d1, gamma_q1, V2, th2, V3, th3 = x

param = [V_nom , omega_nom, eta, mu, Lf, Rf, Cf, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, kp_pwr, ki_pwr, kpi, kii, Kdv, kp_pll, ki_pll]

equilibrium, infodict,ie,msg = fsolve(f, x_init, param, full_output=1)
        
[eig_val,eig_vect] = eig(infodict["fjac"]);
Participation_Factor_Eval(infodict["fjac"],0.001);
eig_real = [ele.real for ele in eig_val];
eig_imag = [ele.imag for ele in eig_val];

plt.figure(2)
plt.plot(eig_real, eig_imag,'s',markersize=3, markeredgewidth=1)
plt.grid()

print("Success ?",ie)


# SImulate the system
# Comment out this part if you do not want to simulate

def diff(x, t, V_nom , omega_nom, eta, mu, Lf, Rf, Cf, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, kp_pwr, ki_pwr, kpi, kii, Kdv, kp_pll, ki_pll):
    
    Iod, Ioq, Vcd, Vcq, Id, Iq, gamma_vq, th_pll, gamma_pwr, gamma_d1, gamma_q1, V2, th2, V3, th3 = x;

    # algebraic substitutions
    th1 = th_pll-phi_ref; # we assume th_pll = 0

    V1q = kpi*(0-Iq)+kii*gamma_q1;
    V1d = (1/((1/kpi)+kp_pwr*1.5*Id))*(kp_pwr*(Pref1-1.5*V1q*Iq)+ki_pwr*gamma_pwr+(kii*gamma_d1)/kpi);


    V2d = V2*np.cos(th2-th1);
    V2q = V2*np.sin(th2-th1);
    
    V3d = V3*np.cos(th3-th1);
    V3q = V3*np.sin(th3-th1);  
    
    omega_g = omega_nom + kp_pll*E*np.cos(th_pll) + ki_pll*gamma_vq;
   
    dx_dt = [ (1/Lg)*(-Rg*Iod + Lg*omega_g*Ioq + Vcd - E*np.cos(th_pll)),
              (1/Lg)*(-Rg*Ioq - Lg*omega_g*Iod + Vcq - E*np.sin(th_pll)),
              (1/Cf)*(Cf*omega_g*Vcq + Id - Iod),
              (1/Cf)*(-Cf*omega_g*Vcd + Iq -Ioq),
              (1/Lf)*(-Rf*Id + Lf*omega_g*Iq +V1d + V2d + V3d - Vcd),
              (1/Lf)*(-Rf*Iq - Lf*omega_g*Id +V1q + V2q + V3q - Vcq),
              E*np.sin(th_pll),
              kp_pll*E*np.sin(th_pll) + ki_pll*gamma_vq + omega_nom,
              Pref1 - 1.5*(V1d*Id+V1q*Iq),
              kp_pwr*(Pref1 - 1.5*(V1d*Id+V1q*Iq))+ki_pwr*gamma_pwr - Id,
              0 - Iq,
              mu*V2*(V_nom**2-V2**2) - (2*eta/(3*V2))*(1.5*(V2d*Id+V2q*Iq)-Pref2),
              omega_nom - Kdv*np.arctan2(1.5*(V2q*Id-V2d*Iq),1.5*(V2d*Id+V2q*Iq)),
              mu*V3*(V_nom**2-V3**2) - (2*eta/(3*V3))*(1.5*(V3d*Id+V3q*Iq)-Pref3),
              omega_nom - Kdv*np.arctan2(1.5*(V3q*Id-V3d*Iq),1.5*(V3d*Id+V3q*Iq))]
  
    return dx_dt
tstart = 0
tstop = 1
increment = 0.00001
x0 = [0,0]
tstart_plot = np.int((99/100)*tstop/increment)
t = np.arange(tstart,tstop+1,increment)
res = odeint(diff, x_init, t, args=(V_nom , omega_nom, eta, mu, Lf, Rf, Cf, Rg, Lg, Pref1, Pref2, Pref3, E, phi_ref, kp_pwr, ki_pwr, kpi, kii, Kdv, kp_pll, ki_pll))
Iod_disp = res[tstart_plot:-1,0]
Ioq_disp = res[tstart_plot:-1,1]
Vcd_disp = res[tstart_plot:-1,2]
Vcq_disp = res[tstart_plot:-1,3]
Id_disp = res[tstart_plot:-1,4]
Iq_disp = res[tstart_plot:-1,5]
gamma_vq_disp = res[tstart_plot:-1,6]
th_pll_disp = res[tstart_plot:-1,7]
gamma_pwr_disp = res[tstart_plot:-1,8]
gamma_d1_disp = res[tstart_plot:-1,9]
gamma_q1_disp = res[tstart_plot:-1,10]
V2_disp = res[tstart_plot:-1,11]
th2_disp = res[tstart_plot:-1,12]
V3_disp = res[tstart_plot:-1,13]
th3_disp = res[tstart_plot:-1,14]

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
plt.plot(t,V2_disp)
plt.plot(t,V3_disp)
plt.title('Voltages')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["Vcd", "Vcq","V2","V3"])
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
plt.plot(t,th2_disp)
plt.plot(t,th3_disp)
plt.title('Angles')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["th_pll", "th2","th3"])
plt.show()
