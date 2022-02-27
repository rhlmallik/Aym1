"""
Created on Sat Oct 23 23:04:32 2021

ALL in the first inverter's frame of reference'

We will have the following three inverters

2. Voltage Controlled Inverters (INV2,3)
- theta


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
Rf = 22.4;
Lf = 1.2e-3;
Cf = 20e-6;
Rg = 0.4;
Lg = 1.2e-3;
E = V_grid;
Pref1 = 100;
Pref2 = 100;
Pref3 = 100;

Kdv = 0.45;


# x = [0=id, 1=iq, 2=delta, 3=V, 4=omega]

def f(x,param):
    
    V_nom , omega_nom, Lf, Rf, Pref1, Pref2, Pref3, Kdv= param;

    Id, Iq, th1, th2, th3 = x;

    V1d = V_nom;
    V1q = 0;
    
    V2d = V_nom*np.cos(th2-th1);
    V2q = V_nom*np.sin(th2-th1);
    
    V3d = V_nom*np.cos(th3-th1);
    V3q = V_nom*np.sin(th3-th1);  
    
    Id = Id*np.cos(-th1);
    Iq = Id*np.sin(-th1);

    omega_g = omega_nom;# can be anything, th2 / th3
    
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_g*Iq +V1d + V2d + V3d ),
              (1/Lf)*(-Rf*Iq - Lf*omega_g*Id +V1q + V2q + V3q ),
              omega_nom - Kdv*np.arctan2(1.5*(V1q*Id-V1d*Iq),1.5*(V1d*Id+V1q*Iq)),
              omega_nom - Kdv*np.arctan2(1.5*(V2q*Id-V2d*Iq),1.5*(V2d*Id+V2q*Iq)),
              omega_nom - Kdv*np.arctan2(1.5*(V3q*Id-V3d*Iq),1.5*(V3d*Id+V3q*Iq))]
  
    return dx_dt

x_init = [Pref1/(1.5*V_nom),0,0.5,0.1,0.3];

# Iod, Ioq, Vcd, Vcq, Id, Iq, gamma_vq, th_pll, gamma_pwr, gamma_d1, gamma_q1, V2, th2, V3, th3 = x

param = [V_nom , omega_nom, Lf, Rf, Pref1, Pref2, Pref3, Kdv]

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

def diff(x, t, V_nom , omega_nom, Lf, Rf, Pref1, Pref2, Pref3, Kdv):
    
    V_nom , omega_nom, Lf, Rf, Pref1, Pref2, Pref3, Kdv= param;

    Id, Iq, th1, th2, th3 = x;

    V1d = V_nom;
    V1q = 0;
    
    V2d = V_nom*np.cos(th2-th1);
    V2q = V_nom*np.sin(th2-th1);
    
    V3d = V_nom*np.cos(th3-th1);
    V3q = V_nom*np.sin(th3-th1);  
    
    Id = Id*np.cos(-th1);
    Iq = Id*np.sin(-th1);

    omega_g = omega_nom;# can be anything, th2 / th3
    
    dx_dt = [ (1/Lf)*(-Rf*Id + Lf*omega_g*Iq +V1d + V2d + V3d ),
              (1/Lf)*(-Rf*Iq - Lf*omega_g*Id +V1q + V2q + V3q ),
              omega_nom - Kdv*np.arctan2(1.5*(V1q*Id-V1d*Iq),1.5*(V1d*Id+V1q*Iq)),
              omega_nom - Kdv*np.arctan2(1.5*(V2q*Id-V2d*Iq),1.5*(V2d*Id+V2q*Iq)),
              omega_nom - Kdv*np.arctan2(1.5*(V3q*Id-V3d*Iq),1.5*(V3d*Id+V3q*Iq))]
    return dx_dt
tstart = 0
tstop = 1
increment = 0.0001
x0 = [0,0]
tstart_plot = np.int((10/100)*tstop/increment)
t = np.arange(tstart,tstop+1,increment)
res = odeint(diff, x_init, t, args=(V_nom , omega_nom, Lf, Rf, Pref1, Pref2, Pref3, Kdv))
Id_disp = res[tstart_plot:-1,0]
Iq_disp = res[tstart_plot:-1,1]
th1_disp = res[tstart_plot:-1,2]
th2_disp = res[tstart_plot:-1,3]
th3_disp = res[tstart_plot:-1,4]

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
plt.legend(["Id", "Iq"])
plt.show()

plt.figure(6)
plt.plot(t,th1_disp)
plt.plot(t,th2_disp)
plt.plot(t,th3_disp)
plt.title('Angles')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid()
#plt.axis([-1, 1, -1.5, 1.5])
plt.legend(["th1", "th2","th3"])
plt.show()

# %%
