# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:40:56 2021
Check stability of series connected converters
@author: rhlma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig

#LCL filter
R_f = 10;
L_f = 1.5e-3;


Grid_volt = 50.0;
omega_nom= 60*2*np.pi;
V_nom= Grid_volt*np.sqrt(2.0);

Zf = np.sqrt( ((L_f)*omega_nom)**2 + (R_f)**2);


def f(x,param):
    
    V_nom , omega_nom, R_f, L_f, Z_f= param;
    th1 = x[1];
    th2 = x[2];
    th3 = x[3];
    
    dxdt = [(1/L_f)*(-R_f*x[0] + L_f*omega_nom*x[1] + x[3] - E*np.cos(x[2])),
            (1/L_f)*(-R_f*x[1] - L_f*omega_nom*x[0] + E*np.sin(x[2])),
             -(2*eta/(3*x[3]**2))*(1.5*x[3]*x[0] - P_ref),
           mu*x[3]*(V_nom**2 - x[3]**2)  - (2*eta/(3*x[3]))*(-1.5*x[3]*x[1] - Q_ref)]

    return dxdt

#%% function ode used to solve system of equations

x_init = [P_ref/(np.sqrt(2)*1.5*V_nom),Q_ref/(np.sqrt(2)*1.5*V_nom), 0,V_nom];
Req = R_f+R_g;
Leq= L_f+L_g;
param = [V_nom , omega_nom, eta, mu, Req, Leq, P_ref, Q_ref, E]
eq= fsolve(f, x_init, param)
Jac = np.array([[-Req/Leq , omega_nom, E*np.sin(eq[2])/Leq, 1/Leq],
               [-omega_nom, -Req/Leq, E*np.cos(eq[2])/Leq, 0 ],
               [-eta/eq[3], 0       ,0                   ,-eta*eq[0]/eq[3]**2 + (4*eta*(-P_ref+1.5*eq[3]*eq[0])/(3*eq[3]**3))],
               [0          , eta     , 0                , mu*(V_nom**2-eq[3]**2)-2*mu*eq[3]**2 +eta*eq[1]/eq[3] + (2*eta*(-Q_ref-1.5*eq[3]*eq[1])/(3*eq[3]**2))]]);
[eig_val,eig_vect] = eig(Jac)
# extract real part
x = [ele.real for ele in eig_val]
# extract imaginary part
y = [ele.imag for ele in eig_val]
  

# plot the complex numbers

plt.scatter(x, y,s=1000, c='k', marker='x',label="without inertia")
plt.ylabel('Imaginary')
plt.xlabel('Real')
