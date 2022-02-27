# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:21:26 2021

@author: rhlma
"""
# ###########################################################################
# # Code to check the stability of the original VOC with the mppt dc link
# # controller and maximum power point trackig control. We use a PI 
# # controller for dc link control and I control for MPPT and verify all 
# # observations in the dq domain. The exact PV module is taken. The VOC is
# # supplanted with the ICOV architecture for the inverters
# # Created: 02/03/2021 (mm/dd/yyyy)
# # Created By: Rahul Mallik 
# # Warning: For Washington Power Electronics Lab use only
# # Reference : 
# ###########################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eig
import sympy as sym

nfig = 0;
## Parameter list


#AHO controller
S_rated = 10000;
S_base = S_rated;
V_nom = 120*np.sqrt(2);
V_base = V_nom/np.sqrt(2);
Z_base = 3*V_base**2/S_base;
omega_nom = 120*np.pi;
delta_omega = 0.5*2*np.pi; #0.5Hz is the max frequency deviation
delta_V     = 0.1*V_nom;
eta = delta_omega*3*(0.9*V_nom)**2/(2*S_rated);
mu  = delta_omega/(delta_V*(V_nom+0.9*V_nom));
H = 2; # 2 to 10
Dpu = 50;
J = 2*H*S_rated/omega_nom**2;
D = Dpu*S_rated/omega_nom;
#LCL filter
R_f = 0.04;
L_f = 1*Z_base/omega_nom;

#power setpoints
P_ref = 500;
Q_ref = 0;

#grid
E = V_nom;

def Jacobian(v_str, f_list):
    vars = sym.symbols(v_str)
    f = sym.sympify(f_list)
    J = sym.zeros(len(f),len(vars))

    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i,j] = sym.diff(fi, s)
    print(J)
    return J

## state variables 
io_d = sym.Symbol('io_d')
io_q = sym.Symbol('io_q')
V = sym.Symbol('V')
delta = sym.Symbol('delta')


## Symbolic substitutions 
#grid voltage
e_d = np.sqrt(2)*E*np.cos(delta);
e_q = -np.sqrt(2)*E*np.sin(delta)
v_d = np.sqrt(2)*V;
v_q = 0;
P       = 3/2*(v_d*io_d + v_q*io_q);
Q       = 3/2*(v_q*io_d - v_d*io_q);

## Differential equations
#  xdot = f(x)
# line current dynamics
# grid side current differentials
f_1  = (1/L_f)*(-R_f*io_d + L_f*omega_nom*io_q + v_d - e_d); 
f_2  = (1/L_f)*(-L_f*omega_nom*io_d - R_f*io_q + v_q - e_q);  

# voc dynamics
f_3 = -(2*eta/(3*V**2))*(P - P_ref); #delta
f_4 = mu*V*(V_nom**2 - V**2)  - (2*eta/(3*V))*(Q - Q_ref); #V

# Steady State Computation
x = ['io_d','io_q','V','delta'];
f_8ord = ['f_1','f_2','f_3','f_4'];

# Jacobian
Jac = Jacobian(x, f_8ord);

# Stopping Criterion
epsilon = 0.0001;

# # Initial Guess
# y_8ord = [P_ref/(np.sqrt(2)*1.5*V_nom),Q_ref/(np.sqrt(2)*1.5*V_nom), 0,V_nom, omega_nom];

# f_8ord_val = J.subs({x[0]:y_8ord(0), x[1]: y_8ord(1),x[2]:y_8ord(2),x[3]:y_8ord(3),x[4]:y_8ord(4)})

# norm_f_8ord = norm(f_8ord_val,2);

# j=1;
# # Iteration
# while(norm_f_8ord > epsilon):
#     # Evaluate the jacobian at the y_k guess
#     Jac_off_val = subs(Jac_off,{io_d, io_q, delta, V, omega_i},...
#         [y_8ord(1),y_8ord(2),y_8ord(3),y_8ord(4),y_8ord(5)]); 
 
#     # Use Newton's method to calculate y_k+1 from y_k
#     y_8ord = y_8ord - (double(Jac_off_val))\double(f_8ord_val);
    
#     # Obtain the value of f(x) at the equilibrium point y_k+1 
#     f_8ord_val = subs(f_8ord,{io_d, io_q, delta, V, omega_i},...
#          [y_8ord(1),y_8ord(2),y_8ord(3),y_8ord(4),y_8ord(5)]); 
  
#     # FInd the norm of the f(x) at the equilibrium soln
#     norm_f_8ord = norm(f_8ord_val);  
#     j=j+1;
# end
# # Steady State Results 
# Id_disp  = double(vpa(y_8ord(1),3));
# Iq_disp = double(vpa(y_8ord(2),3));
# Delta_disp = double(vpa(y_8ord(3),3));
# V_disp = double(vpa(y_8ord(4),3));
# omega_disp =  double(vpa(y_8ord(5),3));
# fprintf('part (a):')
# fprintf('\n D-axis current = #g A , Q axis current =#g A',Id_disp,Iq_disp)
# fprintf('\n Voltage = #g V , Angle (th-th_g) =#g rad',V_disp,Delta_disp)
# fprintf('\n omega =#g rad/s',omega_disp)


# ## Eigen Value and stability analysis
# Jac_off_val = subs(Jac_off,{io_d, io_q, delta, V, omega_i},...
#     [y_8ord(1),y_8ord(2),y_8ord(3),y_8ord(4), y_8ord(5)]); 
# V = double(eig(Jac_off_val));
# fprintf('\n part (b):')
# fprintf('\n Eigen values = #f#+fi ',[real(V(:)), imag(V(:))]')
# nfig = nfig+1;
# figure(nfig)
# plot(V,'.','marker','x',...
#          'MarkerSize', 120, 'MarkerEdgeColor','k',...
#          'MarkerFaceColor','r','LineWidth',1.5); hold on;grid on;
# ## Participation Factors 
# fprintf('\n part (c):')
# Participation_Factor_Eval(double (simplify((Jac_off_val))),1e-2)

