import numpy as np
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple
import importlib
import control


# importing from poleplace_cartpend
poleCart = importlib.import_module('poleplace_cartpend')

# importing from invertedPendulum.py
invP = importlib.import_module('invertedPendulum')

# -- pendswitch --
# pendswitch is either 0 corresponding to pendulum down, or 1 corresponding to pendulum up
pendswitch = 1
# -- pendswitch --

# control input u represents the force in the x direction


# Dynamics Matrix
A = np.array([

    [0, 1, 0, 0],
    [0, -invP.damp / invP.cmass, -invP.pmass * invP.gravity / invP.cmass, 0],
    [0, 0, 0, 1],
    [0, -pendswitch * invP.damp / (invP.cmass * invP.linklength), -pendswitch * (invP.pmass + invP.cmass) * invP.gravity / (invP.cmass * invP.linklength), 0]

])

# Input Matrix
B = np.array([

    [0],
    [1/invP.cmass],
    [0],
    [pendswitch * 1 / (invP.cmass * invP.linklength)]
    
])

# Assigning K

# Potential K matrix -- calculated by runing poleplace_cartpend
#K = np.array([ [-7.20000e-02, -1.68400e+00, 1.33944e+02, 3.73680e+01] ])
#K = np.array([ [-100.0, -142.12812515, 1100.80738538, 497.74707473] ])

'''
Ok the issue with calling K from poleCart (poleplace_cartpend) is that the K that gets produced is an array with one element?
    Like trying to call K[0] returns everything in the array -- there's no K[1]

This issue could probably be solved by relying more on commands instead of manually just allocating things literally

'''
'''
K = poleCart.K
print(K.shape)
print(type(K))
print(K[1])
'''

K = np.array([ [-10.0, -27.61730343, 413.74042088, 180.7242803] ])

'''
# I can just do aminusbk = A - B*K
bk = np.dot(B,K)
aminusbk = np.subtract(A,bk)
'''
bk = B * K
aminusbk = A - B * K

# state is x = [x, xdot, theta, thetadot]
# x = [x1, x2, x3, x4] --> x1 = x, x2 = xdot, x3 = theta, x4 = thetadot
# x1 = x[0]
# x2 = x[1]
# x3 = x[2]
# x4 = x[3]

# this essentially creates a limited form of a class
# anything that you define as a "State", ie something like xstate = State(inputs)
# will create a named tuple -- you create a variable that has the State "class"
# with specific properties. Here, a variable of State "class" (named tuple) has a parent (State?)
# along with an array with 4 elements named x, xdot, theta, and thetadot
# State doesn't have any methods
State = namedtuple("State", ['x', 'xdot', 'theta', 'thetadot'])

def modcartpend(x, t, pmass, cmass, linklength, gravity, damp, x_des):
    state = State(*x)
    pos = x[0]     # pos = state.x                   position
    vel = x[1]     # vel = state.xdot                velocity
    angpos = x[2]     # angpos = state.theta            angular position
    angvel = x[3]     # angvel = state.thetadot         angular velocity

    # desired control input influences
    pos_des = x_des[0]
    vel_des = x_des[1]
    angpos_des = x_des[2]
    angvel_des = x_des[3]


    # could do this with a for loop, figure out later
    # goal is to also inlcude a desired state, des_state, such that the system xdot becomes:
    # xdot = (A - BK)x + BK*des_state

    # "state matrix" A - BK for closed loop system
    mod_state1 = aminusbk[0][0] * pos + aminusbk[0][1] * vel + aminusbk[0][2] * angpos + aminusbk[0][3] * angvel
    mod_state2 = aminusbk[1][0] * pos + aminusbk[1][1] * vel + aminusbk[1][2] * angpos + aminusbk[1][3] * angvel
    mod_state3 = aminusbk[2][0] * pos + aminusbk[2][1] * vel + aminusbk[2][2] * angpos + aminusbk[2][3] * angvel
    mod_state4 = aminusbk[3][0] * pos + aminusbk[3][1] * vel + aminusbk[3][2] * angpos + aminusbk[3][3] * angvel

    # "control matrix" BK*x_des
    mod_control1 = bk[0][0] * pos_des + bk[0][1] * vel_des + bk[0][2] * angpos_des + bk[0][3] * angvel_des
    mod_control2 = bk[1][0] * pos_des + bk[1][1] * vel_des + bk[1][2] * angpos_des + bk[1][3] * angvel_des
    mod_control3 = bk[2][0] * pos_des + bk[2][1] * vel_des + bk[2][2] * angpos_des + bk[2][3] * angvel_des
    mod_control4 = bk[3][0] * pos_des + bk[3][1] * vel_des + bk[3][2] * angpos_des + bk[3][3] * angvel_des


    xdot1 = mod_state1 + mod_control1
    xdot2 = mod_state2 + mod_control2
    xdot3 = mod_state3 + mod_control3
    xdot4 = mod_state4 + mod_control4


    xdot = np.array([ xdot1, xdot2, xdot3, xdot4 ])

    return xdot


# desired state
#x_des = np.array([ 1, 0, np.pi, 0.5])
x_des = np.array([ 0.0, 0.0, 0.0, 0.0])


# tupleizing the system parameters (grouping paramaters into tuple)
params = (invP.pmass, invP.cmass, invP.linklength, invP.gravity, invP.damp, x_des)



# time variable
t0, t1 = 0.0, 20.0
t = np.linspace(t0, t1, 100)
# initial conditions
x0 = [0.0, 0.0, np.pi, 0.5]
x = np.zeros((len(t), len(x0)))
x[0, :] = x0


# integrating using modified system dynamics xdot = (A - BK)*x instead of xdot = Ax
x = odeint(modcartpend, x0, t, args=params)



plt.plot(t, x[:,0], label = 'cart position')
plt.plot(t, x[:,1], label = 'cart velocity')
plt.plot(t, x[:,2], label = 'angle position')
plt.plot(t, x[:,3], label = 'angular velocity')
plt.xlabel('time')
plt.ylabel('state')
plt.legend()
plt.show()

