import numpy as np
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple
import importlib
import control
import controlpy as cpy


'''
System is a block with mass = blockmass moving along a flat surface. The block is attached to a vertical wall with
a spring such that it experiences a force due to a spring with a spring constant = spring and a damper with damping constant = damp.

The block moves in the positive x direction, with a control input u pushing it in the positive x direction.

The system itself is inherently stable

units
blockmass in kg
spring in force/distance
damp in force/velocity

'''

#importing from linearQuadraticRegulator
mod_lqr = importlib.import_module('linearQuadraticRegulator')


# Let's use a class base system for state
State = namedtuple("State", ['pos', 'vel'])


# model to integrate, should return the derivative of the state
def springmass(state, time, blockmass, spring, damp):

    state = State(*state)

    pos = state.pos
    vel = state.vel

    # stateDiv is the derivative of the state -- it is the grouping of parameters that we are trying to integrate
    stateDiv1 = vel

    stateDiv2 = (1 / blockmass) * (-damp * vel - spring * pos)

    stateDiv = np.array([ stateDiv1, stateDiv2 ])


    return stateDiv

# Constants -- blockmass, spring, damnp
blockmass = 10.0
spring = 1.0
damp = 1.0

# tupleizing the system parameters (grouping paramaters into tuple)
params = (blockmass, spring, damp)

# time variable
t0, t1 = 0.0, 40.0
time = np.linspace(t0, t1, 100) 

# initial conditions
state = State(0.0, 1.0)


# integration
state = odeint(springmass, state, time, args=params)


# Plotting
plt.plot(time, state[:,0], label = 'mass position')
plt.plot(time, state[:,1], label = 'mass velocity')
plt.xlabel('time')
plt.ylabel('state')
plt.legend()
plt.show()



# Creating Ax + Bu frame work -- represents the system in a way that allows us to determine the system's controllability.
# Dyanmics Matrix
dyn_mat = np.array([

    [0.0, 1.0],
    [-spring / blockmass, -damp / blockmass]

])

# Input Matrix -- needs to allow 2 control nobs -- control_pos and control_vel -- which means that we need a 2x2 input array.
# "math reasons fill in later"
input_mat = np.array([

    [0],
    [1]
    
    
])

# calculating eigenvalues and eigenvectors of A, to determine if the system is unstable (if it needs to be controlled or not)
# eigval = eigenvalue, eigvec = eigenvector
eigval, eigvec = np.linalg.eig(dyn_mat)

# Controlability matrix control_mat = [B AB A^2*B ... A^n * B], where n is the number of elements in your state vector (n = number of elements in x),
# B is input_mat, and A is dyn_mat
control_mat = control.ctrb(dyn_mat, input_mat)

eigdes = np.array([0.0, 0.0])


# Crafting K using the linear quadratic regulator
# ie crafting K by using a cost function:
# J = integral (from 0 to infinity) of x^T*Q*x + u^T*R*u dt
# Q = cost matrix on state
Q = np.array([

    [1.0, 0.0],
    [0.0, 100.0]

])

# R = cost matrix on control input -- how expensive is the controller?
R = np.array([ 
    
    [10.00]

])

#K = control.lqr(A, B, Q, R)
K, riacatti, lqreigvals = mod_lqr.lqr(dyn_mat, input_mat, Q, R)



def mod_springmass(mod_state, time, blockmass, spring, damp, des_state):

    mod_state = State(*mod_state)

    des_state = State(*des_state)

    pos = mod_state.pos
    vel = mod_state.vel

    des_pos = des_state.pos
    des_vel = des_state.vel


    controlled_dyn_mat = dyn_mat - input_mat*K

    controlled_input_mat = input_mat * K

    

    
    div_pos = controlled_dyn_mat[0,0] * (pos - des_pos) + controlled_dyn_mat[0,1] * (vel - des_vel)  

    div_vel = controlled_dyn_mat[1,0] * (pos - des_pos) + controlled_dyn_mat[1,1] * (vel - des_vel)
       


    mod_stateDiv = State(div_pos, div_vel)



    return mod_stateDiv



mod_state = State(0.0, 1.0)


des_state = State(0.0, 0.0)



params = (blockmass, spring, damp, des_state)


# integration
mod_state = odeint(mod_springmass, mod_state, time, args=params)

#print(state)

plt.plot(time, mod_state[:,0], label = 'mass position')
plt.plot(time, mod_state[:,1], label = 'mass velocity')
plt.xlabel('time')
plt.ylabel('state')
plt.legend()
plt.show()

eigval, eigvec = np.linalg.eig(dyn_mat - input_mat*K)



