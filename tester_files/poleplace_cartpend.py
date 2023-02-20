import numpy as np
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple
import importlib
import control
import controlpy as cpy

#importing from linearQuadraticRegulator
mod_lqr = importlib.import_module('linearQuadraticRegulator')


# importing from invertedPendulum.py
invP = importlib.import_module('invertedPendulum')
# /\ /\ yay it worked!
# uh I could also try from invertedpendulum import X to get X specifically

# jacobian of a function f is Df/Dx, if we're linearizing about points we evaluate the jacobian at those points.

# taking the inverted pendulum and linearizing it about two fixed points
# and calculating its jacobian, we get the following A and B matrices corresponding to xdot = Ax + Bu
# (fixed points are the up position with theta = angpos = pi, and the down position with theta = angpos = 0.
# up position is state = [free, 0, pi, 0], and down position is state = [free, 0, 0, 0])

# -- pendswitch --
# pendswitch is either 0 corresponding to pendulum down, or 1 corresponding to pendulum up
pendswitch = 1
# -- pendswitch --

# control input u represents the force in the x direction

# After linearization, we get our dynamics matrix and our input matrix
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

# calculating eigenvalues and eigenvectors of A, to determine if the system is unstable (if it needs to be controlled or not)
# eigval = eigenvalue, eigvec = eigenvector
eigval, eigvec = np.linalg.eig(A)

# Controlability matrix C= [B AB A^2*B ... A^n * B], where n is the number of elements in your state vector (n = number of elements in x)
C = control.ctrb(A, B)

# If the rank of our controlability matrix C is the same as n, that means our controller can span (reach) the entire possible state space
# ie the possible places that our state variable can go to (possible sets of cart position, cart velocity, pendulum angle, angular velocity).
# We want to establish some control input u that is equal to -Kx (ie -K * our state). This lets us rewrite our linearized system (xdot = Ax+Bu) as
#  -- the matrix (A-BK) represents a modification of the uncontrolled system by some control input u.
# rate of change of the system's state is this modified matrix times the state's current value.
# The importance of this is that xdot = (A-BK)x has some solution of the form x(t) = e^(A-BK)t * x(0),
# or that the state at some time t is based on this modified matrix, and the state's initial condition.
# eigdes is our desired eigenvalues
#eigdes = np.array([
    
#    -1.1, 
#    -1.2, 
#    -1.3, 
#    -1.4
    
#    ])
eigdes = np.array([-0.3, -0.4, -0.5, -0.6])


#K = control.place(A, B, eigdes)
# With established K matrix, we now want to find the eigenvalues of A-BK:
# controlled input matrix BK
#controlledInput = np.dot(B, K)
# Modified matrix A - BK
#modA = np.subtract(A, controlledInput)
#mod_eigval, modeigvec = np.linalg.eig(modA)

# Crafting K using the linear quadratic regulator
# ie crafting K by using a cost function:
# J = integral (from 0 to infinity) of x^T*Q*x + u^T*R*u dt
# Q = cost matrix on state
Q = np.array([

    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]

])

# R = cost matrix on control input -- how expensive is the controller?
R = np.array([ [0.01] ])

#K = control.lqr(A, B, Q, R)
K, riacatti, lqreigvals = mod_lqr.lqr(A, B, Q, R)

print(K)


#bk = np.dot(B,K)
#aminusbk = np.subtract(A,bk)
#print(aminusbk)
#print(bk)

#print(C)
#print(np.linalg.matrix_rank(C))
#print(K)
#print(mod_eigval)

# pole placement


# -- plotting --

# time variable
#t0, t1 = 0.0, 20.0
#t = np.linspace(t0, t1, 100)
# initial conditions
#x0 = [0.0, 0.0, np.pi, 0.5]
#x = np.zeros((len(t), len(x0)))
#x[0, :] = x0

# -- parameters --
# Setting control input u to be -Kx, ie u = -K * state. We're going to use refstate, which will just be the state minus some state that we want to finish at?
# yes, ref is that state we want to finish at, since setting up the system with A - BK means that the controller will try to bring the overall state to 0.
# I think

# next step is to make K into a matrix with 100 rows and 4 columns to match matrix sizes
# after that, change params and cartpend() in invertedPendulum.py to account for the fact that u is an array. u's size is (1,4), ie 1 row and 4 columns


#ref = np.array([ 1, 0, np.pi, 0 ])
#refstate = np.subtract(x, ref)
#print(ref)

#u = np.dot(-K, refstate)
#params = (invP.pmass, invP.cmass, invP.linklength, invP.gravity, invP.damp, invP.u)
# -- parameters --


# integrator
#x = odeint(invP.cartpend, x0, t, args=params)


#plt.plot(t, x[:,0], label = 'cart position')
#plt.plot(t, x[:,1], label = 'cart velocity')
#plt.plot(t, x[:,2], label = 'angle position')
#plt.plot(t, x[:,3], label = 'angular velocity')
#plt.xlabel('time')
#plt.ylabel('state')
#plt.legend()
#plt.show()