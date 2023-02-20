import numpy as np
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from collections import namedtuple

# System is an inverted pendulum on top of a moving cart
# pendulum starts in up position
# cart can move in the x direction, pendulum rotates about center of cart, with theta measuring from top position at 0

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

#state = State(0, 0.1, 3.14, 0.1)
#y6 = state.x

#state.theta = 0

#state[0] = 0

# control input u represents the force in the x direction

# nonlinear ode describing inverted pendulum dynamics
# change constants to be named the actual names for the constants
# cartpend(state, time, pmass, cmass, gravity, damping, controlInput):
def cartpend(x, t, pmass, cmass, linklength, gravity, damp, u):
    state = State(*x)
    pos = x[0]     # pos = state.x                   position
    vel = x[1]     # vel = state.xdot                velocity
    angpos = x[2]     # angpos = state.theta            angular position
    angvel = x[3]     # angvel = state.thetadot         angular velocity


    # control input influences
    upos = u[0]
    uvel = u[1]
    uangpos = u[2]
    uangvel = u[3]


    # truncation of cosine and sine of theta --> theta = y3
    cx3 = np.cos(angpos)
    sx3 = np.sin(angpos)
    # mass truncation?
    D = pmass*linklength**2 * (cmass + pmass*(1 - cx3**2))
    
    xdot1 = vel
    xdot2 = (1.0/D) * (-pmass**2 * linklength**2 * gravity * cx3 * sx3 + pmass * linklength**2 * (pmass * linklength * angvel**2 * sx3 - damp * vel) + pmass * linklength**2 * (1.0/D) * uvel )
    xdot3 = angvel
    xdot4 = (1.0/D) * ( ((pmass + cmass) * pmass * gravity * linklength * sx3) - pmass * linklength * cx3 * (pmass * linklength * angvel**2 * sx3 - damp * vel) ) - pmass * linklength * cx3 * (1.0/D) * uangvel 


    xdot = np.array([ xdot1, xdot2, xdot3, xdot4 ])

    return xdot


# Constants

# parameter constants m, M, L, g, d, (u)
pmass = 1.0
cmass = 10.0
linklength = 2.0
gravity = -10.0
damp = 1.0
u = np.array([ 0, 0, 0, 0 ])

# tupleizing the system parameters (grouping paramaters into tuple)
params = (pmass, cmass, linklength, gravity, damp, u)

# time variable
t0, t1 = 0.0, 20.0
t = np.linspace(t0, t1, 100)
# initial conditions
x0 = [0.0, 0.0, np.pi, 0.5]
x = np.zeros((len(t), len(x0)))
x[0, :] = x0


# Old formating using integrate.ode
#r = integrate.ode(cartpend).set_integrator("dopri5")    
#r.set_initial_value(x0, t0)
#r.set_f_params(params)
#print(r)                        
#for i in range(1, t.size):
#   x[i, :] = r.integrate(t[i])                      
#   if not r.successful():
#       raise RuntimeError("Could not integrate")

# New implementation for integration using odeint

x = odeint(cartpend, x0, t, args=params)



plt.plot(t, x[:,0], label = 'cart position')
plt.plot(t, x[:,1], label = 'cart velocity')
plt.plot(t, x[:,2], label = 'angle position')
plt.plot(t, x[:,3], label = 'angular velocity')
plt.xlabel('time')
plt.ylabel('state')
plt.legend()
plt.show()



def vdp1(t, y):

    #m = 2
    #d = 0
    #k = 2
    #u = 1
    dx = np.array([y[1], (1.0 - y[0]**2)*y[1] - y[0]])
    #dx = np.array([ y[1], (1/m)*(-d*y[1] - k*y[0] + u)  ])


    return dx

t0, t1 = 0, 20                                      # start and end
t = np.linspace(t0, t1, 100)                        # the points of evaluation of solution
y0 = [2, 0]                                         # initial value
y = np.zeros((len(t), len(y0)))                     # array for solution
y[0, :] = y0
r = integrate.ode(vdp1).set_integrator("dopri5")    # choice of method -- r is the integrator ode for vdp1, using the integrator method of "dopr15" I think this is the interpretation of it?
r.set_initial_value(y0, t0)                         # initial values -- line above establishes r as the integrator, and so running the set_initial_value method from r is like doing "r(t0,y0)"
for i in range(1, t.size):
   y[i, :] = r.integrate(t[i])                      # get one more value, add it to the array
   if not r.successful():
       raise RuntimeError("Could not integrate")
plt.plot(t, y[:,0], label = 'y')
plt.plot(t, y[:,1], label = 'dy')
plt.xlabel('time')
plt.ylabel('state')
plt.legend()
plt.show()