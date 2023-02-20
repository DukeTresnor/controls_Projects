from northwestern_python_functions import *

import numpy as np

import math

from numpy.linalg import eig

from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt


####              Getting Eigen Vals and Vectors            ####
# web ref: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter15.04-Eigenvalues-and-Eigenvectors-in-Python.html
'''
eiger_array = np.array( [ [2, 2, 4], 
                          [1, 3, 5],
                          [2, 3, 4] ] )

eigen_val, eigen_vec = eig(eiger_array)

print(f"eigen_val: {eigen_val}")
print(f"eigen_vec: {eigen_vec}")


print(np.matmul(eiger_array, eiger_array))
'''

# inverted pendulum on a spring
# eigen values are all hyperbolic, so I don't think I can linearize this?


# Constants
mass = 0.5
length = 10
gravity = 10
spring_constant = 0.1

eiger_array_down_state = np.array( [ [0, 1, 0, 0],
                                   [-spring_constant/mass, 0, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, -gravity / (length + (mass*gravity/spring_constant)), 0] ] )

eiger_array_up_state = np.array( [ [0, 1, 0, 0],
                                     [-spring_constant/mass, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, gravity / (length + (mass*gravity/spring_constant)), 0] ] )


down_eigen_val, down_eigen_vec = eig(eiger_array_down_state)

up_eigen_val, up_eigen_vec = eig(eiger_array_up_state)


print(f"up_eigen_val: \n{up_eigen_val}")
print(f"up_eigen_vec: \n{up_eigen_vec}")
print("-------------------------------")

print(f"down_eigen_val: \n{down_eigen_val}")
print(f"down_eigen_vec: \n{down_eigen_vec}")

print(f"Size of down_eigen_val: {len(down_eigen_val)}")


print(np.count_nonzero(up_eigen_vec - np.diag(np.diagonal(up_eigen_vec))))


####              Getting Eigen Vals and Vectors            ####


####                        ODE solving                     ####
# web ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

initial_con = np.array([ 0, 0, 0, 0 ])

time_span = np.array([ 0, 50 ])

times = np.linspace(time_span[0], time_span[1], 50)

# Next steps are to rewatch some control videos, and include the control law into this system
# Possibly use the linearized versions?
# What things did I learn in this session (2/8)?
#   Main thing was practicing using the solve_ivp function
#   Needs a function relating state (input) to the time derivative of the state (output)
#   Needs a time span from initial time to final time
#   Useful to include a number of points to use -- a vector of linearly spaced values from initial time to final time
#   Needs a vector of initial conditions -- ie initial state
# Did some plotting work; copy paste what I'm using to make things easier, or to flesh out utility_plotter.py

def linearized_inverted_spring_string(time, state_vector):
    '''
    State is a vector : [position velocity angular_position angular_velocity]^T
                        [x1       x2       x3               x4              ]^T
    Input: time: array
    Input: state_vector: array

    Returns: state_dot: array
    '''

    # Constants defined for system above /\ /\ /\

    # Control Matrix
    


    # State
    state_position = state_vector[0]

    state_velocity = state_vector[1]

    state_angular_position = state_vector[2]

    state_angular_velocity = state_vector[3]


    # B matrix -- control
    control_matrix = np.array([
                                [-1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, -1]
    ])

    # Control state vector
    pos_control = 0
    vec_control = 0.1
    theta_control = 0
    ang_vel_control = 0.1

    control_vector = np.array([
                                [pos_control],
                                [vec_control],
                                [theta_control],
                                [ang_vel_control]
    ])

    # State Derivatives
    #state_dot = np.matmul(eiger_array_down_state, state_vector)

    state_position_dot = state_velocity

    state_velocity_dot = (-spring_constant/mass) * state_position

    state_angular_position_dot = state_angular_velocity

    state_angular_velocity_dot = ( -gravity / (length + (mass*gravity/spring_constant)) ) * state_angular_position

    # Formatting output as a state variable (array)
    state_dot = np.array([ state_position_dot, state_velocity_dot, state_angular_position_dot, state_angular_velocity_dot ])


    return state_dot

def inverted_spring_string(time, state_vector):
    '''
    State is a vector : [position velocity angular_position angular_velocity]^T
                        [x1       x2       x3               x4              ]^T
    Input: time: array
    Input: state_vector: array

    Returns: state_dot: array
    '''

    # Constants defined for system above /\ /\ /\



    # State
    state_position = state_vector[0]

    state_velocity = state_vector[1]

    state_angular_position = state_vector[2]

    state_angular_velocity = state_vector[3]


    # State Derivatives
    state_position_dot = state_velocity

    state_velocity_dot = (length + state_position) * state_angular_velocity**2 + gravity * math.cos(state_angular_position) - (spring_constant / mass) * state_position

    state_angular_position_dot = state_angular_velocity

    state_angular_velocity_dot = ( -2 / (length + state_position) ) * state_velocity * state_angular_velocity - ( gravity / (length + state_position) ) * math.sin(state_angular_position)


    # Formatting output as a state variable (array)
    state_dot = np.array([ state_position_dot, state_velocity_dot, state_angular_position_dot, state_angular_velocity_dot ])

    return state_dot

sol = solve_ivp(inverted_spring_string, time_span, initial_con, t_eval=times)

sol2 = solve_ivp(linearized_inverted_spring_string, time_span, initial_con, t_eval=times)

'''
print(f"Time values: \n{sol.t}")

print("-------")

print(f"Position Solution: \n{sol.y[0]}")

print(f"Velocity Solution: \n{sol.y[1]}")

print(f"Theta Solution: \n{sol.y[2]}")

print(f"Omega Solution: \n{sol.y[3]}")

'''

# Plotting
plt.rc("font", size = 14)
plt.figure()
plt.plot(sol.t, sol.y[0], '-', label = "Position")
plt.plot(sol.t, sol.y[1], '-', label = "Velocity")
plt.plot(sol.t, sol.y[2], '-', label = "Theta")
plt.plot(sol.t, sol.y[2], '-', label = "Omega")
plt.xlabel("time")
plt.ylabel("state output")
plt.legend()
plt.show()

plt.rc("font", size = 14)
plt.figure()
plt.plot(sol2.t, sol2.y[0], '-', label = "Position")
plt.plot(sol2.t, sol2.y[1], '-', label = "Velocity")
plt.plot(sol2.t, sol2.y[2], '-', label = "Theta")
plt.plot(sol2.t, sol2.y[2], '-', label = "Omega")
plt.xlabel("time")
plt.ylabel("state output")
plt.legend()
plt.show()





####                        ODE solving                     ####

'''
sigma = 0.1

tester_mat = np.array([ [0, 1],
                        [-1, -sigma] ])

tester_val, tester_vec = eig(tester_mat)


print("------")
print(f"tester_val: \n{tester_val}")
print("------")
print(f"tester_vec: \n{tester_vec}")
'''