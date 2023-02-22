from control_testing.northwestern_python_functions import *
import numpy as np
import math
from numpy.linalg import eig
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from collections import namedtuple

# Reference
# Kinematic/Dynamic SLAM for Autonomous Vehicles Using the Linear Parameter Varying Approach -- https://www.mdpi.com/1424-8220/22/21/8211

# World Frame
# x represents a horizontal direction
# y represents a vertical direction

# State variable:
# X = [position, velocity, slip_angle, slip_velocity, angular_position, angular_velocitty]^transpose
State = namedtuple("Scooter_State", ['position', 'velocity', 'slip_angle', 'slip_velocity', 'angular_position', 'angular_velocity'])

# Parameters
friction = 0.01                 # N
gravity = 9.8                   # kgm/s^2
air_density = 1.2        # kg/m^3
drag_coefficient = 0.5
tire_stifness = 15000   # N/rad
cross_area = 4         # m^2
length_center_front = 0.758     # m
length_center_rear = 1.036      # m
mass = 683                      # kg
vehicle_inertia = 561           # kgm^2

# Control Actions -- constant for now
front_wheel_steering_angle = 0 # radians
rear_force_x_direction = 0.001 # N


# Function Definitions
def cos(angle):
    return math.cos(angle)

def sin(angle):
    return math.sin(angle)

def scooter_motion_equations(time, state_vector):
    state = State(*state_vector)

    position = state_vector[0]
    velocity = state_vector[1]
    slip_angle = state_vector[2]
    slip_velocity = state_vector[3]
    angular_position = state_vector[4]
    angular_velocity = state_vector[5]

    front_force_y_direction = tire_stifness * (front_wheel_steering_angle - slip_angle - (length_center_front * angular_velocity / velocity ) )
    rear_force_y_direction = tire_stifness * ( -1 * slip_angle - (length_center_front * angular_velocity / velocity ) )
    drag_component = 0.5 * drag_coefficient * air_density * cross_area

    pos_dot = velocity
    vel_dot = ( rear_force_x_direction * cos(slip_angle) + front_force_y_direction * sin(slip_angle - front_wheel_steering_angle) + rear_force_y_direction * sin(slip_angle) - drag_component * velocity**2 ) / mass - friction * gravity
    slip_angle_dot = slip_velocity
    slip_velocity_dot = (-rear_force_x_direction * sin(slip_angle) + front_force_y_direction * cos(slip_angle - front_wheel_steering_angle) + rear_force_y_direction * cos(slip_angle) ) / (mass * velocity) - angular_velocity
    ang_pos_dot = angular_velocity
    ang_vel_dot = ( front_force_y_direction * length_center_front * cos(front_wheel_steering_angle) - rear_force_y_direction * length_center_rear ) / vehicle_inertia

    state_dot = np.array([pos_dot, vel_dot, slip_angle_dot, slip_velocity_dot, ang_pos_dot, ang_vel_dot] )

    return state_dot


# Setting up plots and graphing
initial_con = np.array([ 0, 1, 0, 0, 0, 0 ])

time_span = np.array([ 0, 50 ])

times = np.linspace(time_span[0], time_span[1], 50)

sol = solve_ivp(scooter_motion_equations, time_span, initial_con, t_eval=times)

print(" ---------------------------------- ")


plt.rc("font", size = 14)
plt.figure()
plt.plot(sol.t, sol.y[0], '-', label = "Position")
plt.plot(sol.t, sol.y[1], '-', label = "Velocity")
plt.plot(sol.t, sol.y[2], '-', label = "Slip Angle")
plt.plot(sol.t, sol.y[3], '-', label = "Slip Angle Velocity")
plt.plot(sol.t, sol.y[4], '-', label = "Angular Position")
plt.plot(sol.t, sol.y[5], '-', label = "Angular Velocity")
plt.xlabel("time")
plt.ylabel("state output")
plt.legend()
plt.show()
