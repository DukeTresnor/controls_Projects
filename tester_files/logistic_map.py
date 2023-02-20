import numpy as np
from scipy import integrate
from scipy.integrate import odeint
import matplotlib.pyplot as plt




'''
Goal is to graph the logistic map -- x_n+1 = r * x_n * (1 - x_n)

This is an iterative equation, so we want to run a for loop and at the end of each iteration we want to update 
an output list / vector to store each of the values

x_n is the current value, x_n+1 is the output or next value, r is the growth constant. The current value is represented here as a percentage
of the maximum value -- it's between 0 and 1

We also want a time vector

Then we should plot output vs time

'''

# logistic map function
def logistic_map(current_val, time, growth):

    next_val = growth * current_val * (1 - current_val)

    return next_val

# time vector
t0, t1 = 0.0, 20.0
time = np.linspace(t0, t1, 100)


# constants and initial conditions
current_val = 0.1              # is a % of max value
growth = 3.0
output = np.zeros(np.size(time))
output[0] = current_val



# for loop -- for loop iterable range is worded very poorly atm
for i in range(np.size(time) - 1):
    output[i+1] = logistic_map(output[i], time, growth)

# Plotting
plt.plot(time, output)
plt.xlabel('time')
plt.ylabel('output')
plt.show()



'''

Creating and integrating the logistic model

dx/dt = rx(1-x), where x = N/K

N is the population, K is the maximum sustainable population (ie carrying capaciy of the environment)
N --> pop
K --> max_pop
x --> pop_portion
r is the Malthusian parameter -- the rate of maximum possible growth -- max_growth


'''

# model function that returns div_pop_portion, the derivative of the population percentage as a function of time
def logistic_model(pop_portion, time, max_growth):

    div_pop_portion = max_growth * pop_portion * (1 - pop_portion)


    return div_pop_portion

# constants
max_growth = 0.1

# tupliziing parameters
params = (max_growth,)


# time variable
t0, t1 = 0.0, 20.0
time = np.linspace(t0, t1, 100)

# initial conditions
pop_portion_0 = 0.4


# integration
pop_portion = odeint(logistic_model, pop_portion_0, time, params)


# Plotting
plt.plot(time, pop_portion)
plt.xlabel('time')
plt.ylabel('pop_portion')
plt.show()