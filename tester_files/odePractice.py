import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def vdp1(t, y):
    return np.array([y[1], (1.0 - y[0]**2)*y[1] - y[0]])

t0, t1 = 0, 20                                      # start and end
t = np.linspace(t0, t1, 100)                        # the points of evaluation of solution
y0 = [2, 0]                                         # initial value
y = np.zeros((len(t), len(y0)))                     # array for solution
y[0, :] = y0
r = integrate.ode(vdp1).set_integrator("dopri5")    # choice of method -- r is the integrator ode for vdp1, using the integrator method of "dopr15" I think this is the interpretation of it?
r.set_initial_value(y0, t0)                         # initial values
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

# jacobian input would gp into r = integrate.ode(vdp1, jac) line?