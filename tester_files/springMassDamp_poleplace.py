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
springM = importlib.import_module('springMassDamp')

# Dynamics Matrix
dyn_mat = np.array([

    [0.0, 1.0],
    [-(springM.spring / springM.blockmass), -(springM.damp / springM.blockmass)]

])



# Input Matrix

