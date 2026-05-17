#%%

"""
This code generates a plot of the probability density function for a single solution 
to the Schrodinger equation for a particle in a 1D infinite square well, i.e. |Psi|^2

The conditions for the 1D infinite square well are:

V(x,t) = 0              for 0 <= x <= a
V(x,t) = infinity       elsewhere  (x < 0, x > a)
psi(0) = psi(a) = 0     boundary conditions -- wavefunction must disappear at bounds

Change the value of n below to view probability densities for different excited states 
of the particle

"""

import numpy as np
import matplotlib.pyplot as plt


# choose parameters
# -------------------------------------------------------------------------------------
n = 3       # energy level

a = 1       # width of the square well
hbar = 1    # use natural units (don't worry about it)
mass = 1    # mass of the particle


# define constants
# -------------------------------------------------------------------------------------
k = n * np.pi / a   # spatial frequency of wavefunction
A = np.sqrt(2 / a)  # normalization constant

x = np.linspace(0, a, 200)  # x range to plot


# define 1D square well wavefunction and probability density
# -------------------------------------------------------------------------------------
psi = A * np.sin(k * x)
prob_density = np.abs(psi)**2


# plot probability density
# -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))

ax.plot(x, prob_density)

ax.set_xlim(0, a)
ax.set_ylim(0, 1.1 * prob_density.max())

ax.set_title(rf"$|\Psi(x)|^2$ for $n = {n}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$|\Psi(x)|^2$")

plt.show()

#%%