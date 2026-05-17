#%%

"""
This code generates a plot of the probability density function for a single solution 
to the Schrodinger equation for a particle in a 2D infinite square well, i.e. |Psi|^2

The conditions for the 2D infinite square well are:

V(x,t) = 0                  for 0 <= x <= a  and  0 <= y <= b
V(x,t) = infinity           elsewhere  (x < 0, x > a, y < 0, y > b)
psi(x=0) = psi(x=a) = 0    boundary conditions -- wavefunction must disappear at bounds
psi(y=0) = psi(y=b) = 0    boundary conditions -- wavefunction must disappear at bounds

Change the values of nx and ny below to view probability densities for different excited 
states of the particle

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# choose parameters
# -------------------------------------------------------------------------------------
nx = 4      # x quantum number -- associated with energy level
ny = 4      # y quantum number -- associated with energy level

a = 1       # x dimension of the square well
b = 1       # y dimension of the square well
hbar = 1    # use natural units (don't worry about it)
mass = 1    # mass of the particle


# define constants
# -------------------------------------------------------------------------------------
kx = nx * np.pi / a         # x spatial frequency of wavefunction
ky = ny * np.pi / b         # y spatial frequency of wavefunction
A = np.sqrt(4 / (a * b))    # normalization constant

x = np.linspace(0, a, 101)      # x range to plot
y = np.linspace(0, b, 101)      # y range to plot
X, Y = np.meshgrid(x, y)


# define 2D square well wavefunction and probability density
# -------------------------------------------------------------------------------------

# assume psi(x,y) is a separable function: psi(x,y) = X(x)Y(y)
psi = A * np.sin(kx * X) * np.sin(ky * Y)   # spatial component of wavefunction
prob_density = np.abs(psi)**2


# plot probability density as a colormap plot
# -------------------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 9))

c = ax2.pcolormesh(
    X,
    Y,
    prob_density,
    cmap='plasma',      # can change the color scheme here -- look up Python colormaps
    shading='auto'
)

ax2.set_xlabel("x")
ax2.set_ylabel("y")

ax2.set_title(
    rf"$|\Psi(x,y)|^2$ for $(n_x,n_y)=({nx},{ny})$"
)

# Add colorbar
cb = fig2.colorbar(c)
cb.set_label(r"$|\Psi(x,y)|^2$")

plt.show()


#%%