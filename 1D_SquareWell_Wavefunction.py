#%%

"""
This code generates an animation of a single solution to the Schrodinger equation 
for a particle in a 1D infinite square well, i.e. a wavefunction Psi(x,t) 

The conditions for the 1D infinite square well are:

V(x,t) = 0              for 0 <= x <= a
V(x,t) = infinity       elsewhere  (x < 0, x > a)
psi(0) = psi(a) = 0     boundary conditions -- wavefunction must disappear at bounds

Change the value of n below to view different excited states of the particle

This code can save the animation as a .gif -- you can change the filepath at the 
bottom of the code or keep it commented out. Please note that it may take a few 
minutes to generate the file if you choose to save the file.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# choose parameters
# -------------------------------------------------------------------------------------
n = 3       # energy level

a = 1       # width of the square well
hbar = 1    # use natural units (don't worry about it)
mass = 1    # mass of the particle 


# define constants
# -------------------------------------------------------------------------------------
k = n * np.pi / a                   # spatial frequency of wavefunction
omega = hbar * k**2 / (2 * mass)    # temporal frequency of wavefunction
A = np.sqrt(2 / a)                  # normalization constant

x = np.linspace(0, a, 200)      # x range to plot
t = np.linspace(0, 5, 5000)     # t values (change to adjust animation speed)


# define 1D square well wavefunction
# -------------------------------------------------------------------------------------
psi = A * np.sin(k * x)     # spatial component of wavefunction -- psi(x)

def psi_xt(time):
    return psi * np.exp(-1j * omega * time)     # wavefunction Psi(x,t) at time t


# make animation
# -------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.2))

line, = ax.plot(x, np.real(psi_xt(0)))

ax.set_xlim(0, a)
ax.set_ylim(-1.5, 1.5)
ax.set_title(rf"$\Psi\,(x,t)$ for $n = {n}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\mathrm{Re}[\Psi\,(x,t)]$")

def update(frame):
    time = t[frame]
    line.set_ydata(np.real(psi_xt(time)))
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(t), 3),
    interval=100,       # change interval to adjust animation speed
    blit=True
)

# save animation (optional)
# -------------------------------------------------------------------------------------

# uncomment the code below to save the animation

# ani.save("1DParticle_python.gif",       # save the animation -- can change filename
#          writer=PillowWriter(fps=40))   # change fps (frames per second) to adjust animation speed

plt.show()

#%%