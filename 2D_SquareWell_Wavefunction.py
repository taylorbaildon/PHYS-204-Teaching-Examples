#%%

"""
This code generates an animation of a single solution to the Schrodinger equation 
for a particle in a 2D infinite square well, i.e. a wavefunction Psi(x,y,t) 

The conditions for the 2D infinite square well are:

V(x,t) = 0                  for 0 <= x <= a  and  0 <= y <= b
V(x,t) = infinity           elsewhere  (x < 0, x > a, y < 0, y > b)
psi(x=0) = psi(x=a) = 0    boundary conditions -- wavefunction must disappear at bounds
psi(y=0) = psi(y=b) = 0    boundary conditions -- wavefunction must disappear at bounds

Change the values of nx and ny below to view different excited 
states of the particle

This code can save the animation as a .gif -- you can change the filepath at the 
bottom of the code or keep it commented out. Please note that it may take a few 
minutes to generate the file if you choose to save the file.

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# choose parameters
# -------------------------------------------------------------------------------------
nx = 2      # x quantum number -- associated with energy level
ny = 3      # y quantum number -- associated with energy level

a = 1       # x dimension of the square well
b = 1       # y dimension of the square well
hbar = 1    # use natural units (don't worry about it)
mass = 1    # mass of the particle


# define constants
# -------------------------------------------------------------------------------------
kx = nx * np.pi / a                             # x spatial frequency of wavefunction
ky = ny * np.pi / b                             # y spatial frequency of wavefunction
omega = hbar * (kx**2 + ky**2) / (2 * mass)     # temporal frequency of wavefunction
A = np.sqrt(4 / (a * b))                        # normalization constant

x = np.linspace(0, a, 101)      # x range to plot
y = np.linspace(0, b, 101)      # y range to plot
X, Y = np.meshgrid(x, y)

t = np.linspace(0, 5, 4000)     # t values (change to adjust animation speed)


# define 2D square well wavefunction
# -------------------------------------------------------------------------------------

# assume psi(x,y) is a separable function: psi(x,y) = X(x)Y(y)
psi = A * np.sin(kx * X) * np.sin(ky * Y)       # spatial component of wavefunction

def psi_xyt(time):
    return psi * np.exp(-1j * omega * time)     # wavefunction Psi(x,y,t) at time t


# make animation
# -------------------------------------------------------------------------------------
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

Z0 = np.real(psi_xyt(0))

surf = ax.plot_surface(X, Y, Z0, cmap='twilight', edgecolor='none')

ax.set_xlim(0, a)
ax.set_ylim(0, b)
ax.set_zlim(-A, A)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$\mathrm{Re}[\Psi(x,y,t)]$")

ax.set_title(rf"$\Psi\,(x,y,t)$ for $(n_x,n_y)=({nx},{ny})$")

def update(frame):
    ax.clear()

    time = t[frame]
    Z = np.real(psi_xyt(time))

    ax.plot_surface(X, Y, Z, cmap='twilight', edgecolor='none')

    ax.set_xlim(0, a)
    ax.set_ylim(0, b)
    ax.set_zlim(-A, A)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\mathrm{Re}[\Psi\,(x,y,t)]$")

    ax.set_title(rf"$\Psi\,(x,y,t)$ for $(n_x,n_y)=({nx},{ny})$")

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(t), 4),
    interval=100,       # change interval to adjust animation speed
)

# save animation (optional)
# -------------------------------------------------------------------------------------

# uncomment the code below to save the animation

# ani.save("2DParticle42_python.gif",     # save the animation -- can change filename
#          writer=PillowWriter(fps=45))   # change fps (frames per second) to adjust animation speed

plt.show()


#%%
