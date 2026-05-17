#%%

"""
This code generates a plot of the probability density function for a single solution 
to the Schrodinger equation for a particle in a 3D infinite square well, i.e. |Psi|^2

The conditions for the 3D infinite square well are:

V(x,y,z,t) = 0                  for 0 <= x <= a,  0 <= y <= b,  and 0 <= z <= c
V(x,y,z,t) = infinity           elsewhere  (x < 0, x > a, y < 0, y > b, z < 0, z > c)
psi(x=0) = psi(x=a) = 0    boundary conditions -- wavefunction must disappear at bounds
psi(y=0) = psi(y=b) = 0    boundary conditions -- wavefunction must disappear at bounds
psi(z=0) = psi(z=c) = 0    boundary conditions -- wavefunction must disappear at bounds

Red plotted regions indicate locations where the wavefunction Psi(x,y,z,t) is positive,
blue plotted regions indicate locations where Psi(x,y,z,t) is negative.

Change the values of nx, ny, and nz below to view probability densities for different 
excited states of the particle.

"""


import numpy as np
import matplotlib.pyplot as plt


# choose quantum numbers and well dimensions
# -------------------------------------------------------------------------------------
nx = 2      # x quantum number -- associated with energy level
ny = 1      # y quantum number -- associated with energy level
nz = 3      # z quantum number -- associated with energy level

a = 1       # x dimension of the square well
b = 1       # y dimension of the square well
c = 1       # z dimension of the square well


# set up plot grid
# -------------------------------------------------------------------------------------
N = 100
x = np.linspace(0, a, N)    # x range to plot
y = np.linspace(0, b, N)    # y range to plot
z = np.linspace(0, c, N)    # z range to plot

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# define 3D infinite square well wavefunction and probability density
# -------------------------------------------------------------------------------------
A = np.sqrt(8 / (a * b * c))     # normalization constant

kx = nx * np.pi / a         # x spatial frequency of wavefunction
ky = ny * np.pi / b         # y spatial frequency of wavefunction
kz = nz * np.pi / c         # z spatial frequency of wavefunction

# assume psi(x,y,z) is a separable function: psi(x,y,z) = X(x)Y(y)Z(z)
psi = A * np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z)
prob_density = psi**2

# normalize probability density
P = prob_density / prob_density.max()


# make a 3D plot
# -------------------------------------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

# only keep points above a small probability-density cutoff
cutoff = 0.03

mask_pos = (P > cutoff) & (psi > 0)     # keep track of where wavefunction is positive
mask_neg = (P > cutoff) & (psi < 0)     # keep track of where wavefunction is negative

# make the opacity proportional to the probability density
alpha_pos = 0.08 * P[mask_pos]**1.5
alpha_neg = 0.08 * P[mask_neg]**1.5

colors_pos = [(1, 0, 0, alpha) for alpha in alpha_pos]
colors_neg = [(0, 0, 1, alpha) for alpha in alpha_neg]

# plot of positive wavefunction regions
ax.scatter(
    X[mask_pos],
    Y[mask_pos],
    Z[mask_pos],
    c=colors_pos,
    s=1,
    depthshade=False
)

# plot of negative wavefunction regions
ax.scatter(
    X[mask_neg],
    Y[mask_neg],
    Z[mask_neg],
    c=colors_neg,
    s=1,
    depthshade=False
)


# draw cube edges
# -------------------------------------------------------------------------------------
edges = [
    [(0, 0, 0), (a, 0, 0)], [(0, b, 0), (a, b, 0)],
    [(0, 0, c), (a, 0, c)], [(0, b, c), (a, b, c)],

    [(0, 0, 0), (0, b, 0)], [(a, 0, 0), (a, b, 0)],
    [(0, 0, c), (0, b, c)], [(a, 0, c), (a, b, c)],

    [(0, 0, 0), (0, 0, c)], [(a, 0, 0), (a, 0, c)],
    [(0, b, 0), (0, b, c)], [(a, b, 0), (a, b, c)]
]

for edge in edges:
    xs, ys, zs = zip(*edge)
    ax.plot(xs, ys, zs, color="black", linewidth=1)


# format axes and labels
# -------------------------------------------------------------------------------------
ax.set_xlim(0, a)
ax.set_ylim(0, b)
ax.set_zlim(0, c)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_title(rf"$n_x={nx},\ n_y={ny},\ n_z={nz}$")

plt.show()

#%%