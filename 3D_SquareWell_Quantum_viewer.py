#%%

"""
This code generates an interactive plot of the probability density function for a 
single solution to the Schrodinger equation for a particle in a 3D infinite square well, 
i.e. |Psi|^2

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

In the pop-up window, click on the plot and drag to view different orientations. You 
can re-size the pop-up window to make it bigger.

"""


import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


# the colormaps I'm using make the plot look grey in some areas
# cut these from the colormaps
def truncate_colormap(cmap_name, minval=0.3, maxval=1.0, n=256):
    cmap = plt.get_cmap(cmap_name, n)
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap_name})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

reds_dark = truncate_colormap("Reds", 0.35, 1.0)
blues_dark = truncate_colormap("Blues", 0.35, 1.0)


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
N = 80
x = np.linspace(0, a, N)    # x range to plot
y = np.linspace(0, b, N)    # y range to plot
z = np.linspace(0, c, N)    # z range to plot

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")


# define 3D infinite square well wavefunction and probability density
# -------------------------------------------------------------------------------------
A = np.sqrt(8 / (a * b * c))        # normalization constant

kx = nx * np.pi / a        # x spatial frequency of wavefunction
ky = ny * np.pi / b        # y spatial frequency of wavefunction
kz = nz * np.pi / c        # z spatial frequency of wavefunction

# assume psi(x,y,z) is a separable function: psi(x,y,z) = X(x)Y(y)Z(z)
psi = A * np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z)
prob_density = psi**2

# normalize probability density
P = prob_density / prob_density.max()


# keep only reasonably visible points
# -------------------------------------------------------------------------------------
cutoff = 0.03

mask = P > cutoff

points = np.column_stack(
    (
        X[mask],
        Y[mask],
        Z[mask]
    )
)

# sign of wavefunction
sign = np.sign(psi[mask])

# probability density values
P_visible = P[mask]


# create PyVista point cloud
# -------------------------------------------------------------------------------------
cloud = pv.PolyData(points)

cloud["probability density"] = P_visible
cloud["sign"] = sign


# make a 3D interactive plot
# -------------------------------------------------------------------------------------
plotter = pv.Plotter()

# positive wavefunction cloud
positive = cloud.threshold(
    value=0.5,
    scalars="sign"
)

plotter.add_mesh(
    positive,
    cmap=reds_dark,
    opacity=0.1,
    point_size=8,
    render_points_as_spheres=True,
    show_scalar_bar=False
)

# negative wavefunction cloud
negative = cloud.threshold(
    value=(-1.5, -0.5),
    scalars="sign"
)

plotter.add_mesh(
    negative,
    cmap=blues_dark,
    opacity=0.1,
    render_points_as_spheres=True,
    show_scalar_bar=False
)

# format axes and labels
# -------------------------------------------------------------------------------------

# add box around the well
box = pv.Box(bounds=(0, a, 0, b, 0, c))

plotter.add_mesh(
    box,
    style="wireframe",
    color="black",
    line_width=2,
    show_scalar_bar=False
)

plotter.add_title(
    rf"$n_x =$ {nx}, $n_y =$ {ny}, $n_z =$ {nz}",
    font_size=18
)

plotter.show_axes()
plotter.show_grid()

plotter.camera_position = "iso"

plotter.show()

#%%