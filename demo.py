import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import urllib.request

HERE = os.path.abspath(__file__)
HEREDIR = os.path.dirname(HERE)


# SRTM Digital Surface Model of Germany (https://opendem.info/download_srtm.html)
filename = "srtm_germany_dsm"
tif_filename = os.path.join(HEREDIR, f"{filename}.tif")

if os.path.exists(tif_filename):
    print("Data already downloaded. Skipping.")
else:
    zip_source = f"https://opendem.info/downloads/{filename}.zip"
    zip_target = os.path.join(HEREDIR, f"{filename}.zip")

    # download
    print("Downloading data")
    urllib.request.urlretrieve(zip_source, zip_target)

    # extract
    import shutil
    print("Extracting data")
    shutil.unpack_archive(zip_target)

def get_elevation(x, y):
    coord = (x,y)
    coords = (coord,)

    with rasterio.open(tif_filename) as src:
        vals = src.sample(coords)
        for val in vals:
            return val[0]

# global
# xmin, xmax = 5.0, 16.0
# ymin, ymax = 47.0, 56.0

# to identify these values, click on a place in google maps, then look them up
# in the link
tue_center_x, tue_center_y = 8.9735997, 48.4890488
tue_center_elevation = get_elevation(tue_center_x, tue_center_y)
mpi_x, mpi_y = 9.0584233, 48.5341389
mpi_elevation = get_elevation(mpi_x, mpi_y)
mvl6_x, mvl6_y = 9.0515339, 48.5382876
mvl6_elevation = get_elevation(mvl6_x, mvl6_y)

xmin, xmax = 9.042815, 9.091223
ymin, ymax = 48.514583, 48.545057

steps = 100
xs = np.linspace(xmin, xmax, steps)
ys = np.linspace(ymin, ymax, steps)
zs = np.zeros((steps, steps))

for x_idx, x_coord in enumerate(xs):
    for y_idx, y_coord in enumerate(ys):
        zs[y_idx, x_idx] = get_elevation(x_coord, y_coord)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Circle, Ellipse

def add_point(ax, x, y, z, fc = None, ec = None, radius = 0.005):
    """https://stackoverflow.com/a/65115447"""

    xy_len, z_len = ax.get_figure().get_size_inches()
    axis_length = [x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]]
    axis_rotation =  {'z': ((x, y, z), axis_length[1]/axis_length[0]),
                        'y': ((x, z, y), axis_length[2]/axis_length[0]*xy_len/z_len),
                        'x': ((y, z, x), axis_length[2]/axis_length[1]*xy_len/z_len)}
    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Ellipse((x0, y0), width = radius, height = radius*ratio, fc=fc, ec=ec)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)

xs_mesh, ys_mesh = np.meshgrid(xs, ys)
surf = ax.plot_surface(xs_mesh, ys_mesh, np.log10(zs + 1), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
add_point(ax, mpi_x, mpi_y, np.log10(mpi_elevation + 1), radius=0.001)
fig.colorbar(surf)
plt.show()
