import os
import shutil
import urllib.request
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy
from matplotlib import cm
from matplotlib.patches import Ellipse  # Circle
from mpl_toolkits.mplot3d import art3d  # Axes3D


def download_data() -> str:
    """Download elevation model data and return the raster file.

    Skips download if file already exists.

    The data comes from the SRTM Digital Surface Model of Germany. See
    https://opendem.info/download_srtm.html.

    Returns:
        Path to raster file.
    """
    HERE = os.path.abspath(__file__)
    HEREDIR = os.path.dirname(HERE)
    # global
    # xmin, xmax = 5.0, 16.0
    # ymin, ymax = 47.0, 56.0
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
        print("Extracting data")
        shutil.unpack_archive(zip_target)

    return tif_filename


def elevations(data_file: str, *coords: Iterable[Tuple[float, float]]) -> List[int]:
    """Yield elevations for the requested coordinates using the data file."""
    with rasterio.open(data_file) as src:
        vals = src.sample(coords)
        return [val[0] for val in vals]


src_file = download_data()


# to identify these values, click on a place in google maps, then look them up
# in the link
tue_center = (8.9735997, 48.4890488)
tue_center_elevation = elevations(src_file, tue_center)[0]
mpi = (9.0584233, 48.5341389)
mpi_elevation = elevations(src_file, mpi)[0]
mvl6 = (9.0515339, 48.5382876)
mvl6_elevation = elevations(src_file, mvl6)[0]

xmin, xmax = 9.042815, 9.091223
ymin, ymax = 48.514583, 48.545057

steps = 100
xs = np.linspace(xmin, xmax, steps)
ys = np.linspace(ymin, ymax, steps)
zs = np.zeros((steps, steps))

xy_idx = []
coords = []

for x_idx, x_coord in enumerate(xs):
    for y_idx, y_coord in enumerate(ys):
        xy_idx.append((x_idx, y_idx))
        coords.append((x_coord, y_coord))

for (x_idx, y_idx), elevation in zip(xy_idx, elevations(src_file, *coords)):
    zs[y_idx, x_idx] = elevation

f = scipy.interpolate.RectBivariateSpline(xs, ys, zs.T)
# print(f(*mpi))
# print(mpi_elevation)


f_grad_x = f.partial_derivative(1, 0)
f_grad_y = f.partial_derivative(0, 1)


def f_grad(x, y):
    return np.array([f_grad_x(x, y)[0][0], f_grad_y(x, y)[0][0]])


print(f_grad(*mpi))


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})



def add_point(ax, x, y, z, fc=None, ec=None, radius=0.005):
    """https://stackoverflow.com/a/65115447"""

    xy_len, z_len = ax.get_figure().get_size_inches()
    axis_length = [
        x[1] - x[0] for x in [ax.get_xbound(), ax.get_ybound(), ax.get_zbound()]
    ]
    axis_rotation = {
        "z": ((x, y, z), axis_length[1] / axis_length[0]),
        "y": ((x, z, y), axis_length[2] / axis_length[0] * xy_len / z_len),
        "x": ((y, z, x), axis_length[2] / axis_length[1] * xy_len / z_len),
    }
    for a, ((x0, y0, z0), ratio) in axis_rotation.items():
        p = Ellipse((x0, y0), width=radius, height=radius * ratio, fc=fc, ec=ec)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z0, zdir=a)


xs_mesh, ys_mesh = np.meshgrid(xs, ys)
surf = ax.plot_surface(
    xs_mesh, ys_mesh, zs, cmap=cm.coolwarm, linewidth=0, antialiased=True
)
add_point(ax, *mpi, mpi_elevation, radius=0.001)
fig.colorbar(surf)
plt.show()
