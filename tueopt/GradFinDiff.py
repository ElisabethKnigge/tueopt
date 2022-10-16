"""Plot the elevation of the Tuebingen surrounding."""
import math
import os
import shutil
import urllib.request
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import cm
from scipy import interpolate


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
        zip_target_dir = os.path.dirname(zip_target)

        # download
        print("Downloading data")
        urllib.request.urlretrieve(zip_source, zip_target)

        # extract
        print("Extracting data")
        shutil.unpack_archive(zip_target, extract_dir=zip_target_dir)

    return tif_filename


def elevations(data_file: str, *coords: Iterable[Tuple[float, float]]) -> List[int]:
    """Yield elevations for the requested coordinates using the data file.

    Args:
        data_file: Path to the TIF file that contains the elevation model.
        coords: Pairs of (x, y) coordinates

    Returns:
        List of elevations for the supplied coordinates.
    """
    with rasterio.open(data_file) as src:
        vals = src.sample(coords)
        return [val[0] for val in vals]


src_file = download_data()


# to identify these values, click on a place in google maps, then look them up
# in the link
tue_center = (9.055367217038397, 48.52064681557908)
x_tuec = tue_center[0]
y_tuec = tue_center[1]
tue_center_elevation = elevations(src_file, tue_center)[0]
mpi = (9.0584233, 48.5341389)
x_mpi = 9.0584233
y_mpi = 48.5341389
mpi_elevation = elevations(src_file, mpi)[0]
mvl6 = (9.0515339, 48.5382876)
mvl6_elevation = elevations(src_file, mvl6)[0]
x_ub = 9.03460309417676
y_ub = 48.534037122559994

ymin, xmin = 48.54293574822647, 9.033688803346264
ymax, xmax = 48.51642677684213, 9.092274838312923

steps = 100
xs = np.linspace(xmin, xmax, steps)
ys = np.linspace(ymin, ymax, steps)
zs = np.zeros((steps, steps))

xy_idx = []
coords = []

for x_idx, x_coord in enumerate(xs):
    for y_idx, y_coord in enumerate(ys):
        xy_idx.append((x_idx, y_idx))  # x_idx = welche koordinate
        coords.append((x_coord, y_coord))  # x_coords = x-Koordinate

for (x_idx, y_idx), elevation in zip(xy_idx, elevations(src_file, *coords)):
    zs[y_idx, x_idx] = elevation

f = interpolate.interp2d(xs, ys, zs, kind="cubic")
zf = f(xs, ys)


def grad(f, x, y, alpha, eps_x=1e-3, eps_y=1e-3):
    """Berechnet Gradienten für gegebenen Punkt x, y mit finite differences.

    Args:
        f: Funktion f von x,y.
        x, y: x bzw. y-Wert von Punkt
        alpha: Normalisierungsfaktor (Länge des Vektors oder so)
        eps_x, eps_y: "h-Wert" (Größe der finite diff.)

    Returns:
        Normalisierten Wert des Gradienten (grad_wert).

    """
    grad_x = (f(x + eps_x / 2, y) - f(x - eps_x / 2, y)) / eps_x
    grad_y = (f(x, y + eps_y / 2) - f(x, y - eps_y / 2)) / eps_y
    grad_wert = np.array([grad_x, grad_y])
    norm = alpha / math.sqrt(grad_x[0] ** 2 + grad_y[0] ** 2)
    return grad_wert * norm


def opt(x, y, alpha):
    """Findet theoretisch das lokale Minimum von f.

    Praktisch funktioniert das nur manchmal lol.

    Args:
        x, y: x bzw. y-Wert von Startpunkt

    Returns:
        Liste mit durchlaufenen Punkten.
    """
    z = f(x, y)
    i = f(x, y)
    j = 0
    opt_vec_x = [x]
    opt_vec_y = [y]
    opt_vec_z = [z]
    opt_liste = []

    while i >= 0:
        x_grad, y_grad = -grad(f, x, y, alpha)
        z_wert = f(x, y)
        n_x = x + x_grad[0]
        n_y = y + y_grad[0]
        n_z = f(n_x, n_y)
        i = z_wert - n_z
        j = j + 1
        opt_vec_x = opt_vec_x + [n_x]
        opt_vec_y = opt_vec_y + [n_y]
        opt_vec_z = opt_vec_z + [n_z[0]]
        opt_liste = [opt_vec_x] + [opt_vec_y] + [opt_vec_z]
        z = n_z
        x, y = n_x, n_y
    return opt_liste


def opt_for(x, y, alpha, beta):
    """Findet theoretisch das lokale Minimum von f in beta Schritten.

    Praktisch funktioniert das nur manchmal lol.

    Args:
        x, y: x bzw. y-Wert von Startpunkt
        alpha: Normalisierungsfaktor
        beta: Schritte

    Returns:
        Liste mit durchlaufenen Punkten.
    """
    z = f(x, y)
    j = 0
    opt_vec_x = [x]
    opt_vec_y = [y]
    opt_vec_z = [z]

    plot_vec_x = [x]
    plot_vec_y = [y]
    plot_vec_z = [z]

    # opt_for_liste = []
    plot_liste = []

    for j in range(beta):
        x_grad, y_grad = -grad(f, x, y, alpha)
        n_x = x + x_grad[0]
        n_y = y + y_grad[0]
        n_z = elevations(src_file, (n_x, n_y))
        j = j + 1
        opt_vec_x = opt_vec_x + [n_x]
        opt_vec_y = opt_vec_y + [n_y]
        opt_vec_z = opt_vec_z + [n_z[0]]
        # opt_for_liste = [opt_vec_x] + [opt_vec_y] + [opt_vec_z]
        x, y = n_x, n_y

    for k in range(1000):
        int_val = int((beta / 1000) * k)
        plot_vec_x = plot_vec_x + [opt_vec_x[int_val]]
        plot_vec_y = plot_vec_y + [opt_vec_y[int_val]]
        plot_vec_z = plot_vec_z + [opt_vec_z[int_val]]
        plot_liste = [plot_vec_x] + [plot_vec_y] + [plot_vec_z]
    return plot_liste


fig = plt.figure()

xs_mesh, ys_mesh = np.meshgrid(xs, ys)

ax = fig.add_subplot(121, projection="3d")
surf = ax.plot_surface(
    xs_mesh,
    ys_mesh,
    zs,
    cmap=cm.viridis,
    linewidth=0,
    alpha=0.75,
    antialiased=True,
)

ax2 = fig.add_subplot(122, projection="3d")
surf2 = ax2.plot_surface(
    xs_mesh,
    ys_mesh,
    zs.T,
    cmap=cm.viridis,
    linewidth=0,
    alpha=0.5,
    antialiased=True,
)

# joa
# ax2.quiver(
#     x_mpi,
#     y_mpi,
#     mpi_elevation,
#     norm * float(grad_x[0]),
#     norm * float(grad_y[0]),
#     0,
#     color="red",
#     length=0.4,
# )

color = "purple"

ax2.scatter(x_mpi, y_mpi, mpi_elevation, color=color)
grad_des = opt(x_mpi, y_mpi, 0.00005)
ax2.plot3D(
    grad_des[0],
    grad_des[1],
    grad_des[2],
    linewidth=2,
    color=color,
)

grad_des2 = opt_for(x_ub, y_ub, 0.003, 10000)
print(len(grad_des2), len(grad_des2[0]))
ax.plot3D(
    grad_des2[0],
    grad_des2[1],
    grad_des2[2],
    linewidth=2,
    color="red",
)

grad_des3 = opt(x_mpi, y_mpi, 0.0007)
ax2.plot3D(
    grad_des3[0],
    grad_des3[1],
    grad_des3[2],
    linewidth=2,
    color="blue",
)

# ax2.contour(grad_des[0], grad_des[1], f(grad_des[0], grad_des[1]), cmap=cm.coolwarm)
# --> ich weiß nich genau, was es da macht, aber sieht funny aus

# x = (x_mpi, x_tuec)
# y = (y_mpi, y_tuec)
# z = (mpi_elevation, tue_center_elevation)
# ax.plot3D(x, y, z, color="k")

plt.show()
