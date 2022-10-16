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
mpi = (9.0584233, 48.5341389)
x_mpi = 9.0584233
y_mpi = 48.5341389
x_ub = 9.03460309417676
y_ub = 48.534037122559994

ymax, xmin = 48.54293574822647, 9.033688803346264
ymin, xmax = 48.51642677684213, 9.092274838312923

steps = 100
xs = np.linspace(xmin, xmax, steps)
ys = np.linspace(ymin, ymax, steps)
zs = np.zeros((steps, steps))

p_steps = 10
xs_p = np.linspace(xmin, xmax, p_steps)
ys_p = np.linspace(ymin, ymax, p_steps)
zs_p = np.zeros((p_steps, p_steps))

xy_idx = []
coords = []

for x_idx, x_coord in enumerate(xs):
    for y_idx, y_coord in enumerate(ys):
        xy_idx.append((x_idx, y_idx))  # x_idx = welche koordinate
        coords.append((x_coord, y_coord))  # x_coords = x-Koordinate

for (x_idx, y_idx), elevation in zip(xy_idx, elevations(src_file, *coords)):
    zs[y_idx, x_idx] = elevation

xy_idx_p = []
coords_p = []

for x_idx_p, x_coord_p in enumerate(xs_p):
    for y_idx_p, y_coord_p in enumerate(ys_p):
        xy_idx_p.append((x_idx_p, y_idx_p))  # x_idx = welche koordinate
        coords_p.append((x_coord_p, y_coord_p))  # x_coords = x-Koordinate

for (x_idx_p, y_idx_p), elevation in zip(xy_idx_p, elevations(src_file, *coords_p)):
    zs_p[y_idx_p, x_idx_p] = elevation

# Interpolation und Ableitungen
f = interpolate.interp2d(xs, ys, zs, kind="cubic")
# p = xs**2 + ys**2
p = interpolate.interp2d(xs_p, ys_p, zs_p, kind="cubic")
zf = f(xs, ys)
zp = p(xs, ys)

der_x = p(xs, ys, dx=1, dy=0)  # 100*100 array
der_y = p(xs, ys, dx=0, dy=1)  # 100*100 array
der_huh = p(xs, ys, dx=1, dy=1)


def opt_int(x, y, alpha, beta):
    """Berechnet Gradient für x, y.

    Args:
        x, y = Koordinaten Startpunkt
        alpha = Normalisierung
        beta = Widerholungen

    Returns:
        opt_liste = Liste, mit Punkten die zu Minimum führen.
    """
    i = 0
    z = f(x, y)[0]
    opt_liste = [[x], [y], [z]]
    a_list = [alpha]

    for i in range(beta):
        y_grad = -p(x, y, dx=0, dy=1)
        x_grad = -p(x, y, dx=1, dy=0)
        norm = alpha / math.sqrt(x_grad**2 + y_grad**2)
        x = x + x_grad * norm
        y = y + y_grad * norm
        opt_liste[0].append(x[0])
        opt_liste[1].append(y[0])
        opt_liste[2].append(p(x, y)[0])
        a_list.append(alpha)

        if (x[0] >= xmax or x[0] <= xmin) or (y[0] >= ymax or y[0] <= ymin):
            print("Punkt nich in Fläche duh, Iterations: ", i)
            break

        # a = opt_liste[0].count(x[0])
        # b = opt_liste[1].count(y[0])
        # if a and b >= 2:
        #     alpha = alpha * 1000
        #     a_list.append(alpha)

        print(i, "von", beta)

    return opt_liste, a_list


# Plots

xs_mesh, ys_mesh = np.meshgrid(xs, ys)
xs_p_mesh, ys_p_mesh = np.meshgrid(xs_p, ys_p)

fig = plt.figure()
ax = fig.add_subplot(121, projection="3d")
ax.set_title("True Surface")
surf = ax.plot_surface(xs_mesh, ys_mesh, zs, cmap=cm.twilight, alpha=0.75)
ax4 = fig.add_subplot(122, projection="3d")
ax4.set_title("Interpolation")
surf4 = ax4.plot_surface(xs_mesh, ys_mesh, zp, cmap=cm.twilight, alpha=0.75)

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(122, projection="3d")
# ax2.set_title("?")
# surf2 = ax2.plot_surface(xs_mesh, ys_mesh, der_x.T, cmap=cm.cubehelix)
# ax3 = fig2.add_subplot(121, projection="3d")
# ax3.set_title("?")
# surf3 = ax3.plot_surface(xs_mesh, ys_mesh, der_y, cmap=cm.cubehelix)

opt, a_liste = opt_int(x_mpi, y_mpi, 0.00015, 10000)  # sehr gute val=0.000015
# opt2, a_liste2 = opt_int(x_mpi, y_mpi, 0.0001, 100000)
step = 10000
ex = opt[0][step]
ey = opt[1][step]
ez = opt[2][step]


ax4.scatter(opt[0], opt[1], opt[2], linewidth=1, c=opt[2], cmap=cm.hsv, s=0.89)
# ax.plot3D(opt2[0], opt2[1], opt2[2], linewidth=2, color="slateblue", label="MPI")
ax4.scatter(x_mpi, y_mpi, p(x_mpi, y_mpi), color="magenta", s=0.8)
ax4.scatter(ex, ey, ez, color="magenta", s=7)

# fig4 = plt.figure()
# a_liste = pd.DataFrame(a_liste)
# # a_liste2 = pd.DataFrame(a_liste2)
# ax6 = fig4.add_subplot(121)
# surf6 = ax6.plot(a_liste)
# ax7 = fig4.add_subplot(122)
# # surf7 = ax7.plot(a_liste2)

# fig3 = plt.figure()
# der_x = np.array(der_x)
# der_y = np.array(der_y)
# ax5 = fig3.add_subplot()
# ys_mesh = np.array(ys_mesh)
# ys_mesh_neu = ys_mesh[np.argsort(ys_mesh[:, 0])]
# ax5.streamplot(
#     xs_mesh,
#     ys_mesh_neu,
#     -der_x,
#     -der_y,
#     density=(2),
#     linewidth=zf * 0.001,
#     arrowstyle="fancy",
#     color=zf,
#     cmap="plasma",
# )


plt.show()
