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

ymin, xmin = 48.54293574822647, 9.033688803346264
ymax, xmax = 48.51642677684213, 9.092274838312923

steps = 10
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


def nächster_punkt(punkt):
    """Nächster Punkt für Punkt P.

    Args:
        Punkt: x, y-Koordinaten von Punkt P.

    Returns:
        Punkt N der Punkt P im Datenset am nächsten ist.s
    """
    dist = []
    for i in range(len(coords)):
        xy_k = punkt
        z_k = elevations(src_file, xy_k)
        dist = dist + [
            math.sqrt(
                (xy_k[0] - coords[i][0]) ** 2
                + (xy_k[1] - coords[i][1]) ** 2
                + (z_k[0] - elevations(src_file, coords[i])) ** 2
            )
        ]

    min_val = min(dist)
    min_ind = dist.index(min_val)

    n_punkt = coords[min_ind]

    return n_punkt


# def kürzeste_distanz(startpunkt, endpunkt):
#     plotliste = []
#     while startpunkt != endpunkt:
#         if elevations(src_file, startpunkt) >= elevations(
#             src_file, vier_vergleich_z(startpunkt)
#         ):
#             startpunkt = vier_vergleich_z(startpunkt)
#             plotliste = plotliste + [vier_vergleich_z(startpunkt)]
#         else:
#             startpunkt = vier_vergleich_xy(startpunkt)
#             plotliste = plotliste + [vier_vergleich_xy(startpunkt)]

#     return plotliste


# def vier_vergleich_z(punkt):
#     z_liste = []
#     vier_dist_z = []
#     global xs_l, ys_l
#     xs_l = xs.tolist()
#     ys_l = ys.tolist()
#     i = xs_l.index(punkt[0])
#     j = ys_l.index(punkt[1])
#     a = [xs_l[i + 1], ys_l[j + 1]]
#     b = [xs_l[i - 1], ys_l[j - 1]]
#     c = [xs_l[i + 1], ys_l[j - 1]]
#     d = [xs_l[i - 1], ys_l[j + 1]]
#     z_liste = [a] + [b] + [c] + [d]

#     for i in range(4):
#         vier_dist_z = vier_dist_z + [
#             math.sqrt(
#                 abs(
#                     elevations(src_file, z_liste[i])[0] - elevations(src_file, punkt)[0]
#                 )
#             )
#             ** 2
#         ]
#         punkt = min(vier_dist_z)
#         return punkt


# def vier_vergleich_xy(punkt):
#     xy_liste = []
#     vier_dist_xy = []
#     i = xs_l.index(punkt[0])
#     j = ys_l.index(punkt[1])
#     a = [xs_l[i + 1], ys_l[j + 1]]
#     b = [xs_l[i - 1], ys_l[j - 1]]
#     c = [xs_l[i + 1], ys_l[j - 1]]
#     d = [xs_l[i - 1], ys_l[j + 1]]
#     xy_liste = [a] + [b] + [c] + [d]

#     for i in range(4):
#         vier_dist_xy = vier_dist_xy + [
#             math.sqrt(
#                 (xy_liste[i][0] - punkt[0]) ** 2 + (xy_liste[i][1] - punkt[1]) ** 2
#             )
#         ]
#         punkt = min(vier_dist_xy)
#         return punkt


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

xs_mesh, ys_mesh = np.meshgrid(xs, ys)
surf = ax.plot_surface(
    xs_mesh,
    ys_mesh,
    zs,
    cmap=cm.viridis,
    linewidth=0,
    alpha=0.75,
    antialiased=True,
)

n_mpi = nächster_punkt((x_mpi, y_mpi))
n_center = nächster_punkt((x_tuec, y_tuec))

# print(kürzeste_distanz(n_mpi, n_center))

ax.scatter(x_mpi, y_mpi, mpi_elevation, color="k")
ax.scatter(x_tuec, y_tuec, tue_center_elevation, color="k")
ax.scatter(n_mpi[0], n_mpi[1], elevations(src_file, n_mpi), color="green")
ax.scatter(n_center[0], n_center[1], elevations(src_file, n_center), color="green")
# ax.plot3D(plotliste)

# x = (x_mpi, x_tuec)
# y = (y_mpi, y_tuec)
# z = (mpi_elevation, tue_center_elevation)
# ax.plot3D(x, y, z, color="k")

fig.colorbar(surf)
plt.show()
