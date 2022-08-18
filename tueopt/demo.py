"""Plot the elevation of the Tuebingen surrounding."""
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

        # download
        print("Downloading data")
        urllib.request.urlretrieve(zip_source, zip_target)

        # extract
        print("Extracting data")
        shutil.unpack_archive(zip_target)

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

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

xs_mesh, ys_mesh = np.meshgrid(xs, ys)
surf = ax.plot_surface(
    xs_mesh, ys_mesh, zs, cmap=cm.coolwarm, linewidth=0, antialiased=True
)
fig.colorbar(surf)
plt.show()
