"""Surface, interpolation and streamplot of gradients between 2 points in Germany."""
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

yz, xz = (51.314342229660475, 12.413751897373517)
p1 = (51.37851084786834, 12.281719132563873)
p2 = (51.30740546223145, 12.437053610317154)


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


def surface(Punkt1, Punkt2, Dichte):
    """Arrays of x/y-coordinates and elevations.

    Args:
        Punkt1, Punkt2: x/y-coordinates
        Dichte: integer

    Returns:
        X,Y,Z: seperate arrays of x,y,z-coordinates
    """
    global ymin, ymax, xmin, xmax

    ymax, xmin = Punkt1
    ymin, xmax = Punkt2
    steps = Dichte

    X = np.linspace(xmin, xmax, steps)
    Y = np.linspace(ymin, ymax, steps)
    Z = np.zeros((steps, steps))

    xy_idx = []
    coords = []

    for x_idx, x_coord in enumerate(X):
        for y_idx, y_coord in enumerate(Y):
            xy_idx.append((x_idx, y_idx))  # x_idx = welche koordinate
            coords.append((x_coord, y_coord))  # x_coords = x-Koordinate

    for (x_idx, y_idx), elevation in zip(xy_idx, elevations(src_file, *coords)):
        Z[y_idx, x_idx] = elevation

    return X, Y, Z


f_steps = 50
X, Y, Z = surface(
    (51.37851084786834, 12.281719132563873),
    (51.30740546223145, 12.437053610317154),
    f_steps,
)
f = interpolate.interp2d(X, Y, Z, kind="cubic")

p_steps = 10
XP, YP, ZP = surface(p1, p2, p_steps)
p = interpolate.interp2d(XP, YP, ZP, kind="cubic")


def RectBivariateSpline(p1, p2, steps):
    """Interpolation over x,y,z-coordinates.

    Args:
        steps: integer (steps < dichte)

    Returns:
        p: Interpolated z-coordinates.
    """
    XP, YP, ZP = surface(p1, p2, steps)
    p = interpolate.RectBivariateSpline(XP, YP, ZP, kx=1, ky=1)
    Z_Int = p(X, Y)

    return Z_Int


def Gradient(p1, p2):
    """Calculates gradients of array oder so.

    Args:
        p1, p2: x,y-coordinates for grid-limits

    Returns:
        gx, gy: arrays containing the first-order derivatives.
    """
    gx = f(X, Y, dx=1, dy=0)
    gy = f(X, Y, dx=0, dy=1)

    return gx, gy


class GD:
    """Gradient descent algorithm."""

    def __init__(self, x, y, step_length, momentum=0.0):
        """Set up the gradient descent method.

        Args:
            x: First variable to optimize over.
            y: Second variable to optimize over.
            step_length: Length of the GD update (no momentum).
            momentum: Momentum factor.
        """
        self._x = x
        self._y = y
        self._change_x = [0.0]
        self._change_y = [0.0]
        self._step_length = step_length
        self._momentum = momentum

    def step(self, grad_x, grad_y):
        """Perform one iteration of gradient descent.

        Args:
            grad_x: Gradient w.r.t. x.
            grad_y: Gradient w.r.t. y.
        """
        norm = math.sqrt(grad_x**2 + grad_y**2)
        scaling_factor = self._step_length / norm

        last_change_x = self._change_x[-1]
        self._change_x.append(scaling_factor * grad_x + self._momentum * last_change_x)
        last_change_y = self._change_y[-1]
        self._change_y.append(scaling_factor * grad_y + self._momentum * last_change_y)

        self._x = self._x - self._change_x[-1]
        self._y = self._y - self._change_y[-1]

    def get_x(self):
        """Return x variable."""
        return self._x

    def get_y(self):
        """Return y variable."""
        return self._y


def so(x, y, f, alpha, beta, momentum):
    """Berechnet Gradient für x, y.

    Args:
        x, y: Koordinaten Startpunkt
        f: Interpolationsding
        alpha: Normalisierung
        beta: Widerholungen
        momentum: float betwen [0, 1] --> 0 = gd ohne momentum

    Returns:
        opt_liste: Liste, mit Punkten die zu Minimum führen.
    """
    p = f
    z = p(x, y)[0]
    opt_liste = [[x], [y], [z]]

    optimizer = GD(x, y, alpha, momentum=momentum)

    for i in range(beta):
        y_grad = p(x, y, dx=0, dy=1)
        x_grad = p(x, y, dx=1, dy=0)

        optimizer.step(x_grad, y_grad)

        x = optimizer.get_x()
        y = optimizer.get_y()

        opt_liste[0].append(x)
        opt_liste[1].append(y)
        opt_liste[2].append(p(x, y)[0])

        if (x >= xmax or x <= xmin) or (y >= ymax or y <= ymin):
            print("Punkt nich in Fläche duh, Iterations: ", i)
            break

        print(i, "von", beta)

    return opt_liste


def adam(x, y, f, steps, alpha, beta1, beta2):
    """Adam-Optimizer.

    Args:
        :)

    Returns:
        adam_list: Liste mit allen Punkten die Adam bis zum convergen durchläuft.
    """
    p = f
    z = p(x, y)
    adam_list = [[x], [y], [z]]
    mx = [0]
    my = [0]
    vx = [0]
    vy = [0]
    upx = [0]
    upy = [0]
    eps = 10e-8
    t = 0

    while t in range(steps):  # noch andere bedingung?
        t += 1
        y_grad = p(x, y, dx=0, dy=1)
        x_grad = p(x, y, dx=1, dy=0)
        mx.append(beta1 * mx[t - 1] + (1 - beta1) * x_grad)
        my.append(beta1 * my[t - 1] + (1 - beta1) * y_grad)
        vx.append(beta1 * vx[t - 1] + (1 - beta1) * x_grad**2)
        vy.append(beta1 * vy[t - 1] + (1 - beta1) * y_grad**2)
        norm = alpha / math.sqrt(x_grad**2 + y_grad**2)
        upx.append(norm * mx[t] / math.sqrt(vx[t] + eps))
        upy.append(norm * my[t] / math.sqrt(vy[t] + eps))

        x = x - upx[t]
        y = y - upy[t]

        adam_list[0].append(x)
        adam_list[1].append(y)
        adam_list[2].append(f(x, y)[0])

        if (x >= xmax or x <= xmin) or (y >= ymax or y <= ymin):
            print("Punkt nich in Fläche duh, Iterations: ", t)
            break

        print(t, "von", steps)

    return adam_list


def Plots(x, y, f, p1, p2):
    """True surface, interpolated surface and vectorfield of the gradients.

    Args:
        p1, p2: x,y-coordinates for grid-limits

    Returns:
        ax, ax2, ax3: plot objects (?)
    """
    X, Y, Z = surface(p1, p2, 50)
    A, B, C = surface(p1, p2, 10)
    ZB = RectBivariateSpline(p1, p2, 10)
    ZP = p(X, Y)

    sur = plt.figure()
    arr = plt.figure()

    xmesh, ymesh = np.meshgrid(X, Y)
    amesh, bmesh = np.meshgrid(A, B)

    ax = sur.add_subplot(131, projection="3d")
    ax.plot_surface(xmesh, ymesh, Z, cmap=cm.cividis, antialiased=True)
    ax2 = sur.add_subplot(132, projection="3d")
    ax2.plot_surface(xmesh, ymesh, ZP, cmap=cm.PiYG, antialiased=True, alpha=0.8)
    axb = sur.add_subplot(133, projection="3d")
    axb.plot_surface(xmesh, ymesh, ZB, cmap=cm.cividis, antialiased=True, alpha=0.8)

    opt = so(x, y, p, 0.0015, 1_000, 0.6)  # sehr gute val=0.000015
    # step = 10_000
    # ex = opt[0][step]
    # ey = opt[1][step]
    # ez = opt[2][step]

    ax2.scatter(opt[0], opt[1], opt[2], linewidth=1, c=opt[2], cmap=cm.hsv, s=0.89)
    # ax.plot3D(opt2[0], opt2[1], opt2[2], linewidth=2, color="slateblue", label="MPI")
    X, Y, Z = surface(p1, p2, 50)
    f = interpolate.interp2d(X, Y, Z, kind="cubic")
    ax2.scatter(xz, yz, f(xz, yz), color="magenta", s=0.8)
    # ax2.scatter(ex, ey, ez, color="magenta", s=7)

    gx = p(XP, YP, dx=1, dy=0)
    gy = p(XP, YP, dx=0, dy=1)

    ax3 = arr.add_subplot()
    bmesh_in = np.array(bmesh)
    bmesh_in = bmesh_in[np.argsort(bmesh_in[:, 0])]
    ax3.streamplot(
        amesh,
        bmesh_in,
        -gx,
        -gy,
        density=2,
        linewidth=C * 0.01,
        arrowstyle="fancy",
        color=C,
        cmap="plasma",
    )

    return ax, ax2, ax3


# ax, ax2, ax3 = Plots(
#     xz,
#     yz,
#     f,
#     (51.37851084786834, 12.281719132563873),
#     (51.30740546223145, 12.437053610317154),
# )


# plt.show()

# class GD:

#     def init(px, py, steps):
#         opt = GD([X, Y])
#         x, y = px, py
#         res = [[x], [y]]
#         for i in range(steps):
#             gradx = p(x, y, dx=1, dy=0)
#             grady = p(x, y, dx=0, dy=1)
#             stepx, stepy = opt.step(gradx, grady)
#             x += stepx
#             y += stepy
#             res[0].append(x) #--> p(x, y)?
#             res[1].append(y)
#     # --> returns gradient!? as grad

#     def steps(grad):
#         return steps, params


def DPlots(x, y, f, p1, p2):
    """Different trajectories."""
    p = f
    X, Y, Z = surface(p1, p2, 50)
    ZP = p(X, Y)

    sur = plt.figure()

    xmesh, ymesh = np.meshgrid(X, Y)

    opt1 = so(x, y, p, 0.0015, 10_000, 0.6)
    opt2 = so(x, y, p, 0.0015, 10_000, 0.0)  # kein Momentum
    opt3 = so(x, y, p, 0.0015, 10_000, 0.5)
    opt4 = so(x, y, p, 0.0015, 10_000, 0.8)
    opt5 = so(x, y, p, 0.0015, 10_000, 0.35)
    opt6 = adam(x, y, p, 10_000, 0.00015, 0.4, 0.5)

    ax = sur.add_subplot(231)
    ax.set_title("Alpha=0.0015, Momentum=0.6")
    ax.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax.scatter(opt1[0], opt1[1], c=opt1[2], cmap=cm.binary, s=1.2)

    ax2 = sur.add_subplot(232)
    ax2.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax2.scatter(opt2[0], opt2[1], c="pink", s=1.2)

    ax3 = sur.add_subplot(233)
    ax3.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax3.scatter(opt3[0], opt3[1], c="blue", s=1.2)

    ax4 = sur.add_subplot(234)
    ax4.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax4.scatter(opt4[0], opt4[1], c="lime", s=1.2)

    ax5 = sur.add_subplot(235)
    ax5.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax5.scatter(opt5[0], opt5[1], c="yellow", s=1.2)

    ax6 = sur.add_subplot(236)
    ax6.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax6.scatter(opt6[0], opt6[1], c="red", s=1.2)

    return ax, ax2, ax3, ax4, ax5, ax6


ax1, ax2, ax3, ax4, ax5, ax6 = DPlots(
    xz,
    yz,
    p,
    (51.37851084786834, 12.281719132563873),
    (51.30740546223145, 12.437053610317154),
)

plt.show()
