"""Surface, interpolation and streamplot of gradients between 2 points in Germany."""
import os
import shutil
import urllib.request
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import rasterio
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate

# Leipzig, Völkerschlachtdenkmal
# yz, xz = (51.314342229660475, 12.413751897373517)
# p1 = (51.37851084786834, 12.281719132563873)
# p2 = (51.30740546223145, 12.437053610317154)
yz, xz = (48.54113160303172, 9.057397243195163)
p1 = (48.544567833068285, 9.016150012642221)
p2 = (48.516836785951845, 9.128937904248804)


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
X, Y, Z = surface(p1, p2, f_steps)
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
        # self._change_x = [0.0]
        # self._change_y = [0.0]
        self._chx = 0
        self._chy = 0
        self._step_length = step_length
        self._momentum = momentum

    def step(self, grad_x, grad_y):
        """Perform one iteration of gradient descent.

        Args:
            grad_x: Gradient w.r.t. x.
            grad_y: Gradient w.r.t. y.
        """
        # grad dings siehe train func
        self._chx = self._momentum * self._chx + grad_x
        self._chy = self._momentum * self._chy + grad_y

        self._x = self._x - self._step_length * self._chx
        self._y = self._y - self._step_length * self._chy

        # last_change_x = self._change_x[-1]
        # self._change_x.append(
        #     self._step_length * grad_x + self._momentum * last_change_x
        # )
        # last_change_y = self._change_y[-1]
        # self._change_y.append(
        #     self._step_length * grad_y + self._momentum * last_change_y
        # )

        # self._x = self._x - self._change_x[-1]
        # self._y = self._y - self._change_y[-1]

    def get_x(self):
        """Return x variable."""
        return self._x

    def get_y(self):
        """Return y variable."""
        return self._y


class ADAM:
    """ADAM algorithm."""

    def __init__(self, x, y, step_length, beta1, beta2):
        """Set up the ADAM method.

        Args:
            x: First variable to optimize over.
            y: Second variable to optimize over.
            step_length: Length of the GD update (no momentum).
            beta1, beta2: good question.
        """
        self._x = x
        self._y = y
        self.sl = step_length
        self.beta1 = beta1
        self.beta2 = beta2
        self.mx, self.my, self.vx, self.vy = 0, 0, 0, 0

    def step(self, grad_x, grad_y):
        """Perform one iteration of ADAM.

        Args:
            grad_x: Gradient w.r.t. x.
            grad_y: Gradient w.r.t. y.
        """
        epsilon = 1e-09

        self.mx = self.beta1 * self.mx + (1 - self.beta1) * grad_x
        self.my = self.beta1 * self.my + (1 - self.beta1) * grad_y

        self.vx = self.beta2 * self.vx + (1 - self.beta2) * grad_x**2
        self.vy = self.beta2 * self.vy + (1 - self.beta2) * grad_y**2

        self._x = self._x - self.sl * self.mx / (np.sqrt(self.vx) + epsilon)
        self._y = self._y - self.sl * self.my / (np.sqrt(self.vy) + epsilon)

    def get_x(self):
        """Return x variable."""
        return self._x

    def get_y(self):
        """Return y variable."""
        return self._y


def train(optimizer, f, beta):
    """Train func.

    Args:
        optimizer: GD or ADAM
        f: function
        beta: iterations

    Returns:
        opt_liste: list with all steps.
    """
    p = f
    x = optimizer.get_x()
    y = optimizer.get_y()
    z = p(x, y)[0]
    opt_liste = [[x], [y], [z]]

    for i in range(beta):
        y_grad = p(x, y, dx=0, dy=1)
        x_grad = p(x, y, dx=1, dy=0)

        optimizer.step(x_grad, y_grad)

        x = optimizer.get_x()
        y = optimizer.get_y()

        opt_liste[0].append(x[0])
        opt_liste[1].append(y[0])
        opt_liste[2].append(p(x, y)[0] + 1)

        if (x >= xmax or x <= xmin) or (y >= ymax or y <= ymin):
            print("Punkt nich in Fläche duh, Iterations: ", i)
            break

        print(i, "von", beta)

    return opt_liste


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


def Plots(x, y, f, p1, p2, cmap_surface, cmap_contour, color1, color2):
    """True surface, interpolated surface and vectorfield of the gradients.

    Args:
        p1, p2: x,y-coordinates for grid-limits

    Returns:
        ax, ax2, ax3: plot objects (?)
    """
    X, Y, Z = surface(p1, p2, 50)
    A, B, C = surface(p1, p2, 10)
    ZP = p(X, Y)

    def register_uni_tue_cmap(name: str):
        """Create and register Uni Tuebingen colormap under the specified name."""
        TUred = [165 / 255, 30 / 255, 55 / 255]
        TUgold = [180 / 255, 160 / 255, 105 / 255]
        TUanthrazit = [50 / 255, 65 / 255, 75 / 255]  # sometimes referred to as TUdark
        colors = np.array([TUgold, TUred, TUanthrazit])

        cmap = LinearSegmentedColormap.from_list(name, colors)
        plt.register_cmap(name, cmap)

    register_uni_tue_cmap("uni_tue")

    sur = plt.figure()
    arr = plt.figure()

    xmesh, ymesh = np.meshgrid(X, Y)
    amesh, bmesh = np.meshgrid(A, B)

    # ax = sur.add_subplot(131, projection="3d")
    # ax.plot_surface(xmesh, ymesh, Z, cmap=cm.cividis, antialiased=True)

    ax2 = sur.add_subplot(projection="3d")
    # ax2 = sur.add_subplot(projection="3d")
    # ax2.set_axis_off()

    ax2.plot_surface(
        xmesh, ymesh, ZP, cmap=cmap_surface, antialiased=True, alpha=0.8, zorder=2.0
    )
    # axb = sur.add_subplot(133, projection="3d")
    # axb.plot_surface(xmesh, ymesh, ZB, cmap=cm.cividis, antialiased=True, alpha=0.8)

    opt = train(ADAM(x, y, 0.00002, 0.8, 0.99), p, 10_000)
    opt2 = train(GD(x, y, 0.000000009, 0.9), p, 10_000)
    dx, dy, dz = opt2[0], opt2[1], opt2[2]

    ax2.plot(opt[0], opt[1], opt[2], linewidth=2, color=color1, alpha=1, zorder=1.0)
    ax2.plot(dx, dy, dz, linewidth=2, color=color2)
    ax2.contourf(X, Y, Z, zdir="z", offset=270, cmap=cmap_contour, alpha=0.67)

    # ax.plot3D(opt2[0], opt2[1], opt2[2], linewidth=2, color="slateblue", label="MPI")

    gx = p(XP, YP, dx=1, dy=0)
    gy = p(XP, YP, dx=0, dy=1)

    ax3 = arr.add_subplot()
    ax3.set_axis_off()
    bmesh_in = np.array(bmesh)
    bmesh_in = bmesh_in[np.argsort(bmesh_in[:, 0])]
    ax3.streamplot(
        amesh,
        bmesh_in,
        -gx,
        -gy,
        density=2,
        linewidth=C * 0.003,
        arrowstyle="fancy",
        color=C,
        cmap=cmap_contour,
    )

    return ax2, ax3


# ax2, ax3 = Plots(
#     xz, yz, f, p1, p2, "Greys", "uni_tue", (180 / 255, 160 / 255, 105 / 255), "darkred"
# )

# ax3, ax4 = Plots(xz, yz, f, p1, p2, "Greys", "magma", "purple", "darkorange")


def DPlots(x, y, f, p1, p2):
    """Different trajectories."""
    p = f
    X, Y, Z = surface(p1, p2, 50)
    ZP = p(X, Y)

    sur = plt.figure()

    xmesh, ymesh = np.meshgrid(X, Y)

    opt1 = train(GD(x, y, 0.0000002, 0.0), p, 10_000)
    opt2 = train(ADAM(x, y, 0.00002, 0.9, 0.9), p, 10_000)
    opt3 = train(ADAM(x, y, 0.00003, 0.9, 0.8), p, 10_000)
    opt4 = train(ADAM(x, y, 0.00004, 0.9, 0.3), p, 10_000)
    opt5 = train(ADAM(x, y, 0.00005, 0.4, 0.9), p, 10_000)
    opt6 = train(GD(x, y, 0.0000002, 0.9), p, 10_000)

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


# ax1, ax2, ax3, ax4, ax5, ax6 = DPlots(
#     xz,
#     yz,
#     p,
#     (51.37851084786834, 12.281719132563873),
#     (51.30740546223145, 12.437053610317154),
# )

# plt.show()


# PyVista
xmesh, ymesh = np.meshgrid(X, Y)
Z = p(X, Y)
grid = pv.StructuredGrid(xmesh, ymesh, Z)


def register_uni_tue_cmap(name: str):
    """Create and register Uni Tuebingen colormap under the specified name."""
    TUred = [165 / 255, 30 / 255, 55 / 255]
    TUgold = [180 / 255, 160 / 255, 105 / 255]
    TUanthrazit = [50 / 255, 65 / 255, 75 / 255]  # sometimes referred to as TUdark
    colors = np.array([TUgold, TUred, TUanthrazit])

    cmap = LinearSegmentedColormap.from_list(name, colors)
    plt.register_cmap(name, cmap)


register_uni_tue_cmap("uni_tue")


plotter = pv.Plotter()
plotter.add_mesh(
    grid,
    scalars=grid.points[:, -1],
    cmap="uni_tue",
    lighting=True,
    opacity=0.8,
    ambient=0.4,
    roughness=0.9,
)

opt = train(ADAM(xz, yz, 0.0002, 0.8, 0.99), p, 10_000)

points = list(zip(opt[0], opt[1], opt[2]))


def polyline_from_points(points):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly


polyline = polyline_from_points(points)
polyline["scalars"] = np.arange(polyline.n_points)
tube = polyline.tube(radius=0.0003)
# tube.plot(smooth_shading=True)

plotter.add_mesh(tube)

plotter.show_grid()
plotter.set_scale(
    xscale=2, yscale=xmesh.ptp() / ymesh.ptp(), zscale=xmesh.ptp() / Z.ptp()
)

plotter.set_background("beige")

print()
plotter.show()
