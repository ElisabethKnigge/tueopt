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
            xy_idx.append((x_idx, y_idx))
            coords.append((x_coord, y_coord))

    for (x_idx, y_idx), elevation in zip(xy_idx, elevations(src_file, *coords)):
        Z[y_idx, x_idx] = elevation

    return X, Y, Z


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
        self._chx = self._momentum * self._chx + grad_x
        self._chy = self._momentum * self._chy + grad_y

        self._x = self._x - self._step_length * self._chx
        self._y = self._y - self._step_length * self._chy

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
            beta1: good.
            beta2: question.
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
        epsilon = 1e-04

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


def register_uni_tue_cmap(name: str):
    """Create and register Uni Tuebingen colormap under the specified name."""
    TUred = [165 / 255, 30 / 255, 55 / 255]
    TUgold = [180 / 255, 160 / 255, 105 / 255]
    TUanthrazit = [50 / 255, 65 / 255, 75 / 255]  # sometimes referred to as TUdark
    colors = np.array([TUgold, TUred, TUanthrazit])

    cmap = LinearSegmentedColormap.from_list(name, colors)
    plt.register_cmap(name, cmap)


def Plots(
    p,
    XP,
    YP,
    ZP,
    p1,
    p2,
    xz,
    yz,
    cmap_surface,
    cmap_contour,
    color1,
    color2,
    adam_its,
    gd_its,
    lr_adam,
    lr_gd,
):
    """True surface, interpolated surface and vectorfield of the gradients.

    Args:
        p1, p2: x,y-coordinates for grid-limits

    Returns:
        ax, ax_surface, ax_streamplot: plot objects (?)
    """
    X, Y, Z = surface(p1, p2, 50)
    A, B, C = surface(p1, p2, 10)
    ZP = p(X, Y)

    sur = plt.figure()
    arr = plt.figure()
    loss = plt.figure()

    xmesh, ymesh = np.meshgrid(X, Y)
    amesh, bmesh = np.meshgrid(A, B)

    ax_surface = sur.add_subplot(projection="3d")
    ax_surface.set_axis_off()

    ax_surface.plot_surface(
        xmesh, ymesh, ZP, cmap=cmap_surface, antialiased=True, alpha=0.8, zorder=2.0
    )

    opt = train(ADAM(xz, yz, lr_adam, 0.8, 0.99), p, adam_its)
    opt2 = train(GD(xz, yz, lr_gd, 0.9), p, gd_its)
    dx, dy, dz = opt2[0], opt2[1], opt2[2]

    ax_loi = loss.add_subplot(121, title="ADAM")
    ax_loi2 = loss.add_subplot(122, title="GD with momentum")
    ax_loi.plot(opt[2], c=color1)
    ax_loi2.plot(dz, c=color2)

    ax_surface.plot(
        opt[0], opt[1], opt[2], linewidth=2, color=color1, alpha=1, zorder=1.0
    )
    ax_surface.plot(dx, dy, dz, linewidth=2, color=color2)
    ax_surface.contourf(
        X, Y, Z, zdir="z", offset=np.min(Z) - 15, cmap=cmap_contour, alpha=0.67
    )

    gx = p(XP, YP, dx=1, dy=0)
    gy = p(XP, YP, dx=0, dy=1)

    ax_streamplot = arr.add_subplot()
    ax_streamplot.set_xlim(p1[1] - 0.005, p2[1] + 0.005)
    ax_streamplot.set_ylim(p2[0] - 0.005, p1[0] + 0.005)
    bmesh_in = np.array(bmesh)
    bmesh_in = bmesh_in[np.argsort(bmesh_in[:, 0])]
    ax_streamplot.streamplot(
        amesh,
        bmesh_in,
        -gx,
        -gy,
        density=2.3,
        linewidth=C * 0.004,
        arrowstyle="fancy",
        color=C,
        cmap=cmap_contour,
    )

    return ax_surface, ax_streamplot, ax_loi, ax_loi2


def DPlots(p, x, y, p1, p2, gd_its, adam_its):
    """Different trajectories."""
    X, Y, Z = surface(p1, p2, 50)
    ZP = p(X, Y)

    sur = plt.figure()

    xmesh, ymesh = np.meshgrid(X, Y)

    opt1 = train(GD(x, y, 0.00000002, 0.9), p, gd_its)
    opt2 = train(GD(x, y, 0.0000002, 0), p, gd_its)
    opt3 = train(GD(x, y, 0.0000003, 0.5), p, gd_its)
    opt4 = train(ADAM(x, y, 0.00004, 0.9, 0.3), p, adam_its)
    opt5 = train(ADAM(x, y, 0.00005, 0.4, 0.9), p, adam_its)
    opt6 = train(ADAM(x, y, 0.00002, 0.9, 0.6), p, adam_its)

    ax = sur.add_subplot(231)
    ax.tick_params(labelsize=7)
    ax.set_title(label=("Momentum", "Iterations: ", gd_its, "lr: ", 0.0000002), size=7)
    ax.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax.scatter(opt1[0], opt1[1], c=opt1[2], cmap=cm.binary, s=1.2)

    ax2 = sur.add_subplot(232)
    ax2.tick_params(labelsize=7)
    ax2.set_title(label=("Momentum", "Iterations: ", gd_its, "lr: ", 0.0000002), size=7)
    ax2.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax2.scatter(opt2[0], opt2[1], c="pink", s=1.2)

    ax3 = sur.add_subplot(233)
    ax3.tick_params(labelsize=7)
    ax3.set_title(label=("Momentum", "Iterations: ", gd_its, "lr: ", 0.0000002), size=7)
    ax3.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax3.scatter(opt3[0], opt3[1], c="blue", s=1.2)

    ax4 = sur.add_subplot(234)
    ax4.tick_params(labelsize=7)
    ax4.set_title(label=("ADAM", "Iterations: ", adam_its, "lr: ", 0.0000002), size=7)
    ax4.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax4.scatter(opt4[0], opt4[1], c="lime", s=1.2)

    ax5 = sur.add_subplot(235)
    ax5.tick_params(labelsize=7)
    ax5.set_title(label=("ADAM", "Iterations: ", adam_its, "lr: ", 0.0000002), size=7)
    ax5.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax5.scatter(opt5[0], opt5[1], c="yellow", s=1.2)

    ax6 = sur.add_subplot(236)
    ax6.tick_params(labelsize=7)
    ax6.set_title(label=("ADAM", "Iterations: ", adam_its, "lr: ", 0.0000002), size=7)
    ax6.contourf(xmesh, ymesh, ZP, levels=70, cmap=cm.viridis)
    ax6.scatter(opt6[0], opt6[1], c="red", s=1.2)

    return ax, ax2, ax3, ax4, ax5, ax6


def VisPlots(
    p,
    xz,
    yz,
    X,
    Y,
    cmap_surface,
    cmap_contour,
    color1,
    color2,
    adam_its,
    gd_its,
    lr_adam,
    lr_gd,
):
    """Plotting 3D-Surface with PyVista."""
    xmesh, ymesh = np.meshgrid(X, Y)
    Z = p(X, Y)
    grid = pv.StructuredGrid(xmesh, ymesh, Z)

    plotter = pv.Plotter(
        polygon_smoothing=True,
    )
    plotter.add_mesh(
        grid,
        scalars=grid.points[:, -1],
        cmap=cmap_surface,
        opacity=0.9,
        ambient=0.4,
        roughness=0.9,
        show_scalar_bar=False,
        flip_scalars=True,
        # show_edges=True,
        edge_color="lightgrey",
    )

    poly = grid.extract_surface()
    origin = poly.center
    origin[-1] -= poly.length / 2
    projected = poly.project_points_to_plane(origin=origin)

    plotter.add_mesh(projected, cmap=cmap_contour, show_scalar_bar=False, opacity=0.8)

    contours = grid.contour()
    plotter.add_mesh(
        contours,
        color="white",
        line_width=1.5,
        lighting=False,
        interpolate_before_map=True,
        smooth_shading=1,
    )

    opt = train(ADAM(xz, yz, lr_adam, 0.7, 0.99), p, adam_its)
    opt2 = train(GD(xz, yz, lr_gd, 0.6), p, gd_its)

    points = list(zip(opt[0], opt[1], opt[2]))
    points2 = list(zip(opt2[0], opt2[1], opt2[2]))

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

    polyline2 = polyline_from_points(points2)
    polyline2["scalars"] = np.arange(polyline2.n_points)
    tube2 = polyline2.tube(radius=0.0003)

    plotter.add_mesh(tube, color=color1)
    plotter.add_mesh(tube2, color=color2)

    plotter.set_scale(
        xscale=1.3,
        yscale=xmesh.ptp() / ymesh.ptp(),
        zscale=ymesh.ptp() / Z.ptp(),
    )

    plotter.set_background("white")

    return plotter


register_uni_tue_cmap("uni_tue")


class Plot:
    """Plot a simple example."""

    def __init__(self, x, y, p1, p2):
        """Set up.

        Args:
            x: Initial x-coordinate for Optimizer.
            y: Initial y-coordinate for Optimizer.
            p1: Border coordinates (north/west)
            p2: Border coordinates (south/east).
        """
        self.x = x
        self.y = y
        self.p1 = p1
        self.p2 = p2
        f_steps = 50
        X, Y, Z = surface(self.p1, self.p2, f_steps)
        f = interpolate.interp2d(X, Y, Z, kind="cubic")
        self.X, self.Y, self.Z = X, Y, Z
        self.f = f
        p_steps = 10
        XP, YP, ZP = surface(p1, p2, p_steps)
        p = interpolate.interp2d(XP, YP, ZP, kind="cubic")
        self.XP, self.YP, self.ZP = XP, YP, ZP
        self.p = p

    def PMatplotlib(
        self,
        surface_color="Greys",
        contour_color="magma",
        adam_color="pink",
        gd_color="darkorange",
        adam_its=500,
        gd_its=500,
        lr_adam=0.0003,
        lr_gd=0.0000004,
    ):
        """Visualization via Matplotlib."""
        ax_surface, ax_streamplot, ax_loi, ax_loi2 = Plots(
            self.p,
            self.XP,
            self.YP,
            self.ZP,
            self.p1,
            self.p2,
            self.x,
            self.y,
            surface_color,
            contour_color,
            adam_color,
            gd_color,
            adam_its,
            gd_its,
            lr_adam,
            lr_gd,
        )

        plt.show()

    def PPyvista(
        self,
        surface_color="Greys",
        contour_color="magma",
        adam_color="darkorange",
        gd_color="hotpink",
        adam_its=1000,
        gd_its=500,
        lr_adam=0.0002,
        lr_gd=0.000000002,
    ):
        """Visualization via PyVista."""
        plotter = VisPlots(
            self.p,
            self.x,
            self.y,
            self.X,
            self.Y,
            surface_color,
            contour_color,
            adam_color,
            gd_color,
            adam_its,
            gd_its,
            lr_adam,
            lr_gd,
        )

        plotter.show()

    def Trajectories(self, adam_its=1000, gd_its=500):
        """2D Trajectories.

        Args:
            adam_its: Iterations of ADAM.
            gd_its: Iterations of Momentum.
        """
        ax1, ax2, ax3, ax4, ax5, ax6 = DPlots(
            self.p, self.x, self.y, self.p1, self.p2, gd_its, adam_its
        )
        plt.show()
