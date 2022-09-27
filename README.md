# Optimization trajectories in a landscape in Germany.

Example: Tübingen

Coordinates:
y, x = (48.53740847396933, 9.05109407405155)
p1 = (48.53779261272054, 9.018199965543145)
p2 = (48.50026378163786, 9.100493830179525)

1. 3D interpolated surface between the chosen coordinates (p1: north/west, p2: south/east) and ADAM/Momentum-Optimization trajectory, starting from point yz, xz. 
    - PyVista-Visualization: https://github.com/NurEinJemand/tueopt/Figures/PyVistaSurface_ExPlot.png
    - Matplotlib-Visualization: https://github.com/NurEinJemand/tueopt/Figures/MatplotlibSurface_ExPlot.png
2. 2D Streamplot of the first-order-derivatives (w.r.t. x and y). Plotted with Matplotlib.
    - https://github.com/NurEinJemand/tueopt/Figures/Streamplot_ExPlot.png
3. 2D Loss-over-iteration-Plot for ADAM/Momentum.
    - https://github.com/NurEinJemand/tueopt/Figures/LossOverIteration_ExPlot.png
4. 2D Plots of Optimizer trajectories with different parameters.
    - https://github.com/NurEinJemand/tueopt/Figures/2DTrajectories_ExPlot.png




