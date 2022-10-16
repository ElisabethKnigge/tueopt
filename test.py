"""Run the demo."""
from tueopt import SimpleScript

yz, xz = (48.53868348072251, 9.056415391427574)
p1 = (48.552829738956845, 8.986801421938567)
p2 = (48.50787067087853, 9.111856574087847)

SimpleScript.Plot(xz, yz, p1, p2).PMatplotlib()
SimpleScript.Plot(xz, yz, p1, p2).PPyvista(adam_its=2000)
