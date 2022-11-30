# ensure python version is 3.8 or higher
import sys
assert sys.version_info >= (3, 8)
import meshio
import numpy as np

vtk_file = meshio.read('/home/SENSETIME/xulixin2/code/MultiDomainMeshing/cmake-build-release/simple_liver.vtu')
meshio.vtk.write('/home/SENSETIME/xulixin2/code/MultiDomainMeshing/cmake-build-release/simple_liver.vtk',vtk_file,fmt_version='4.2',binary=False)
meshio.write()
#write tetgen node
meshio.write()