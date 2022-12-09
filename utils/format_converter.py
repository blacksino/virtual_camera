# ensure python version is 3.8 or higher
import sys
# assert sys.version_info >= (3, 8)
import meshio
import numpy as np

vtk_file = meshio.read('/home/SENSETIME/xulixin2/图片/ct.vtk')
meshio.write('/home/SENSETIME/xulixin2/图片/ct.node', vtk_file.points, file_format='tetgen')

cells = vtk_file.cells[0].data
with open('/home/SENSETIME/xulixin2/图片/ct.face', 'w') as f:
    pass