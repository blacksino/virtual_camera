import pypbd
import tetgen
import pyvista as pv

stl_file_path = '/home/SENSETIME/xulixin2/Downloads/001713343/liver.stl'

pc = pv.read(stl_file_path)
loader = pypbd.TetGenLoader()

tet = tetgen.TetGen(pc)
tet.tetrahedralize()

grid = tet.grid
cells = grid.cells.reshape(-1, 5)[:, 1:]
cell_center = grid.points[cells].mean(1)

# extract cells below the 0 xy plane
mask = cell_center[:, 2] < cell_center[:,2].mean()
cell_ind = mask.nonzero()[0]
subgrid = grid.extract_cells(cell_ind)

# advanced plotting
plotter = pv.Plotter()
plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True)
plotter.add_mesh(pc, 'r', 'wireframe')
plotter.add_legend([[' Input Mesh ', 'r'],
                    [' Tessellated Mesh ', 'black']])
# tet.write()
tet_loader = pypbd.TetGenLoader()
# tet_loader.loadTetFile()

plotter.show()