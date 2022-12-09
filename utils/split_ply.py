import meshio
import numpy as np
from tqdm import tqdm

mesh = meshio.read('/home/SENSETIME/xulixin2/下载/seg.ply')

red = mesh.cell_data['red'][0]
green = mesh.cell_data['green'][0]
blue = mesh.cell_data['blue'][0]

attr = np.stack([red, green, blue], axis=1)


cells = mesh.cells[0].data
cells = cells.tolist()
# group cells by it's attribute
grouped_cells = {}
for i in tqdm(range(len(cells))):
    key = tuple(attr[i])
    if key not in grouped_cells:
        grouped_cells[key] = []
    grouped_cells[key].append(cells[i])

for each_key,value in grouped_cells.items():
    print(each_key)
    new_mesh = meshio.Mesh(points=mesh.points, cells=[('triangle', np.array(value))])
    meshio.write(f'/home/SENSETIME/xulixin2/下载/seg_{each_key}.ply', new_mesh, file_format='ply')