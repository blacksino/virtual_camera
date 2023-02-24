import trimesh
import meshio
import numpy as np
from tqdm import tqdm

vtk_mesh = meshio.vtk.read('/home/SENSETIME/xulixin2/RJ_demo/mesh/all.vtk')

points = vtk_mesh.points
cells = vtk_mesh.cells[0].data
label_data = vtk_mesh.cell_data['MeshDomain'][0]

label_value = set(label_data)
color_data = list(range(len(label_value)))
color_dict = dict(zip(label_value, color_data))

with open('/home/SENSETIME/xulixin2/RJ_demo/mesh/all.face', 'r') as f:
    lines = f.readlines()
    surface_faces = []
    for line in lines[1:]:
        # skip the first line
        surface_faces.append([int(x) for x in line.split()[1:4]])
    surface_faces = np.array(surface_faces)


face_dict = {}
for i, cell in enumerate(cells):
    cell_faces = []
    for j in range(4):
        for k in range(j + 1, 4):
            for l in range(k + 1, 4):
                face = tuple(sorted([cell[j], cell[k], cell[l]]))
                cell_faces.append(face)
                for face in cell_faces:
                    if face in face_dict: face_dict[face].append(i)
                    else: face_dict[face] = [i]

face_label = []
for each_face in tqdm(surface_faces, total=len(surface_faces)):
    face = tuple(sorted(each_face))
    if face not in face_dict.keys():
        print(f'face:{face} not in face_dict.keys()')
        continue
    tetra_index = face_dict[face]
    label = label_data[tetra_index]
    face_label.append(color_dict[min(label)])


with open('/home/SENSETIME/xulixin2/RJ_demo/mesh/fine_liver_labeled.face', 'a') as f:
    f.write(f'{len(surface_faces)} 0\n')
    for i, each_face in enumerate(surface_faces):
        f.write(f'{i} {each_face[0]} {each_face[1]} {each_face[2]} {int(face_label[i])}\n')