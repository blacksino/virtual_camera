import trimesh
import meshio
import numpy as np
vtk_mesh = meshio.vtk.read('/home/SENSETIME/xulixin2/deformed_mesh_ascii_mdm.vtk')
# surface_faces = meshio.read('/home/SENSETIME/xulixin2/deformed_simple_liver.face')

points = vtk_mesh.points
cells = vtk_mesh.cells[0].data
label_data = vtk_mesh.cell_data['MeshDomain'][0]

with open('/home/SENSETIME/xulixin2/deformed_simple_liver.face', 'r') as f:
    lines = f.readlines()
    surface_faces = []
    for line in lines[1:]:
        # skip the first line
        surface_faces.append([int(x) for x in line.split()[1:]])
    surface_faces = np.array(surface_faces)



face_label = []
for each_face in surface_faces:
    # find out which tetra cell contains this face
    tetra_index = []
    for i,each_cell in enumerate(cells):
        face_set = set(each_face)
        cell_set = set(each_cell)
        if face_set.issubset(cell_set):
            tetra_index.append(i)
    #determine the label of this face
    label = label_data[tetra_index]
    face_label.append(min(label))

# add face_label to face file at last column
with open('/home/SENSETIME/xulixin2/deformed_simple_liver_w_label.face', 'a') as f:
    f.write(f'{len(surface_faces)} 0\n')
    for i,each_face in enumerate(surface_faces):
        f.write(f'{i} {each_face[0]} {each_face[1]} {each_face[2]} {face_label[i]}\n')

