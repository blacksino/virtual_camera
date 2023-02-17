# ensure python version is 3.8 or higher
import sys
# assert sys.version_info >= (3, 8)
import meshio
import numpy as np

# convert vtu to vtk
def vtu2vtk(vtu_path, vtk_path):
    mesh = meshio.read(vtu_path)
    # rotate mesh along x axis by i degree ,y axis by j degree, z axis by k degree
    i,j,k = 0,0,0
    theta = np.radians(i)
    c, s = np.cos(theta), np.sin(theta)
    R_x = np.array(((1, 0, 0),
                    (0, c, -s),
                    (0, s, c)))
    theta = np.radians(j)
    c, s = np.cos(theta), np.sin(theta)
    R_y = np.array(((c, 0, s),
                    (0, 1, 0),
                    (-s, 0, c)))
    theta = np.radians(k)
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array(((c, -s, 0),
                    (s, c, 0),
                    (0, 0, 1)))
    R = R_z @ R_y @ R_x
    mesh.points = mesh.points @ R
    # translate mesh to its center
    mesh.points = mesh.points - np.mean(mesh.points, axis=0)
    meshio.write(vtk_path, mesh, file_format='vtk')


vtu2vtk("/home/SENSETIME/xulixin2/RJ_demo/mesh/fine_liver.vtu", "/home/SENSETIME/xulixin2/RJ_demo/mesh/fine_liver.vtk")