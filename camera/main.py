from my_camera import *
from glob import glob
import meshio
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    fx = 1736
    fy = 1736
    cx = 941
    cy = 601


    w = 1920
    h = 1080


    K = np.array([[fx, 0., cx],
                  [0., fy, cy],
                  [0., 0., 1.]])

    stl_path = '../all.stl'
    full_stl_path = '/home/SENSETIME/xulixin2/Downloads/001713343/liver.stl'
    simple_stl_path = '/home/SENSETIME/xulixin2/Downloads/001713343/simple_liver.stl'
    simple_tet_path = '/data/tetgen_test/simple_liver.1.vtk'
    complex_tet_path = '/home/SENSETIME/xulixin2/code/MultiDomainMeshing/cmake-build-release/all_liver.vtu'
    # deformed_tet_path = '/home/SENSETIME/xulixin2/deformed_mesh.vtk'
    # deformed_tet_path = '/home/SENSETIME/xulixin2/large_deformed.vtk'
    deformed_tet_path = '/home/SENSETIME/xulixin2/code/SofaScene/example.vtk0.vtu'
    data_root_path = '/data/endoscope/simulation_data'
    # stl_poly_data = loadSTL(simple_stl_path)
    manual_align_vtk_path = '/home/SENSETIME/xulixin2/下载/SERV-CT-ALL/CT/008/Anatomy.vtk'


    mesh_list = glob(f'/home/SENSETIME/xulixin2/下载/seg_*')
    Camera_VTK(w, h, K, mesh_path="/home/SENSETIME/xulixin2/RJ_demo/mesh/deformed.vtk",
               data_root_path=data_root_path,
               background_path=None,
               video_path=None,
               read_tet=True)




