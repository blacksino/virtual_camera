from my_camera import *


if __name__ == '__main__':
    fx = 9.9640e+02
    fy = 9.9640e+02
    w = 720
    h = 576

    cx = 375
    cy = 240

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
    Camera_VTK(w, h, K, manual_align_vtk_path, data_root_path,background_path=
               '/home/SENSETIME/xulixin2/图片/endo_label.png',
               marker_path="/data/endoscope/simulation_data/14:50:17/registration.json")



