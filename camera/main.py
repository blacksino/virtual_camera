from my_camera import *


if __name__ == '__main__':
    fx = 500.0
    fy = 500.0

    w = 1280
    h = 720

    cx = w/2
    cy = h/2

    K = np.array([[fx, 0., cx],
                  [0., fy, cy],
                  [0., 0., 1.]])

    stl_path = '../all.stl'
    full_stl_path = '/home/SENSETIME/xulixin2/Downloads/001713343/liver.stl'
    simple_stl_path = '/home/SENSETIME/xulixin2/Downloads/001713343/simple_liver.stl'
    simple_tet_path = '/data/tetgen_test/simple_liver.1.vtk'
    data_root_path = '/data/endoscope/simulation_data'
    # stl_poly_data = loadSTL(simple_stl_path)
    Camera_VTK(w, h, K, simple_tet_path, data_root_path)

