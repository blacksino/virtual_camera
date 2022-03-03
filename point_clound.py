import open3d as o3d
import copy
import numpy as np
pc = o3d.io.read_point_cloud('/data/endoscope/simulation_data/contour.ply')
projected_pc = o3d.io.read_point_cloud('/data/endoscope/simulation_data/label_contour_reprojected.ply')

pseudo_3d = False

if pseudo_3d:
    z_points = np.asarray(copy.deepcopy(pc.points))
    z_points.std(axis=0)
    z_points[:,-1] = np.random.normal(loc=0.1,scale=0.001,size=z_points.shape[0])
    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.open3d.utility.Vector3dVector(z_points)
else:
    new_pc = copy.deepcopy(pc)

def cloud2image(extrinsics,intrinsics,point_cloud_path="/data/endoscope/simulation_data/contour.ply"):
    pc = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pc.points)
    points = np.c_[points,np.ones(points.shape[0])]
    new_points = np.linalg.inv(extrinsics) @ points.T
    new_points = new_points[:3,:]
    new_points /= new_points[:,-1]
    return new_points
    projected_new_points = intrinsics @ new_points
    return projected_new_points

# R = new_pc.get_rotation_matrix_from_xyz((0,0,np.pi/18))
# new_pc.rotate(R)
# print(R)
# new_pc.translate((0.0002, 0.001, 0.5))

# pc_r.scale(1.1)
# o3d.io.write_point_cloud('/home/SENSETIME/xulixin2/code/multimodal-registration-master/new_gradient-hystheresis-image.ply',new_pc,write_ascii=True)
o3d.visualization.draw_geometries([pc])



