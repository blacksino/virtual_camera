import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
"""
本段代码假设虚拟相机principle point处于图像中心，且两方向焦距相同。

"""

def projection_scene_points_into_frame(scene_points, current_extrinsics, intrinsics, distort=None):
    """
    :param scene_points:
    :param current_extrinsics:
    :param intrinsics:
    :param distort:
    :return:
    """
    N , _ = scene_points.shape
    h,w = intrinsics[0,-1]*2,intrinsics[1,-1]*2
    ones = np.ones((1,N))
    scene_points = scene_points.T
    scene_points = np.r_[scene_points,ones]
    intrinsics = np.c_[intrinsics,np.zeros((3,1))]
    intrinsics[-1,-2] = 1

    camera_scene_points = np.linalg.inv(current_extrinsics) @ scene_points
    depth_in_camera_ = camera_scene_points[2,:]

    frame_pixel_points = (intrinsics @ camera_scene_points)
    frame_pixel_points /= depth_in_camera_

    frame_pixel_points[0,:] = w - frame_pixel_points[0,:]
    frame_pixel_points[1,:] = h = frame_pixel_points[1,:]
    """
    注意此时图像坐标满足vtk image的坐标系。
    """
    return frame_pixel_points

def PnP_reprojection(scene_points,):
    """
    如果要使用opencv解决PnP问题，坐标系应该转换至opencv下
    """
    raise NotImplemented



def find_points_pair_correspondence(projected_scene_points,contour_points):
    """
    我们假设contour点集存在N个点，场景标记点存在M个点，则必定有M>=N.
    在忽略透视的情况下，我们首先将contour_point进行离散化。
    projected scene points 为Set target
    contour points 为Set reference
    """
    num_of_target = len(projected_scene_points)
    num_of_contour = len(contour_points)
