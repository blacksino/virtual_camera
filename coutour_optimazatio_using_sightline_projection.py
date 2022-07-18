import cv2
import sklearn
from sklearn.neighbors import NearestNeighbors
import skimage
import numpy as np


def calculate_light_of_sight(contour_points, intrinsics):
    projection_matrix_list = list()
    for each_points in contour_points:
        current_sight_line = np.linalg.inv(intrinsics) @ each_points
        orthogonal_projection_matrix = current_sight_line @ current_sight_line.T
        projection_matrix_list.append(orthogonal_projection_matrix)
    return projection_matrix_list


def find_closest_point(projected_scenes_points, target_contour_points):
    nbrs = NearestNeighbors(n_neighbors=1).fit(target_contour_points)
    distance, index = nbrs.kneighbors(projected_scenes_points)
    """
    对于scene_points经过投影到视线后，再次投影到二维平面后，找到最近邻点后返回其下标。
    """
    return index


def project_scene_points_onto_sight_of_light(scene_points, projection_matrix_list, matching_index):
    projected_scene_points = list()
    for i, each_scene_points in enumerate(scene_points):
        index = matching_index[i]
        projected_scene_point = projection_matrix_list[index] @ each_scene_points
        projected_scene_points.append(projected_scene_point)
    return projected_scene_points


def determine_extrinsics_(scene_points, projected_scene_points):
    cv2.SVDecomp()
