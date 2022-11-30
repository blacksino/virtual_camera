import numpy as np
import matplotlib.pyplot as plt
import json
import os
import scipy
import vtk
import sklearn
import sklearn.decomposition
import cv2

np.set_printoptions(suppress=True)

def project_world_to_pixel(project_matrix_from_vtk_camera, h, w, world_points):
    scale_matrix = np.array([[w / 2, 0, 0, w / 2], [0, h / 2, 0, h / 2], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert world_points.shape[0] == 3
    # convert world point in homogeneous coordinate
    world_points = np.concatenate((world_points, np.ones((1, world_points.shape[1]))), axis=0)
    # project world point to pixel
    projection_matrix = np.dot(scale_matrix, project_matrix_from_vtk_camera)
    pixel_points = np.dot(projection_matrix, world_points)
    pixel_points[:3, :] /= pixel_points[3, :]
    return pixel_points[:2, :]


def get_main_direction_of_normals(normals):
    mean, std = np.mean(normals, axis=0), np.std(normals, axis=0)
    # remove outliers
    mask = np.abs(normals - mean) < std * 3
    # select out rows with all true.
    row_index = mask.all(axis=1)
    normals = normals[row_index, :]
    # get main direction of normals
    main_direction = np.mean(normals, axis=0)
    return main_direction


def get_center_of_mass(aabb_box):
    box_center = np.array([(aabb_box[1] + aabb_box[0]) / 2, (aabb_box[3] + aabb_box[2]) / 2, (aabb_box[5] + aabb_box[4]) / 2])
    return box_center

def get_aabb_of_vtk_polydata(vtk_polydata):
    """
    # get aabb of vtk_polydata
    # input: vtk_polydata: vtkPolyData
    # output: aabb: np.array
    """
    aabb = np.array(vtk_polydata.GetBounds())
    # bounding box: [xmin, xmax, ymin, ymax, zmin, zmax]
    return aabb


def determine_znear_zfar(aabb_box,project_matrix_from_vtk_camera,K):
    # make sure that aabb_box is in frustum at znear
    x_length, y_length, z_length = aabb_box[1] - aabb_box[0], aabb_box[3] - aabb_box[2], aabb_box[5] - aabb_box[4]
    aabb_box_center = np.array([(aabb_box[1] + aabb_box[0]) / 2, (aabb_box[3] + aabb_box[2]) / 2, (aabb_box[5] + aabb_box[4]) / 2])
    right_top_far_point = np.array([x_length / 2, y_length / 2, z_length / 2])

    # calculate znear to make sure that aabb_box is in frustum at znear

    w,h = 2*K[0,2], 2*K[1,2]
    scale_matrix = np.array([[w / 2, 0, 0, w / 2], [0, h / 2, 0, h / 2], [0, 0, 1, 0], [0, 0, 0, 1]])
    projection_matrix = np.dot(scale_matrix, project_matrix_from_vtk_camera)

    # generate corner points of aabb_box
    to_be_examined_points = [[x_length,y_length],
                             [y_length,x_length],
                             [x_length,z_length],
                             [z_length,x_length],
                             [y_length,z_length],
                             [z_length,y_length]]
    to_be_examined_points = np.array(to_be_examined_points)

    z_near = 0

    for each_point in to_be_examined_points:
        A1 = projection_matrix[0,0] * each_point[0]
        B1 = projection_matrix[0,1] * each_point[1]
        C1_para = projection_matrix[0,2]
        D1 = projection_matrix[0,3]

        A2 = projection_matrix[3,0] * each_point[0]
        B2 = projection_matrix[3,1] * each_point[1]
        C2_para = projection_matrix[3,2]
        D2 = projection_matrix[3,3]

        current_z_near = (A1+B1+D1 - w*(A2+B2+D2)) / (w*C2_para - C1_para)
        z_near = min(z_near,current_z_near)

    for each_point in to_be_examined_points:
        A1 = projection_matrix[1,0] * each_point[0]
        B1 = projection_matrix[1,1] * each_point[1]
        C1_para = projection_matrix[1,2]
        D1 = projection_matrix[1,3]

        A2 = projection_matrix[3,0] * each_point[0]
        B2 = projection_matrix[3,1] * each_point[1]
        C2_para = projection_matrix[3,2]
        D2 = projection_matrix[3,3]

        current_z_near = (A1+B1+D1 - h*(A2+B2+D2)) / (h*C2_para - C1_para)
        z_near = min(z_near,current_z_near)

    # calculate zfar to make sure that aabb_box take up the half of the screen at zfar

    z_far = -10000

    for each_point in to_be_examined_points:
        A1 = projection_matrix[0, 0] * each_point[0]
        B1 = projection_matrix[0, 1] * each_point[1]
        C1_para = projection_matrix[0, 2]
        D1 = projection_matrix[0, 3]

        A2 = projection_matrix[3, 0] * each_point[0]
        B2 = projection_matrix[3, 1] * each_point[1]
        C2_para = projection_matrix[3, 2]
        D2 = projection_matrix[3, 3]

        current_z_far = (A1 + B1 + D1 - (3*w/4) * (A2 + B2 + D2)) / ((3*w/4) * C2_para - C1_para)
        z_far = max(z_far, current_z_far)

    for each_point in to_be_examined_points:
        A1 = projection_matrix[1, 0] * each_point[0]
        B1 = projection_matrix[1, 1] * each_point[1]
        C1_para = projection_matrix[1, 2]
        D1 = projection_matrix[1, 3]

        A2 = projection_matrix[3, 0] * each_point[0]
        B2 = projection_matrix[3, 1] * each_point[1]
        C2_para = projection_matrix[3, 2]
        D2 = projection_matrix[3, 3]

        current_z_far = (A1 + B1 + D1 - (3*h/4) * (A2 + B2 + D2)) / ((3*h/4) * C2_para - C1_para)
        z_far = max(z_far, current_z_far)

    return z_near,z_far


def read_json(json_file):
    """
    # read json file
    # input: json_file: string
    # output: data: dict
    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_rotation_matrix(main_direction, look_at_direction):
    """
    # get rotation matrix
    # input: main_direction: np.array, look_at_direction: np.array
    # output: rotation_matrix: np.array
    """
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[0, 0] = main_direction[0]
    rotation_matrix[0, 1] = main_direction[1]
    rotation_matrix[0, 2] = main_direction[2]
    rotation_matrix[1, 0] = look_at_direction[0]
    rotation_matrix[1, 1] = look_at_direction[1]
    rotation_matrix[1, 2] = look_at_direction[2]
    rotation_matrix[2, 0] = np.cross(main_direction, look_at_direction)[0]
    rotation_matrix[2, 1] = np.cross(main_direction, look_at_direction)[1]
    rotation_matrix[2, 2] = np.cross(main_direction, look_at_direction)[2]
    return rotation_matrix


if __name__ == '__main__':
    # data = read_json("/data/endoscope/simulation_data/15:52:43/registration.json")
    # scene_points = np.array(data['scene_points'])
    # scene_points_normals = np.array(data['scene_points_normal'])
    # origin = data['cam_position_for_openGL']
    # look_at = data['look_at_position_for_openGL']
    # origin = np.array(origin)
    # look_at = np.array(look_at)
    #
    # main_dir = get_main_direction_of_normals(scene_points_normals)
    data = read_json('/data/endoscope/simulation_data/10:11:42/registration.json')
    look_at = data['look_at_position_for_openGL']
    origin = data['cam_position_for_openGL']

    look_at = np.array(look_at)
    origin = np.array(origin)

    look_at_direction = (look_at - origin) / np.linalg.norm(look_at - origin)

    bbox = data['init_bbox']
    bbox = np.array(bbox)
    cam_projection_matrix = data['cam_projection_matrix']
    cam_projection_matrix = np.array(cam_projection_matrix)
    K = data['intrinsics']
    K = np.array(K)
    znear,zfar = determine_znear_zfar(bbox,cam_projection_matrix,K)

    assert znear < 0 and zfar <0, "znear and zfar should be negative"
    assert znear > zfar, "znear should be greater than zfar"

    bbox_center =np.array([(bbox[1] + bbox[0]) / 2, (bbox[3] + bbox[2]) / 2, (bbox[5] + bbox[4]) / 2])

    z_distance_range = np.random.uniform(znear,zfar)
    random_position = z_distance_range * look_at_direction + origin
    translation = random_position - bbox_center

    # now we rotate the object
    all_normals = data['scene_points_normals']
    all_normals = np.array(all_normals)
    main_direction = get_main_direction_of_normals(all_normals)

    # rotate the object
    rotation_matrix = get_rotation_matrix(main_direction,look_at_direction)

    # generate homogeneous transformation matrix
    homogeneous_transformation_matrix = np.zeros((4, 4))
    homogeneous_transformation_matrix[0:3, 0:3] = rotation_matrix
    homogeneous_transformation_matrix[0:3, 3] = translation
    homogeneous_transformation_matrix[3, 3] = 1







