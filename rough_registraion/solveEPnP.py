import cv2
import json
import os
import numpy as np
import open3d as o3d
import skimage
from scipy.spatial.transform import Rotation as R

import math
from matplotlib import pyplot as plt
from skimage.measure import regionprops,label

init_frame = True

from skimage import morphology


def extract_specific_color_region(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    key_points_mask = np.zeros(image.shape[:-1])
    key_points_mask[(image[:,:,0]>150)&(image[:,:,1]<100)&(image[:,:,-1]<100)] = 1
    return key_points_mask.astype(np.uint8)

def skeleton(image):
    gray = image
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton

def image2cloud(image,K,scale,distortion=None):
    #we only convert skeleton point onto 3d space
    index = np.where(image==1)
    arr = np.array(list(zip(index[1],index[0])))
    arr[:,0] = 1279 - arr[:,0]
    arr[:,1] = 719 -arr[:,1]
    arr = np.c_[arr,np.ones((arr.shape[0],1))]
    K_inv = np.linalg.inv(K)

    reprojected_pts = K_inv @ arr.T
    pc = o3d.open3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(reprojected_pts).T)
    o3d.io.write_point_cloud("/data/endoscope/simulation_data/label_contour_reprojected.ply",
                             pc, write_ascii=True)


def cloud2image(extrinsics,intrinsics,point_cloud_path="/data/endoscope/simulation_data/contour.ply",using_guess=True):
    pc = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pc.points)
    points = np.c_[points,np.ones(points.shape[0])]
    new_points = np.linalg.inv(extrinsics) @ points.T
    new_points = new_points[:3,:]
    if using_guess:
        new_pc = o3d.open3d.geometry.PointCloud()
        new_pc.points = o3d.open3d.utility.Vector3dVector(new_points.T)
        o3d.io.write_point_cloud("/data/endoscope/simulation_data/contour_guessed.ply",
                                 new_pc, write_ascii=True)
    new_points /= new_points[-1, :]
    projected_new_points = intrinsics @ new_points
    return projected_new_points


def rot_error(r_gt,r_est):
    dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
    #公式计算结果单位为弧度，转成角度返回
    return dis*180/math.pi

if init_frame:
    json_path = '/data/endoscope/simulation_data/17:39:29/registration.json'
    registration_json_path = '/data/endoscope/simulation_data/17:39:29/registration.json'
    info = dict()
    registration_info = dict()

    with open(json_path, 'r') as f:
        info = json.load(f)

    with open(registration_json_path, 'r') as f:
        registration_info = json.load(f)

    gt_extrinsics = np.array(info['extrinsics'])
    K = np.array(info['intrinsics'])
    K[-1][-1] = 1
    scene_points = np.array(registration_info['scene_points'])
    # contour_points = np.array(registration_info['coutour'])

    image = cv2.imread('/data/endoscope/simulation_data/17:39:29/registration.png')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    key_points_mask = np.zeros(image.shape[:-1])
    key_points_mask[(image[:,:,0]>150)&(image[:,:,1]<100)&(image[:,:,-1]<100)] = 1
    key_points_mask = label(key_points_mask)
    regions = regionprops(key_points_mask)

    key_points = np.array([each_region.centroid for each_region in regions])
    key_points = key_points[:,[1,0]] # 交换xy
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    # key_points[:,0] = w-1 - key_points[:,0]
    # key_points[:,1] = h-1 - key_points[:,1]
    key_points = key_points[key_points[:, 0].argsort(), :]
    transform = cv2.solvePnP(scene_points.astype(np.float64),key_points.astype(np.float64),
                             K.astype(np.float64),distCoeffs=None,flags=cv2.SOLVEPNP_EPNP) # 当可用点仅有四点时。

    rot_matrix = cv2.Rodrigues(transform[1])[0].T
    translation = -rot_matrix @ transform[2]
    homogeneous_transform = np.zeros((4,4))
    homogeneous_transform[:3,:3] = rot_matrix
    homogeneous_transform[:3, 3] = translation.flatten()
    homogeneous_transform[-1,-1] = 1
    translation_error = translation.flatten() - gt_extrinsics[:3,3]
    angle_error = rot_error(gt_extrinsics[:3,:3],rot_matrix)
    print(f'translation error is {round(np.linalg.norm(translation_error),3)} mm.')
    print(f'angle error is {180 - round(angle_error,3)} degree.')
    # pc = o3d.open3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(np.array(contour_points))
    # o3d.io.write_point_cloud("/data/endoscope/simulation_data/contour.ply", pc, write_ascii=True)
    #
    # skeleton_image = extract_specific_color_region('/data/endoscope/simulation_data/contour.jpg')
    # image2cloud(skeleton_image,K,1.0)
    # cloud2image(homogeneous_transform,K)



