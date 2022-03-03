import cv2
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from matplotlib import pyplot as plt
from skimage.measure import regionprops,label

init_frame = True

def rot_error(r_gt,r_est):
    dis = abs(math.acos((np.trace(np.dot(np.linalg.inv(r_gt),r_est))-1)/2))
    #公式计算结果单位为弧度，转成角度返回
    return dis*180/math.pi

if init_frame:
    json_path = '/data/endoscope/simulation_data/20:51:44.json'
    info = dict()

    with open(json_path, 'r') as f:
        info = json.load(f)

    gt_extrinsics = np.array(info['extrinsics'])
    image_points = np.array(info['pixel_points'])
    scene_points = np.array(info['scene_points'])
    K = np.array(info['intrinsics'])

    w, h = 2 * K[0, 2], 2 * K[1, 2]
    image_points[:, 0] = w - 1 - image_points[:, 0]
    # image_points[:, 1] = h - 1 - image_points[:, 1]


    transform = cv2.solvePnP(scene_points.astype(np.float64),image_points.astype(np.float64),
                             K.astype(np.float64),distCoeffs=None,flags=cv2.SOLVEPNP_EPNP) # 当可用点仅有四点时。

    rot_matrix = cv2.Rodrigues(transform[1])[0].T
    translation = -rot_matrix @ transform[2]

    translation_error = translation.flatten() - gt_extrinsics[:3,3]
    angle_error = rot_error(gt_extrinsics[:3,:3],rot_matrix)
    print(f'translation error is {round(np.linalg.norm(translation_error),3)} mm.')
    print(f'angle error is {180 - round(angle_error,3)} degree.')

else:
    image = cv2.imread('/home/SENSETIME/xulixin2/Pictures/endo_ar.png')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    json_path = '/data/endoscope/simulation_data/19:41:09.json'
    old_json_path = '/data/endoscope/simulation_data/14:19:49.json'

    info = dict()
    old_info = dict()
    with open(json_path, 'r') as f:
        info = json.load(f)
    with open(old_json_path, 'r') as f:
        old_info = json.load(f)


    gt_extrinsics = np.array(old_info['extrinsics'])
    image_points = np.array(info['pixel_points'])
    scene_points = np.array(info['scene_points'])
    K = np.array(info['intrinsics'])
    scene_points.sort(0)

    key_points_mask = np.zeros(image.shape[:-1])
    key_points_mask[(image[:,:,0]>150)&(image[:,:,1]<100)&(image[:,:,-1]<100)] = 1
    key_points_mask = label(key_points_mask)
    regions = regionprops(key_points_mask)

    key_points = np.array([each_region.centroid for each_region in regions])
    key_points = key_points[:,[1,0]]    #交换xy
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    key_points[:,0] = w-1 - key_points[:,0]
    key_points[:,1] = h-1 - key_points[:,1]

    transform = cv2.solvePnP(scene_points.astype(np.float64), key_points.astype(np.float64),
                             K.astype(np.float64), distCoeffs=None,flags=cv2.SOLVEPNP_EPNP)


    rot_matrix = cv2.Rodrigues(transform[1])[0].T
    translation = -rot_matrix @ transform[2]

    translation_error = translation.flatten() - gt_extrinsics[:3, 3]
    angle_error = rot_error(gt_extrinsics[:3, :3], rot_matrix)

    print(f'translation error is {round(np.linalg.norm(translation_error),3)} mm.')
    print(f'angle error is {round(angle_error,3)} degree.')

