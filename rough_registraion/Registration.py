import numpy as np
from matplotlib import pyplot as plt
import json
import os
import ShapeContextMatching as SC
import cv2
from MatchTestForPnP import load_json, project_points, \
    apply_affine_on_points, convert_coordinates_from_cv_to_gl,plot_points_in_log_polar
from scipy.spatial.transform import Rotation as R
from solveEPnP import rot_error

fx = 500.0
fy = 500.0

w = 1280
h = 720

cx = w / 2
cy = h / 2

K = np.array([[fx, 0., cx],
              [0., fy, cy],
              [0., 0., 1.]])


def plot_log_polar_for_sc(shape_context,vectorized_shape_context):
    # generate log polar coordinates
    ax,fig = plt.subplots(1,2,figsize=(10,5),dpi=500)






if __name__ == '__main__':
    extrinsics, scene_points = load_json("/home/SENSETIME/xulixin2/registration.json")
    image_points = np.loadtxt("/data/image_points.txt")
    projected_object_points = np.loadtxt("/data/projected_points.txt")
    # reverse image_points and projected_object_points along the y axis
    image_points[:, 1] = - image_points[:, 1]
    projected_object_points[:, 1] = - projected_object_points[:, 1]

    image_points[:,0] = fx * image_points[:,0]  + cx
    image_points[:,1] = fy * image_points[:,1]  + cy
    projected_object_points[:,0] = fx * projected_object_points[:,0]  + cx
    projected_object_points[:,1] = fy * projected_object_points[:,1]  + cy

    projected_object_points_using_extrinsics = project_points(scene_points, extrinsics, K)
    image_points = image_points[np.argsort(image_points[:, 0])]
    projected_object_points = projected_object_points[np.argsort(projected_object_points[:, 0])]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(image_points[:, 0], image_points[:, 1], c='r', s=1)
    ax[1].scatter(projected_object_points[:, 0], projected_object_points[:, 1], c='r', s=1)
    ax[2].scatter(projected_object_points_using_extrinsics[:, 0], projected_object_points_using_extrinsics[:, 1], c='r',
                  s=1)
    ax[0].set_title("image_points")
    ax[1].set_title("projected_points")
    ax[2].set_title("projected_points_using_extrinsics")
    plt.show()

    # try to solve pnp using opencv

    shape_image = SC.Shape(image_points)
    shape_projected = SC.Shape(projected_object_points)
    match_result,perm = shape_image.matching(shape_projected,with_perm=True)
    match_result_inv = shape_projected.matching(shape_image)

    _,rvec, tvec =cv2.solvePnP(scene_points, match_result_inv[1],K,distCoeffs=None,flags=cv2.SOLVEPNP_SQPNP,useExtrinsicGuess=False)
    cv2.solvePnPRansac(scene_points, match_result_inv[1], K, distCoeffs=None, useExtrinsicGuess=False,reprojectionError=1,confidence=0.99,maxIterations=100,minInlierRatio=0.1,inlierMask=None)

    new_points,_ = cv2.projectPoints(scene_points, rvec, tvec, K, distCoeffs=None)

    plt.scatter(new_points[:,0,0],new_points[:,0,1],c='r',s=1,label='solved by PnP')
    plt.scatter(projected_object_points[:,0],projected_object_points[:,1],c='b',s=1,label='gt')
    # make some annotations
    plt.legend()
    plt.show()

    sc_test_1 = shape_image.shape_contexts[0].reshape(6,-1)
    sc_test_2 = shape_projected.shape_contexts[0].reshape(6,-1)
    fig,ax = plt.subplots(1,2,figsize=(10,5),dpi=500)
    ax[0].imshow(sc_test_1.reshape(6,-1),cmap='gray')
    ax[1].imshow(sc_test_2.reshape(6,-1),cmap='gray')
    plt.show()



    plot_points_in_log_polar(shape_image,shape_projected,result=match_result)


