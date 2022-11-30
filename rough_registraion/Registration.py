import numpy as np
from matplotlib import pyplot as plt
import json
import os
import ShapeContextMatching as SC
import cv2
from MatchTestForPnP import load_json, project_points, \
    apply_affine_on_points, convert_coordinates_from_cv_to_gl, plot_points_in_log_polar
from scipy.spatial.transform import Rotation as R
from solveEPnP import rot_error
from OSC_DP import OrientedShape





fx = 500.0
fy = 500.0

w = 1280
h = 720

cx = w / 2
cy = h / 2

K = np.array([[fx, 0., cx],
              [0., fy, cy],
              [0., 0., 1.]])


def apply_vec_on_points_3d(scene_points, rvec, tvec: np.array):
    rot_mat = R.from_rotvec(rvec.flatten()).as_matrix()
    scene_points_transformed = np.dot(scene_points, rot_mat.T) + tvec.T.repeat(scene_points.shape[0], axis=0)
    return scene_points_transformed


def modified_hausdorff_distance(source_points, target_points):
    min_distance_sum_from_source_to_target = 0
    min_distance_sum_from_target_to_source = 0
    for source_point in source_points:
        min_distance_from_source_to_target = float('inf')
        for target_point in target_points:
            distance = np.linalg.norm(source_point - target_point)
            if distance < min_distance_from_source_to_target:
                min_distance_from_source_to_target = distance
        min_distance_sum_from_source_to_target += min_distance_from_source_to_target

    for target_point in target_points:
        min_distance_from_target_to_source = float('inf')
        for source_point in source_points:
            distance = np.linalg.norm(source_point - target_point)
            if distance < min_distance_from_target_to_source:
                min_distance_from_target_to_source = distance
        min_distance_sum_from_target_to_source += min_distance_from_target_to_source

    mhd = max(min_distance_sum_from_source_to_target / source_points.shape[0],
              min_distance_sum_from_target_to_source / target_points.shape[0])

    return mhd


if __name__ == '__main__':
    extrinsics, scene_points = load_json("/home/SENSETIME/xulixin2/registration.json")
    # image_points = np.loadtxt("/data/image_points.txt")
    image_points = np.loadtxt("/data/image_points_wo_sampled.txt")
    projected_object_points = np.loadtxt("/data/projected_points.txt")
    # reverse image_points and projected_object_points along the y-axis
    image_points[:, 1] = - image_points[:, 1]
    projected_object_points[:, 1] = - projected_object_points[:, 1]

    image_points[:, 0] = fx * image_points[:, 0] + cx
    image_points[:, 1] = fy * image_points[:, 1] + cy
    projected_object_points[:, 0] = fx * projected_object_points[:, 0] + cx
    projected_object_points[:, 1] = fy * projected_object_points[:, 1] + cy

    # r = R.from_euler('xyz', [-np.pi/12, -np.pi/3, -np.pi/12])
    # rot = r.as_matrix()
    # noise = np.zeros((4, 4))
    # noise[:3, :3] = rot
    # noise[3, 3] = 1
    # extrinsics = noise @ extrinsics

    projected_object_points_using_extrinsics = project_points(scene_points, extrinsics, K)

    plt.scatter(image_points[:, 0], image_points[:, 1], c='r', s=1)
    plt.scatter(projected_object_points_using_extrinsics[:, 0], projected_object_points_using_extrinsics[:, 1], c='b',s=1)
    #set aspact ratio to 1
    plt.gca().set_aspect('equal')
    # set title
    plt.title('image points and projected object points')
    plt.show()

    # plot 3d scene points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2], c='g', s=2)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_title("scene_points")
    # # change view angle
    # ax.view_init(elev=12., azim=10)
    #
    # plt.show()

    # try to solve pnp using opencv
    shape_image = SC.Shape(image_points)
    shape_projected = SC.Shape(projected_object_points_using_extrinsics)

    match_result, p1 = shape_image.matching(shape_projected, 1)
    match_result_inv, p2 = shape_projected.matching(shape_image, 1)



    matrix,_ = OrientedShape(projected_object_points_using_extrinsics).preprocess_for_dp_matching(image_points)
    matched_using_dp = projected_object_points_using_extrinsics[np.where(matrix==1)[1]]

    # _,rvec, tvec =cv2.solvePnP(scene_points, match_result_inv[1],K,distCoeffs=None,flags=cv2.SOLVEPNP_SQPNP,useExtrinsicGuess=False)
    _, rvec, tvec, inliers = cv2.solvePnPRansac(scene_points, match_result_inv[1], K, distCoeffs=None,
                                                useExtrinsicGuess=False, reprojectionError=2, confidence=0.99)

    # for i in range(4):
    #     # #apply rotation and translation on scene_points
    #     # scene_points = apply_vec_on_points_3d(scene_points,rvec/2,tvec/2)
    #     #match again
    #     projected_object_points_using_extrinsics = cv2.projectPoints(scene_points,rvec,tvec,K,distCoeffs=None)[0].reshape(-1,2)
    #     shape_projected = SC.Shape(projected_object_points_using_extrinsics)
    #     match_result, p1 = shape_image.matching(shape_projected, 1)
    #     match_result_inv, p2 = shape_projected.matching(shape_image, 1)
    #     plot_points_in_log_polar(image_points, projected_object_points_using_extrinsics, result=match_result)
    #     _, rvec, tvec, inliers = cv2.solvePnPRansac(scene_points, match_result_inv[1], K, distCoeffs=None,
    #                                                 useExtrinsicGuess=True,rvec=rvec,tvec=tvec, reprojectionError=16/(2**i), confidence=0.99)

    new_points, _ = cv2.projectPoints(scene_points, rvec, tvec, K, distCoeffs=None)

    mhd_before = modified_hausdorff_distance(image_points, projected_object_points_using_extrinsics)
    mhd = modified_hausdorff_distance(image_points, new_points)
    print("mhd_before:", mhd_before)
    print("mhd:", mhd)

    plt.scatter(new_points[:, 0, 0], new_points[:, 0, 1], c='r', s=2, label='solved by LAR')
    plt.scatter(image_points[:, 0], image_points[:, 1], c='b', s=2, label='gt')
    plt.scatter(projected_object_points_using_extrinsics[:, 0], projected_object_points_using_extrinsics[:, 1], c='g',
                s=2, label='guess')
    # make some annotations
    plt.legend()
    plt.show()

    plot_points_in_log_polar(image_points, projected_object_points_using_extrinsics, result=match_result)
