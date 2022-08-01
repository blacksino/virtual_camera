import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import solveEPnP
import json
from ShapeContextMatching import *
from matplotlib.patches import ConnectionPatch
from torchvision.transforms import RandomAffine
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

fx = 500.0
fy = 500.0

w = 1280
h = 720

cx = w / 2
cy = h / 2

K = np.array([[fx, 0., cx],
              [0., fy, cy],
              [0., 0., 1.]])


def scatter_logpolar_mpl(ax, theta, r):
    ax.scatter(theta, r)
    ax.set_rlim(0)
    ax.set_rscale('symlog')
    ax.set_title('log-polar matplotlib')


def plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=None,use_shape=False):
    if use_shape:
        source_points = [[each_point.x, each_point.y] for each_point in shape_source.shape_pts]
        target_points = [[each_point.x, each_point.y] for each_point in shape_target.shape_pts]

        source_points = np.array(source_points)
        target_points = np.array(target_points)
    else:
        source_points = shape_source
        target_points = shape_target
    if style == 'polar':
        center_source = np.mean(source_points, axis=0)
        center_target = np.mean(target_points, axis=0)

        source_points = source_points - center_source
        target_points = target_points - center_target

    a = [list(Point(each_pt[0], each_pt[1]).cart2logpolar()) for each_pt in source_points]
    b = [list(Point(each_pt[0], each_pt[1]).cart2logpolar()) for each_pt in target_points]

    a = np.array(a)
    b = np.array(b)

    if style == 'cartesian':
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # set dpi
        fig.set_dpi(500)
        ax[0].scatter(source_points[:, 0], source_points[:, 1], s=1)
        ax[0].set_title('source points')
        ax[1].scatter(target_points[:, 0], target_points[:, 1], s=1)
        ax[1].set_title('target points')
        if result is not None:
            # add ConnectionPatch to connect the two points
            for i in range(result[0].shape[0]):
                con = ConnectionPatch(xyA=result[0][i], xyB=result[1][i],
                                      coordsA="data", coordsB="data",
                                      axesA=ax[0], axesB=ax[1], color="r")
                ax[1].add_artist(con)
        # set equal scale
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')

        plt.show()
        plt.cla()
        plt.clf()
    elif style == 'polar':
        fig, ax = plt.subplots(1, 2, subplot_kw={'polar': True}, figsize=(10, 5))
        ax = ax.flatten()
        scatter_logpolar_mpl(ax[0], a[:, 0], a[:, 1])
        scatter_logpolar_mpl(ax[1], b[:, 0], b[:, 1])
        plt.show()
        plt.cla()
        plt.clf()
    else:
        plt.figure(figsize=(10, 10), dpi=500)
        plt.scatter(source_points[:, 0], source_points[:, 1], s=1)
        plt.scatter(target_points[:, 0], target_points[:, 1], s=1)
        if result is not None:
            for i in range(result[0].shape[0]):
                plt.plot([result[1][i][0], result[0][i][0]], [result[1][i][1], result[0][i][1]], 'r')
        plt.show()
        plt.cla()
        plt.clf()


def extract_red_curve(img_path):
    image = cv2.imread(img_path)
    # resize image
    # image = cv2.resize(image, (2*w, 2*h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    key_points_mask = np.zeros(image.shape[:-1])
    key_points_mask[(image[:, :, 0] > 150) & (image[:, :, 1] < 100) & (image[:, :, -1] < 100)] = 1
    return key_points_mask.astype(np.uint8)


def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return np.array(data['extrinsics']), np.array(data['scene_points'])


def project_points(points, extrinsics, K):
    extrinsics = np.linalg.inv(extrinsics)  # extrinsics must be inverted, because the points are in the camera coordinate system
    # convert points to homogeneous coordinates
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    # transform points to camera coordinates
    points = np.dot(extrinsics, points.T).T[:, :3]
    points = points / points[:, 2][:, np.newaxis]
    # project points to image coordinates
    points = np.dot(K, points.T).T
    # convert points to 2D

    return points[:, :2]


def convert_coordinates_from_cv_to_gl(points: np.array):
    assert points.ndim==2
    flag = False
    if points.shape[1] == 2:
       points = points.T
       flag = True
    # exchange x and y
    points = points[::-1, :]
    points[0, :] = w - points[0, :]
    if flag:
        points = points.T
    return points


def apply_affine_on_points(points,theta = 0):
    # apply random rotation on points
    assert points.ndim == 2
    if points.shape[1] == 2:
        points = points.T
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        points = np.dot(rotation_matrix, points)
        return points.T
    else:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        points = np.dot(rotation_matrix, points)
        return points

if __name__ == '__main__':
    # mask = extract_red_curve("/home/SENSETIME/xulixin2/label.png")
    # mask = morphology.skeletonize(mask)
    # mask = mask.astype(np.uint8) * 255
    # labeled_2d_points = np.where(mask == 255)
    # # plt.imshow(mask)
    # # plt.show()
    #
    extrinsics, scene_points = load_json("/home/SENSETIME/xulixin2/registration.json")
    # extrinsics = np.array(extrinsics)
    # scene_points = np.array(scene_points)
    #
    # points_2d = project_points(scene_points, extrinsics, K)
    # points_2d = points_2d[np.argsort(points_2d[:, 0])]

    # read points from txt
    labeled_2d_points = np.loadtxt("/data/image_points.txt")
    points_2d = np.loadtxt("/data/projected_points.txt")

    # labeled_2d_points = np.array(labeled_2d_points).T
    # target_points_2d = convert_coordinates_from_cv_to_gl(labeled_2d_points.T).T
    target_points_2d = labeled_2d_points
    # target_points_2d = target_points_2d[np.argsort(target_points_2d[:, 0])]

    target_points_2d_rotated = apply_affine_on_points(target_points_2d)
    points_2d_rotated = apply_affine_on_points(points_2d)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(target_points_2d_rotated[:, 0], target_points_2d_rotated[:, 1])
    ax[0].set_title('target points')
    ax[1].scatter(points_2d_rotated[:, 0], points_2d_rotated[:, 1])
    ax[1].set_title('projected points')
    plt.show()

    # shape_target = Shape(shape=points_2d_rotated.tolist())
    shape_target = Shape(shape=target_points_2d_rotated.tolist())
    shape_source = Shape(shape=points_2d.tolist())

    # test code for rotation invariance

    # plot_points_in_log_polar(shape_source, shape_target, style='polar')
    shape_context_source_test = shape_source.shape_contexts[0].reshape(6, -1)
    shape_context_target_test = shape_target.shape_contexts[0].reshape(6, -1)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(shape_context_source_test, cmap='gray')
    ax[1].imshow(shape_context_target_test, cmap='gray')
    plt.show()

    # result = shape_source.matching(shape_target)
    # result = np.array(result)
    result_1 = shape_target.matching(shape_source)
    result_1 = np.array(result_1)

    # compare to the result of using nearest neighbor

    tree = KDTree(target_points_2d_rotated)
    dist, ind = tree.query(points_2d, k=1)

    nn_result = []
    nn_result.append(np.array([points_2d[i] for i in range(len(points_2d))]))
    nn_result.append(np.array([target_points_2d_rotated[each_ind] for each_ind in ind]).reshape(-1, 2))
    nn_result = np.array(nn_result)
    # convert list to numpy array

    # result[0] = result[0][::-1]
    # plt.scatter()

    plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=result_1)
    # plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=nn_result)
    # # plot_points_in_log_polar(shape_source,shape_target,style='polar')
    # img_1 = cv2.imread("/home/SENSETIME/xulixin2/sc_1.png")
    # img_2 = cv2.imread("/home/SENSETIME/xulixin2/sc_2.png")
    # img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    # img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    # # plot 2 image in one figure
    # fig, ax = plt.subplots(1, 2, figsize=(12, 8),dpi=500)
    # ax[0].imshow(img_1)
    # ax[1].imshow(img_2)
    # plt.show()
    #
    # #
    # beta_1 = extract_red_curve("/home/SENSETIME/xulixin2/sc_1.png")
    # beta_2 = extract_red_curve("/home/SENSETIME/xulixin2/sc_2.png")
    # beta_1 = morphology.skeletonize(beta_1)
    # beta_2 = morphology.skeletonize(beta_2)
    # beta_1 = beta_1.astype(np.uint8) * 255
    # beta_2 = beta_2.astype(np.uint8) * 255
    # labeled_2d_points_1 = np.where(beta_1 == 255)
    # labeled_2d_points_2 = np.where(beta_2 == 255)
    # labeled_2d_points_1 = np.array(labeled_2d_points_1).T
    # labeled_2d_points_2 = np.array(labeled_2d_points_2).T
    # labeled_2d_points_1 = convert_coordinates_from_cv_to_gl(labeled_2d_points_1.T).T
    # labeled_2d_points_2 = convert_coordinates_from_cv_to_gl(labeled_2d_points_2.T).T
    # labeled_2d_points_1 = labeled_2d_points_1[np.argsort(labeled_2d_points_1[:, 0])]
    # labeled_2d_points_2 = labeled_2d_points_2[np.argsort(labeled_2d_points_2[:, 0])]
    # shape_target_1 = Shape(shape=labeled_2d_points_1.tolist())
    # shape_target_2 = Shape(shape=labeled_2d_points_2.tolist())
    # shape_context_1 = shape_target_1.shape_contexts[0].reshape(6, -1)
    # shape_context_2 = shape_target_2.shape_contexts[0].reshape(6, -1)
    # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    # ax[0].imshow(shape_context_1, cmap='gray')
    # ax[1].imshow(shape_context_2, cmap='gray')
    # plt.show()
    # result_1 = shape_target_1.matching(shape_target_2)
    # result_1 = np.array(result_1)
    # plot_points_in_log_polar(shape_target_1, shape_target_2, style='cartesian', result=result_1)
    #
    # kd_tree_1 = KDTree(labeled_2d_points_1)
    # dist,ind = kd_tree_1.query(labeled_2d_points_2,k=1)
    # nn_result_1 = []
    # nn_result_1.append(np.array([labeled_2d_points_1[each_ind] for each_ind in ind]).reshape(-1, 2))
    # nn_result_1.append(np.array([labeled_2d_points_2[i] for i in range(len(labeled_2d_points_2))]))
    # nn_result_1 = np.array(nn_result_1)
    # plot_points_in_log_polar(shape_target_1, shape_target_2, style='cartesian', result=nn_result_1)
