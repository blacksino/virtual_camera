import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import solveEPnP
import json
from ShapeContextMatching import *
from matplotlib.patches import ConnectionPatch
from torchvision.transforms import RandomAffine

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


def plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=None):
    source_points = [[each_point.x, each_point.y] for each_point in shape_source.shape_pts]
    target_points = [[each_point.x, each_point.y] for each_point in shape_target.shape_pts]

    source_points = np.array(source_points)
    target_points = np.array(target_points)

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
        ax[0].scatter(source_points[:, 0], source_points[:, 1])
        ax[0].set_title('source points')
        ax[1].scatter(target_points[:, 0], target_points[:, 1])
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
    else:
        fig, ax = plt.subplots(1, 2, subplot_kw={'polar': True}, figsize=(10, 5))
        ax = ax.flatten()
        scatter_logpolar_mpl(ax[0], a[:, 0], a[:, 1])
        scatter_logpolar_mpl(ax[1], b[:, 0], b[:, 1])
        plt.show()
        plt.cla()
        plt.clf()


def extract_red_curve(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    key_points_mask = np.zeros(image.shape[:-1])
    key_points_mask[(image[:, :, 0] > 150) & (image[:, :, 1] < 100) & (image[:, :, -1] < 100)] = 1
    return key_points_mask.astype(np.uint8)


def load_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data['extrinsics'], data['scene_points']


def project_points(points, extrinsics, K):
    extrinsics = np.linalg.inv(
        extrinsics)  # extrinsics must be inverted, because the points are in the camera coordinate system
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
    # exchange x and y
    points = points[::-1, :]
    points[0, :] = w - points[0, :]
    return points


def apply_affine_on_points(points):

    # apply random rotation on points
    theta = np.pi
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    points = np.dot(rotation_matrix, points)
    # apply random translation on points
    # points[:, 0] += np.random.uniform(-500, 500)
    # points[:, 1] += np.random.uniform(-500, 500)
    # apply random shear on points
    # points[:, 0] += np.random.uniform(-0.3, 0.3) * points[:, 1]
    return points




if __name__ == '__main__':
    mask = extract_red_curve("/home/SENSETIME/xulixin2/label.png")
    mask = morphology.skeletonize(mask)
    mask = mask.astype(np.uint8) * 255
    labeled_2d_points = np.where(mask == 255)
    # plt.imshow(mask)
    # plt.show()

    extrinsics, scene_points = load_json("/home/SENSETIME/xulixin2/registration.json")
    extrinsics = np.array(extrinsics)
    scene_points = np.array(scene_points)

    points_2d = project_points(scene_points, extrinsics, K)
    points_2d = points_2d[np.argsort(points_2d[:, 0])]

    labeled_2d_points = np.array(labeled_2d_points).T
    target_points_2d = convert_coordinates_from_cv_to_gl(labeled_2d_points.T).T
    target_points_2d = target_points_2d[np.argsort(target_points_2d[:, 0])]

    target_points_2d_rotated = apply_affine_on_points(target_points_2d.T).T
    points_2d_rotated = apply_affine_on_points(points_2d.T).T

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

    result = shape_source.matching(shape_target)
    result = np.array(result)
    # result[0] = result[0][::-1]
    # plt.scatter()

    plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=result)
    # plot_points_in_log_polar(shape_source,shape_target,style='polar')
