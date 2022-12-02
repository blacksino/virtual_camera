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
from utils.FPS import FarthestPointSampler

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


def plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=None, use_shape=False):
    if use_shape:
        source_points = [[each_point.x, each_point.y] for each_point in shape_source.original_shape_pts]
        target_points = [[each_point.x, each_point.y] for each_point in shape_target.original_shape_pts]

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
                con = ConnectionPatch(xyA=result[1][i], xyB=result[0][i],
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
    three_d_points = points
    extrinsics = np.linalg.inv(
        extrinsics)  # extrinsics must be inverted, because the points are in the camera coordinate system
    extrinsics[0, :] = -extrinsics[0, :]
    # convert points to homogeneous coordinates
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    # transform points to camera coordinates
    points = np.dot(extrinsics, points.T).T[:, :3]
    points = points / points[:, 2][:, np.newaxis]
    # project points to image coordinates
    points = np.dot(K, points.T).T
    # convert points to 2D

    # get rvec,tvec from extrinsics
    rvec, tvec = cv2.Rodrigues(extrinsics[:3, :3])[0], extrinsics[:3, 3]
    # project points to image coordinates
    test, _ = cv2.projectPoints(three_d_points, rvec, tvec, K, np.zeros(5))

    return points[:, :2]


def convert_coordinates_from_cv_to_gl(points: np.array):
    assert points.ndim == 2
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


def apply_affine_on_points(points, theta=0):
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
    mask = extract_red_curve("/home/SENSETIME/xulixin2/图片/endo_label.png")
    img = cv2.imread("/home/SENSETIME/xulixin2/图片/endo_label.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = morphology.skeletonize(mask)
    mask = mask.astype(np.uint8) * 255
    labeled_2d_points = np.where(mask == 255)
    extrinsics, scene_points = load_json("/data/endoscope/simulation_data/14:50:17/registration.json")

    K = np.array([
        [9.9640068207290187e+02, 0.0, 3.7502582168579102e+02],
        [0.0,9.9640068207290187e+02, 2.4026374816894531e+02],
        [0.0, 0.0,  1.0,]]
    )

    projected_points = project_points(scene_points, extrinsics, K)
    #get coordinates of mask
    x,y = mask.nonzero()
    mask_points = np.vstack((x,y)).T

    # plot projected points
    projected_img = np.zeros(img.shape[:-1])
    for each in projected_points:
        projected_img[int(each[1]), int(each[0])] = 255

    # fig, ax = plt.subplots(3, 1, figsize=(4, 6),dpi=300)
    # ax[0].imshow(img)
    # ax[1].imshow(mask,cmap='gray')
    # ax[2].imshow(projected_img,cmap='gray')
    #
    # plt.tight_layout()
    # plt.show()

    # shape_target = Shape(shape=points_2d_rotated.tolist())
    sampler = FarthestPointSampler(mask_points,projected_points.shape[0]+1)
    mask_points = sampler.sample()

    shape_target = Shape(shape=projected_points.tolist())
    shape_source = Shape(shape=mask_points.tolist())

    matching_result,perm = shape_target.matching(shape_source,with_perm=True)
    matching_result = np.array(matching_result)

    plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=matching_result,use_shape=True)
