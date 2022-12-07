import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import json
from ShapeContextMatching import *
from matplotlib.patches import ConnectionPatch
from torchvision.transforms import RandomAffine
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree



def scatter_logpolar_mpl(ax, theta, r):
    ax.scatter(theta, r)
    ax.set_rlim(0)
    ax.set_rscale('symlog')
    ax.set_title('log-polar matplotlib')


def plot_points_in_log_polar(shape_source, shape_target, style='cartesian', result=None,use_shape=False):
    if use_shape:
        source_points = [[each_point[0], each_point[1]] for each_point in shape_source.shape]
        target_points = [[each_point[0], each_point[1]] for each_point in shape_target.shape]

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

    if style == 'cartesian':
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # set dpi
        fig.set_dpi(500)
        ax[0].plot(source_points[:, 0], source_points[:, 1],'b+',label='source')
        ax[0].set_title('source points')
        ax[1].plot(target_points[:, 0], target_points[:, 1],'r+', label='target')
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
        a = [list(Point(each_pt[0], each_pt[1]).cart2logpolar()) for each_pt in source_points]
        b = [list(Point(each_pt[0], each_pt[1]).cart2logpolar()) for each_pt in target_points]

        a = np.array(a)
        b = np.array(b)
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
    extrinsics = np.linalg.inv(extrinsics)# extrinsics must be inverted, because the points are in the camera coordinate system
    extrinsics[0,:] = -extrinsics[0,:]
    # convert points to homogeneous coordinates
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    # transform points to camera coordinates
    points = np.dot(extrinsics, points.T).T[:, :3]
    points = points / points[:, 2][:, np.newaxis]
    # project points to image coordinates
    points = np.dot(K, points.T).T
    # convert points to 2D

    # get rvec,tvec from extrinsics
    rvec, tvec = cv2.Rodrigues(extrinsics[:3,:3])[0], extrinsics[:3,3]
    # project points to image coordinates
    test, _ = cv2.projectPoints(three_d_points, rvec, tvec, K, np.zeros(5))

    return points[:, :2]



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

