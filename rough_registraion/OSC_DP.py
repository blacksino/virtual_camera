import math, cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import tps
import json
from TSP_greedy import creatMSTMap, TSP_greedy_heuristic
from choose_refer_and_recirc import choose_refer_and_recirc
from sklearn.metrics.pairwise import euclidean_distances
import itertools
from matplotlib import pyplot as plt


# from MatchTestForPnP import plot_points_in_log_polar


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def cart2logpolar(self):
        '''
            return (rho, theta)
        '''
        rho = math.sqrt(self.x * self.x + self.y * self.y)
        theta = math.atan2(self.y, self.x)
        return math.log(rho), theta

    def dist2(self, other):
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2


class Shape:
    def __init__(self, shape=None, img=None):
        '''
            shape -> 2d list of [[x1, y1], ...],
                     default shape is canny edges
            self.shape -> shape
            self.shape_pts -> list of Point instead of lists.
            self.shape_contexts -> list of [arrays -> shape_context]
        '''
        self.img = img
        if shape is None:
            shape = utils.canny_edge_shape(img)
        self.shape = shape
        self.shape_pts = []
        for point in shape:
            self.shape_pts.append(Point(point[0], point[1]))
        self.distance_map = np.zeros((len(self.shape_pts), len(self.shape_pts)))
        self.shape_contexts = self.get_shape_contexts()
        # self.log_polar_points = [point.cart2logpolar() for point in self.shape_pts]

    def log_space_generation(self, d1, d2, n):
        log_space = [(10 ** (d1 + k * (d2 - d1) / (n - 1))) for k in range(0, n - 1)]
        log_space.append(10 ** d2)
        return log_space

    def get_shape_contexts(self, angular_bins=12, radius_bins=6, inner_factor=None, outer_factor=None,
                           rotation_invariance=True, scale_invariance=True):
        """
            angular_bins -> number of bins for angle.
            radius_bins -> number of bins for radius,
                            default is maximum radius.
            return -> list of shape context in image (bin array)
        """
        # get maximum number of radius_bins
        log_space = None

        if not scale_invariance:
            print('You have disabled scale invariance.')
            print('radius_bins will be set automatically according to the shape.')
            max_dist2 = 0
            for i in range(len(self.shape_pts)):
                for j in range(len(self.shape_pts)):
                    max_dist2 = max(max_dist2, self.shape_pts[i].dist2(self.shape_pts[j]))
            radius_bins = int(math.log(math.sqrt(max_dist2))) + 1
        # old behavior
        else:
            for i in range(len(self.shape_pts)):
                for j in range(len(self.shape_pts)):
                    self.distance_map[i, j] = np.sqrt(self.shape_pts[i].dist2(self.shape_pts[j]))
            average_distance = np.average(self.distance_map)
            self.distance_map /= average_distance
            if inner_factor is None:
                inner_factor = self.distance_map.min() + 1e-7
            if outer_factor is None:
                outer_factor = self.distance_map.max() + 1e-2
            log_space = self.log_space_generation(math.log10(inner_factor * average_distance),
                                                  math.log10(outer_factor * average_distance),
                                                  radius_bins)

        shape_contexts = [np.zeros((radius_bins, angular_bins), dtype=float) for _ in range(len(self.shape_pts))]
        # get barycenter of shape
        barycenter = Point(0, 0)
        for point in self.shape_pts:
            barycenter.x += point.x
            barycenter.y += point.y
        barycenter.x /= len(self.shape_pts)
        barycenter.y /= len(self.shape_pts)
        # move shape to barycenter
        for i in range(len(self.shape_pts)):
            self.shape_pts[i].x -= barycenter.x
            self.shape_pts[i].y -= barycenter.y
        # compute bins
        for i in range(len(self.shape_pts)):
            for j in range(len(self.shape_pts)):
                if i == j:
                    continue
                pt = Point(self.shape_pts[j].x - self.shape_pts[i].x,
                           self.shape_pts[j].y - self.shape_pts[i].y)
                r, theta = pt.cart2logpolar()

                if rotation_invariance:
                    theta -= math.atan2(self.shape_pts[i].y, self.shape_pts[i].x)
                    theta = (theta + 2 * math.pi) % (2 * math.pi)
                    y = int(angular_bins * (theta / (math.pi + math.pi)))
                else:
                    if theta == math.pi:
                        y = angular_bins - 1
                    else:
                        y = int(angular_bins * ((theta + math.pi) / (math.pi + math.pi)))

                if scale_invariance:
                    assert log_space is not None
                    distance = np.exp(r)
                    # determine which bin the scaled_distance falls in
                    x = np.digitize(distance, log_space, right=True)
                else:
                    x = int(r)
                shape_contexts[i][x][y] += 1
        return [shape_context.reshape((radius_bins * angular_bins)) for shape_context in shape_contexts]

    def get_cost_matrix(self, Q, beta=0, robust=False, dummy_cost=1):
        '''
            Q -> instance of Shape
            beta -> coefficient of tangent_angle_dissimilarity,
                    1-beta is coefficient of shape_context_cost
            return -> (cost matrix for matching a points
                      from shape_context1 to shape_context2,
                      flag -> dummies added or not
                              cif not -> 0
                              if added to P -> -n
                              if added to Q -> m)
        '''

        def normalize_histogram(hist, total):
            new_hist = hist.copy()
            for i in range(hist.shape[0]):
                new_hist[i] /= float(total)
            return new_hist

        def shape_context_cost(nh1, nh2):
            '''
                nh1, nh2 -> normalized histogram
                return cost of shape context of
                two given shape context of the shape.
            '''
            cost = 0
            if nh1.shape[0] > nh2.shape[0]:
                nh1, nh2 = nh2, nh1
            nh1 = np.hstack([nh1, np.zeros(nh2.shape[0] - nh1.shape[0])])
            for k in range(nh1.shape[0]):
                if nh1[k] + nh2[k] == 0:
                    continue
                cost += (nh1[k] - nh2[k]) ** 2 / (nh1[k] + nh2[k])
            return cost / 2.0

        def tangent_angle_dissimilarity(p1, p2):
            '''
                p1 -> Point 1
                p2 -> Point 2
                return -> tangent angle dissimilarity of
                          given two points
            '''
            theta1 = math.atan2(p1.x, p1.y)
            theta2 = math.atan2(p2.x, p2.y)
            return .5 * (1 - math.cos(theta1 - theta2))

        if robust:
            raise ValueError('robust=True not supported yet.')
        n, m = len(self.shape_pts), len(Q.shape_pts)
        flag = min(n, m) if (n != m) else 0
        if flag and (n < m):
            flag = -flag
        mx = max(n, m)
        C = np.zeros((mx, mx))
        for i in range(mx):
            if n <= i:
                for j in range(mx):
                    C[i, j] = dummy_cost
            else:
                p = self.shape_pts[i]
                hist_p = normalize_histogram(self.shape_contexts[i], n - 1)
                for j in range(mx):
                    if m <= j:
                        C[i, j] = dummy_cost
                    else:
                        q = Q.shape_pts[j]
                        hist_q = normalize_histogram(Q.shape_contexts[j], m - 1)
                        C[i, j] = (1 - beta) * shape_context_cost(hist_p, hist_q) \
                                  + beta * tangent_angle_dissimilarity(p, q)

        return C, flag

    def matching(self, Q, with_perm=False):
        '''
            return -> two 2 x min(n, m) array.
                      (Pshape, Qshape) point i
                      from Pshape matched to
                      point i from Qshape.
        '''
        cost_matrix, flag = self.get_cost_matrix(Q, beta=0)
        perm = linear_sum_assignment(cost_matrix)[1]
        Pshape = np.array(self.shape)
        Qshape = np.array(Q.shape)
        # removing dummy matched.
        if flag < 0:
            mn = -flag
            perm = perm[:mn]
            Qshape = Qshape[perm]
        elif flag > 0:
            mn = flag
            mask = perm < mn
            perm = perm[mask]
            Pshape = Pshape[mask]
            Qshape = Qshape[perm]
        if with_perm:
            return Pshape, Qshape, perm
        return Pshape, Qshape

    def estimate_transformation(source, target):
        '''
            source -> n x 2 array of source points.
            target -> n x 2 array of source points.
            return -> bending energy, TPS class for transformation
        '''
        T = tps.TPS()
        BE = T.fit(source, target)
        return (BE, T)

    def shape_context_distance(self, Q_transformed, T):
        '''
            Q_transformed -> transformed target shape.
            T -> transformation function (TPS class)
            return -> shape context distance
        '''
        n, m = len(self.shape), len(Q_transformed.shape)
        cost_matrix = self.get_cost_matrix(Q_transformed)[0]
        ret1, ret2 = 0.0, 0.0
        for i in range(n):
            mn = 1e20
            for j in range(m):
                mn = min(mn, cost_matrix[i, j])
            ret1 += mn
        for j in range(m):
            mn = 1e20
            for \
                    i in range(n):
                mn = min(mn, cost_matrix[i, j])
            ret2 += mn
        return ret1 / n + ret2 / m

    def appearance_cost(source, target_transformed, img_p, img_q, std=1, window_size=3):
        '''
            source -> n x 2 array [source shape].
            target_transformed -> n x 2 array transformed target shape.
                                  [point i matched with point i from source]
            img_p -> source image.
            img_q -> target image.
            std -> scalar [standard deviation for guassian window].
            window_size -> size of guassian window.
            return -> appearance cost.
        '''

        def guassian_window(std, window_size):
            '''
                std -> scalar [standard deviation].
                window_size -> size of guassian window.
                return -> guassian window.
            '''
            window = np.zeros((window_size, window_size))
            for x in range(-(window_size // 2), window_size // 2 + 1):
                for y in range(-(window_size // 2), window_size // 2 + 1):
                    window[x][y] = math.exp(-(x * x + y * y) / (2 * std * std)) / (2 * math.pi * std * std)
            return window

        ret = 0
        G = guassian_window(std, window_size)
        for i in range(source.shape[0]):
            for x in range(-(window_size // 2), window_size // 2 + 1):
                for y in range(-(window_size // 2), window_size // 2 + 1):
                    px = min(int(x + source[i, 0]), img_p.shape[0] - 1)
                    py = min(int(y + source[i, 1]), img_p.shape[1] - 1)
                    Ip = int(img_p[px, py])
                    qx = min(int(x + target_transformed[i, 0]), img_q.shape[0] - 1)
                    qy = min(int(y + target_transformed[i, 1]), img_q.shape[1] - 1)
                    Iq = int(img_q[qx, qy])
                    ret += G[x + window_size // 2, y + window_size // 2] * (Ip - Iq) ** 2
        return ret / source.shape[0]

    def _distance(self, Q, w1, w2, w3, iterations=3):
        '''
            Q -> instance of Shape.
            w1 -> weight of Appearance Cost.
            w2 -> weight of Shape Contex distance.
            w3 -> weigth of Transformation Cost.
            iteration -> number of re-estimation of Transformation
                         estimation.
            return -> distance between two shapes.
        '''

        def transform_shape(Q, T):
            '''
                Q -> instance of Shape.
                T -> instance of TPS.
                return -> new Q which transformed with T.
            '''
            transformed_shape = []
            for q in Q.shape:
                Tq = T.transform(np.array(q).reshape((1, 2)))
                transformed_shape.append([Tq[0, 0], Tq[0, 1]])
            Q_transformed = Shape(transformed_shape, Q.img)
            return Q_transformed

        def transform_points(target_points, T):
            '''
                target_points -> n x 2 array of (x, y).
                T -> instance of TPS.
                return -> transform target_points with T.
            '''
            transformed_target = np.zeros_like(target_points)
            for i in range(target_points.shape[0]):
                new_pt = T.transform(target_points[i, :].reshape((1, 2)))
                transformed_target[i, :] = new_pt
            return transformed_target

        for i in range(iterations):
            source, target = self.matching(Q)
            BE, T = Shape.estimate_transformation(source, target)
            self = transform_shape(self, T)
        Q_transformed = transform_shape(Q, T)
        target_transformed = transform_points(target, T)
        AC = Shape.appearance_cost(source, target_transformed, self.img, Q.img)
        SC = self.shape_context_distance(Q, T)
        return w1 * AC + w2 * SC + w3 * BE


def distance(source_img, target_img, w1=1.6, w2=1, w3=.3):
    P = Shape(img=source_img)
    Q = Shape(img=target_img)
    return P._distance(Q, w1, w2, w3)


class utils:
    def canny_edge_shape(img, max_samples=100, t1=100, t2=200):
        '''
            return -> list of sampled Points from edges
                      founded by canny edge detector.
        '''
        edges = cv2.Canny(img, t1, t2)
        x, y = np.where(edges != 0)
        if x.shape[0] > max_samples:
            idx = np.random.choice(x.shape[0], max_samples, replace=False)
            x, y = x[idx], y[idx]
        shape = []
        for i in range(x.shape[0]):
            shape.append([x[i], y[i]])
        return shape


class OrientedShape(Shape):
    def __init__(self, shape=None, img=None, method='Hamilton'):
        self.img = img
        if shape is None:
            shape = utils.canny_edge_shape(img)
        self.shape = shape
        self.MST_map = None
        self.re_shape = None
        self.mapping = None

        self.distance_map = self.generate_distance_map()
        self.normalized_map = self.distance_map / np.mean(self.distance_map)
        self.reorder_shape()

        self.re_shape_pts = []
        for point in self.re_shape:
            self.re_shape_pts.append(Point(point[0], point[1]))

    def generate_distance_map(self):
        map = euclidean_distances(self.shape, self.shape)
        return map

    def reorder_shape(self):
        # shift to set the reference point as the 1st point
        farthest_idx, self.re_shape = choose_refer_and_recirc(self.shape, 'close')
        # preserve mapping between original shape and ordered shape
        Nx = self.shape.shape[0]
        self.mapping = np.zeros(Nx, dtype=int)
        self.mapping[:Nx - farthest_idx] = np.arange(farthest_idx, Nx)
        self.mapping[Nx - farthest_idx:] = np.arange(0, farthest_idx)
        self.mapping = np.argsort(self.mapping)
        zeros = np.zeros((Nx, Nx))
        index = np.array([np.arange(Nx), self.mapping]).T
        zeros[index[:, 0], index[:, 1]] = 1
        self.mapping = zeros

    def preprocess_for_dp_matching(self, Y):
        edge_frames = np.array([np.arange(self.re_shape.shape[0]), np.zeros(self.re_shape.shape[0])]).T.astype(int)
        sc_cost = self.sc_cost(edge_frames, Y)
        # generate the MST with first point excluded
        MST_adjacent_matrix = creatMSTMap(self.re_shape[1:])
        # convert adjacent matrix to edges list
        MST_edges = list()
        # Traverse the upper triangular part of the adjacency matrix
        for i in range(MST_adjacent_matrix.shape[0]):
            for j in range(i + 1, MST_adjacent_matrix.shape[1]):
                if MST_adjacent_matrix[i, j] != 0:
                    MST_edges.append([i + 1, j + 1])
        MST_edges = np.array(MST_edges)
        # plus 1 because the first point is excluded
        distance_map_Y = euclidean_distances(Y, Y)
        # generate Y_pair matrix using Cartesian product
        combinations = np.array(list(itertools.product(Y, Y))).reshape(Y.shape[0], Y.shape[0], 2, 2)
        combinations = combinations.reshape(Y.shape[0], Y.shape[0], 4)
        combinations = combinations.T
        Y_pair_flatten = combinations.reshape(4, Y.shape[0] * Y.shape[0])
        # generate 2*4*Nx similarity transformation matrix

        X_cords = np.expand_dims(self.re_shape.T, axis=1)
        X_t_cords_minus = np.expand_dims(np.array([-self.re_shape[:, 1], self.re_shape[:, 0]]), axis=1)
        eyes = np.repeat(np.eye(2).reshape(2, 2, 1), self.re_shape.shape[0], axis=2)
        Jacobian_X = np.concatenate((X_cords, X_t_cords_minus, eyes), axis=1)

        # number of nearest neighbours
        Nn = 15
        # find Nn nearest neighbour indices for each point in Y
        Y_nn_idx = np.argsort(distance_map_Y, axis=1)
        Y_nn_idx = Y_nn_idx[:, :Nn].T
        dist_x_to_ref = np.sqrt(np.sum(np.square(self.re_shape[0, :] - self.re_shape), axis=1))
        # choose closest point as root point
        root_idx = np.argsort(dist_x_to_ref)[1]
        pointer = np.zeros((Y.shape[0], Y.shape[0], self.re_shape.shape[0]))
        energy, pointer = self.DP_matching(root_idx, MST_edges.tolist(), sc_cost, self.re_shape, Jacobian_X,
                                           distance_map_Y, Y_pair_flatten, pointer, Y_nn_idx, 1)

        # rows for Yi， cols for ref point of Y

        # find the optimal path
        correspondences = np.ones(self.re_shape.shape[0]).astype(np.int)
        # select minimum energy point from current energy map
        idx, idy = np.unravel_index(np.argmin(energy), energy.shape)
        correspondences[0] = idy
        correspondences[root_idx] = idx
        pointer = pointer[:, idy, :]
        self.backtrack(root_idx, MST_edges.tolist(), pointer, correspondences)
        P = np.zeros((self.re_shape.shape[0], Y.shape[0]))
        for i in range(self.re_shape.shape[0]):
            P[i, correspondences[i]] = 1
        P = self.mapping @ P
        return P, energy.min()

    def backtrack(self, root, edges, pointer, correspondences):
        for i in range(len(edges)):
            if root in edges[i]:
                next = edges[i][0] if edges[i][1] == root else edges[i][1]
                correspondences[next] = pointer[correspondences[root], next]
                new_edges = edges.copy()
                new_edges.remove(edges[i])
                self.backtrack(next, new_edges, pointer, correspondences)

    def sc_cost(self, edge_frames, Y, num_of_Y_angles=50, outlier=None):

        def normalize_histogram(hist):
            # normalize each row of the histogram
            return hist / np.sum(hist, axis=1).reshape(hist.shape[0], 1)

        def shape_context_cost(h1, h2):
            '''
                nh1, nh2 -> normalized histogram
                return cost of shape context of
                two given shape context of the shape.
            '''

            # add new axis to make broadcasting possible
            h1 = h1.reshape(h1.shape[0], 1, h1.shape[1])
            h2 = h2.reshape(1, h2.shape[0], h2.shape[1])

            cost = np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10), axis=-1) * 0.5
            return cost

        num_of_Y = Y.shape[0]
        num_of_edge_frames = edge_frames.shape[0]

        # generate edge frames angles for self.shape
        edge_frames_vecs = self.re_shape[edge_frames[:, 0]] - self.re_shape[edge_frames[:, 1]]
        edge_frames_angles = np.arctan2(edge_frames_vecs[:, 1], edge_frames_vecs[:, 0])
        # These angles are used to determine the main orientation of the source set features.
        diff_y = np.zeros((num_of_Y, num_of_Y, 2))
        for i in range(num_of_Y):
            for j in range(num_of_Y):
                diff_y[i, j] = Y[i] - Y[j]
        angle_y = np.arctan2(diff_y[:, :, 1], diff_y[:, :, 0])

        self.shape_contexts = self.get_shape_contexts(self.re_shape, edge_frames_angles, )
        self.shape_contexts = np.array(self.shape_contexts)

        # generate 50 shape contexts for each point in Y,because we assume that the target set features are not oriented.
        cost = np.zeros((num_of_edge_frames, num_of_Y, num_of_Y_angles))
        for i in range(num_of_Y_angles):
            curreng_Y_angles = i * 2 * np.pi / num_of_Y_angles * np.ones(num_of_Y)
            current_sc = self.get_shape_contexts(Y, curreng_Y_angles)
            current_sc = np.array(current_sc)
            nh1 = normalize_histogram(self.shape_contexts)
            nh2 = normalize_histogram(current_sc)
            cost[:, :, i] = shape_context_cost(nh1, nh2)

        # make sure angle_y is in the range of [0,2*pi]
        angle_y = (angle_y + 2 * np.pi) % (2 * np.pi)
        # quantize the angle_y to 50 bins
        angle_y_bined = np.floor(angle_y / (2 * np.pi / num_of_Y_angles))
        # cost = np.transpose(cost, (1, 0, 2))
        final_cost = np.zeros((num_of_edge_frames, num_of_Y, num_of_Y))
        for i in range(num_of_edge_frames):
            for j in range(num_of_Y):
                final_cost[i, j] = cost[i, j, angle_y_bined[j].astype(int)]

        return np.transpose(final_cost, (1, 2, 0))

    def get_shape_contexts(self, shape_points, edge_frames_angles, angular_bins=12, radius_bins=5, inner_factor=0.125,
                           outer_factor=2.0, outliers=None):

        barycenter = np.mean(shape_points, axis=0)
        # move shape_points to origin
        shape_points = shape_points - barycenter
        diff_map = np.zeros((shape_points.shape[0], shape_points.shape[0], 2))
        for i in range(shape_points.shape[0]):
            for j in range(shape_points.shape[0]):
                diff_map[i, j] = shape_points[i] - shape_points[j]
        # generate dist map for shape_points
        dist_map = euclidean_distances(shape_points, shape_points)
        normed_dist_map = dist_map / np.mean(dist_map)

        if inner_factor is None:
            inner_factor = normed_dist_map.min() + 1e-7
        if outer_factor is None:
            outer_factor = normed_dist_map.max() + 1e-2

        log_space = self.log_space_generation(math.log10(inner_factor),
                                              math.log10(outer_factor),
                                              radius_bins)
        angle_map = np.arctan2(diff_map[:, :, 1], diff_map[:, :, 0]).T
        angle_map -= np.repeat(edge_frames_angles.reshape(-1, 1), shape_points.shape[0], axis=1)

        angle_map = (angle_map + 2 * np.pi) % (2 * np.pi)
        angle_idx = np.floor((angle_map / (2 * np.pi / angular_bins))-1e-6)
        angle_idx = angle_idx.astype(int)
        distance_idx = np.digitize(normed_dist_map, log_space)

        # frame angle is used to ensure rotation invariance

        shape_contexts = [np.zeros((radius_bins, angular_bins), dtype=float) for _ in range(len(shape_points))]

        for i in range(len(shape_points)):
            for j in range(len(shape_points)):
                if i == j:
                    continue
                # In the previous version, the theta angle will be subtracted from the angle of the source point.
                # Here, the angle of the edge frame will be subtracted.
                # theta -= math.atan2(shape_points[i].y, shape_points[i].x)
                y = angle_idx[i, j]
                # determine which bin the scaled_distance falls in
                x = distance_idx[i, j]
                if x >= radius_bins:
                    continue
                shape_contexts[i][x][y] += 1
        return [shape_context.reshape((radius_bins * angular_bins)) for shape_context in shape_contexts]

    def matching(self, Q, with_perm=False):
        new_shape = choose_refer_and_recirc(self.shape, 'close')
        Nx, Ny = self.shape.shape[0], Q.shape.shape[0]
        # generate distance map between shape and new shape
        distance_map = euclidean_distances(self.shape, new_shape)
        original_mapping = np.zeros((Nx, Nx))
        for i in range(Nx):
            min_idx = np.argmin(distance_map[i])
            original_mapping[i, min_idx] = 1
        Q_sc_contexts = Q.shape_contexts
        sc_cost = self.get_cost_matrix(self.shape_contexts, Q_sc_contexts)

    def DP_matching(self, root, edges, sc_cost, X, Jx, Y_dist_map,
                    Y_pair, pointer: np.array, Y_neighbor, lamda):
        num_of_neighbors = Y_neighbor.shape[0]
        num_of_Y = Y_dist_map.shape[0]
        root_edge_length = np.linalg.norm(X[root] - X[0])
        energy_map = np.zeros((num_of_Y, num_of_Y))
        # for each root ,there will be a energy map
        inv_Jacobian_root = np.linalg.inv(np.vstack((Jx[:, :, root], Jx[:, :, 0])))
        trans_root = np.reshape(inv_Jacobian_root @ Y_pair, (4, num_of_Y, num_of_Y))
        next_node = -1
        # loop through the tree
        for i in range(len(edges)):
            if root in edges[i]:
                next_node = edges[i][0] if edges[i][0] != root else edges[i][1]
                new_edges = edges.copy()
                new_edges.remove(edges[i])
                child_energy, pointer = self.DP_matching(next_node, new_edges, sc_cost, X, Jx, Y_dist_map,
                                                         Y_pair, pointer, Y_neighbor, lamda)
                Nc = 5
                refer_y_radius_range = max(root_edge_length / 7, 0.2)
                ref_y_map = np.abs(Y_dist_map - root_edge_length)
                ref_y_map[ref_y_map > refer_y_radius_range] = np.inf
                possible_y_ref_idx = [np.where(ref_y_map[i] != np.inf) for i in range(len(ref_y_map))]
                inv_Jacobian_next = np.linalg.inv(np.vstack((Jx[:, :, next_node], Jx[:, :, 0])))
                trans_next = np.reshape(inv_Jacobian_next @ Y_pair, (4, num_of_Y, num_of_Y))
                current_energy = np.ones((num_of_Y, num_of_Y)) * np.inf
                # loop all possible refer point of Y
                for j in range(num_of_Y):
                    possible_rows = Y_neighbor[:, j]
                    # we only consider the neighbors of selected point,because we assume the corresponding edge is small.
                    possible_cols = possible_y_ref_idx[j][0]
                    # we only consider the points within the circle band of the edge length of the root node.

                    candidate_energy = child_energy[possible_rows][:, possible_cols] + lamda * np.sum(np.abs(
                        np.repeat(trans_root[:, j:j + 1, possible_cols], repeats=num_of_neighbors, axis=1) - trans_next[
                                                                                                             :,
                                                                                                             possible_rows,
                                                                                                             :][:, :,
                                                                                                             possible_cols]
                    ), axis=0)
                    # choose points which minimize the energy
                    # in this loop, j represents the current point matched for Xi,
                    # possible rows represents Yj which is used to match the boundary edge of Xi in MST.
                    # possible cols represents the possible refer points of Point set Y.

                    # rows of candidate energy represents the possible points of Yj
                    # cols of candidate energy represents the possible refer points of Point set Y.

                    elected_energy = np.min(candidate_energy, axis=0)
                    # Yj choosed for Xi

                    current_energy[j, possible_cols] = elected_energy
                    Yj_index_selected = np.argmin(candidate_energy, axis=0)
                    for k in range(possible_cols.shape[0]):
                        pointer[j, possible_cols[k], next_node] = Y_neighbor[Yj_index_selected[k], j]
                        # pointer stores the mapping from Y to X
                        # shape of pointer is (num_of_Y, num_of_Y, num_of_X)

                energy_map += current_energy
        energy_map += sc_cost[:, :, root]
        # add shape context cost when we reach the leaf node or we have already looped through the tree of current root.

        return energy_map, pointer


if __name__ == '__main__':
    # X = np.loadtxt('fish_model_set.txt')
    # Y = np.loadtxt('fish_target_set.txt')
    X = np.loadtxt('/data/image_points.txt')
    Y = np.loadtxt('/data/projected_points.txt')

    # normalize the points
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    X = X / np.max(np.linalg.norm(X, axis=1))
    Y = Y / np.max(np.linalg.norm(Y, axis=1))


    plt.scatter(X[234:, 0], X[234:, 1], c='r', s=10)
    plt.scatter(Y[67:, 0], Y[67:, 1], c='b', s=10)
    plt.show()

    # X = np.loadtxt('fu_model_set.txt')
    # Y = np.loadtxt('fu_target_set.txt')
    # add some noise points to Y
    X = X[234:]
    Y = Y[67:]
    Y = np.delete(Y,63,axis=0)

    X_old_shape = Shape(X)
    Y_old_shape = Shape(Y)
    result = X_old_shape.matching(Y_old_shape)
    result = np.array(result)
    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c='r', s=10)
    # plt.scatter(Y[:, 0], Y[:, 1], c='b', s=10)
    # for i in range(result.shape[1]):
    #     plt.plot([result[0, i, 0], result[1, i, 0]],
    #              [result[0, i, 1], result[1, i, 1]],
    #              'g')
    # # keep x y scale same
    # plt.axis('equal')
    # plt.show()

    Y_OSC = OrientedShape(Y)
    correspondence_matrix, minimum_energy = Y_OSC.preprocess_for_dp_matching(X)

    # X-= np.mean(X,axis=0)
    # Y-= np.mean(Y,axis=0)

    # plot X and Y
    # fig = plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c='r', s=10)
    # plt.scatter(Y[:, 0], Y[:, 1], c='b', s=10)
    # draw line according to the correspondence matrix
    # for i in range(correspondence_matrix.shape[0]):
    #     for j in range(correspondence_matrix.shape[1]):
    #         if correspondence_matrix[i, j] == 1:
    #             plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'g')
    # keep x y scale same
    # plt.axis('equal')
    # plt.show()

    noisy_Y = Y
    # add some noise points to Y
    Y_new_shape = Shape(noisy_Y)
    noisy_result = X_old_shape.matching(Y_new_shape)
    noisy_result = np.array(noisy_result)
    new_matrix, new_mini = Y_OSC.preprocess_for_dp_matching(X)
    corresponding_X = X[np.where(new_matrix == 1)[0]]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=400)
    axes[0, 0].scatter(X[:, 0], X[:, 1], c='r', s=5, label='original point set')
    axes[0, 0].scatter(Y[:, 0], Y[:, 1], c='b', s=5, label='target point set')
    for i in range(result.shape[1]):
        axes[0, 0].plot([result[0, i, 0], result[1, i, 0]], [result[0, i, 1], result[1, i, 1]], 'g', linewidth=0.5)
    axes[0, 0].set_title('Matching result without noise using SC')
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    axes[0, 1].scatter(X[:, 0], X[:, 1], c='r', s=5, label='original point set')
    axes[0, 1].scatter(noisy_Y[:, 0], noisy_Y[:, 1], c='b', s=5, label='target point set')
    for i in range(noisy_result.shape[1]):
        axes[0, 1].plot([noisy_result[0, i, 0], noisy_result[1, i, 0]], [noisy_result[0, i, 1], noisy_result[1, i, 1]],
                        'g', linewidth=0.5)
    axes[0, 1].set_title('Matching result with noise using SC')
    axes[0, 1].legend()
    axes[0, 1].axis('equal')
    axes[1, 0].scatter(X[:, 0], X[:, 1], c='r', s=5, label='original point set')
    axes[1, 0].scatter(Y[:, 0], Y[:, 1], c='b', s=5, label='target point set')
    for i in range(correspondence_matrix.shape[0]):
        for j in range(correspondence_matrix.shape[1]):
            if correspondence_matrix[i, j] == 1:
                axes[1, 0].plot([Y[i, 0], X[j, 0]], [Y[i, 1], X[j, 1]], 'g', linewidth=0.5)
    axes[1, 0].set_title('Matching result without noise using OSC')
    axes[1, 0].legend()
    axes[1, 0].axis('equal')
    axes[1, 1].scatter(X[:, 0], X[:, 1], c='r', s=5, label='original point set')
    axes[1, 1].scatter(noisy_Y[:, 0], noisy_Y[:, 1], c='b', s=5, label='target point set')
    for i in range(new_matrix.shape[0]):
        for j in range(new_matrix.shape[1]):
            if new_matrix[i, j] == 1:
                axes[1, 1].plot([Y[i, 0], X[j, 0]], [Y[i, 1], X[j, 1]], 'g', linewidth=0.5)
    axes[1, 1].set_title('Matching result with noise using OSC')
    axes[1, 1].legend()
    axes[1, 1].axis('equal')
    # set main title
    fig.suptitle('Comparison of SC(Hungarian) and OSC(Dynamic Programming).')
    plt.show()


