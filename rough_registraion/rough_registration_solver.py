from tqdm import tqdm
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
from Registration import apply_vec_on_points_3d


class LAR_rough_solver:

    def __init__(self, image_points, scene_points, K, guess=None,
                 max_iter=100, max_reprojection_error=8, match_step=10):
        self.image_points = image_points
        self.scene_points = scene_points
        self.image_shape = SC.Shape(image_points)
        self.scene_shape = SC.Shape(scene_points)
        self.K = K
        if guess is not None:
            self.guess = guess
        self.max_error = list(range(1, max_reprojection_error))
        self.max_iter = max_iter
        self.match_step = match_step
        self.inital_mhd = self.modified_hausdorff_distance(image_points, scene_points)

    def solve(self):
        print("Start to solve the rough registration")
        print(f'The initial mhd is {self.inital_mhd}')
        best_match = None
        best_rvec = None
        best_tvec = None
        for i in tqdm(range(self.max_iter)):
            if self.guess is not None:
                rvec, tvec = self.guess
                current_match = self.scene_shape.match(self.image_shape)
                _, rvec, tvec, current_inliers = cv2.solvePnPRansac(self.scene_points, self.image_points, self.K,
                                                                    disCoeffs=None,
                                                                    rvec=rvec if self.guess is not None else None,
                                                                    tvec=tvec if self.guess is not None else None,
                                                                    useExtrinsicGuess=True if self.guess is not None else False, )
            for j in range(self.match_step):
                current_scene_points = apply_vec_on_points_3d(self.scene_points, rvec / self.match_step,
                                                              tvec / self.match_step)
                current_projected_points = cv2.projectPoints(current_scene_points, None, None, self.K, disCoeffs=None)

    def modified_hausdorff_distance(self, source_points, target_points):
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



