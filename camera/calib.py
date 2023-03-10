import cv2
import numpy as np
import os
from glob import glob
from matplotlib import pyplot as plt
import json


class ImageCalibrator:
    def __init__(self, img_root, chessboard_size=(8, 11), save_path=None, draw_corners=True):
        self.img_root = img_root
        self.chessboard_size = chessboard_size
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.draw_corners = draw_corners

    def _get_imgs_list(self):
        if glob(f"{self.img_root}/*.jpg"):
            self.imgs_path_list = glob(f"{self.img_root}/*.jpg")
        else:
            self.imgs_path_list = glob(f"{self.img_root}/*.png")
            if not self.imgs_path_list:
                raise Exception("No image found in the directory")

    def _read_all_imgs(self):
        self.imgs = []
        for img_path in self.imgs_path_list:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)

    def _find_chessboard_corner(self):

        self.obj_points = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.obj_points_list = []
        self.img_points_list = []

        for index, each_img in enumerate(self.imgs):
            gray = cv2.cvtColor(each_img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                # increase the accuracy of the corners
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                self.img_points_list.append(corners)
                self.obj_points_list.append(self.obj_points)
            else:
                print(f"No chessboard found in No.{index} image.")

    def _draw_corners(self, index):
        cv2.drawChessboardCorners(self.imgs[index], self.chessboard_size, self.img_points_list[index], True)
        # set dpi
        plt.figure(dpi=500)
        plt.imshow(self.imgs[index])
        plt.show()

    def calibrate_camera(self):
        self._get_imgs_list()
        self._read_all_imgs()
        self._find_chessboard_corner()
        if self.draw_corners:
            index = np.random.randint(0, len(self.imgs))
            self._draw_corners(index)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points_list, self.img_points_list,
                                                           self.imgs[0].shape[:2], None, None)

        self.mtx = mtx
        self.dist = dist

        if ret:
            print("Camera calibration is done.")
            print(f"Camera matrix is:{mtx}\n")
            print(f"Distortion coefficients are:{dist}")
        else:
            print("Camera calibration failed.")
        with open(os.path.join(self.save_path, "camera_calibration.json"), "w") as f:
            json.dump({"mtx": mtx.tolist(), "dist": dist.tolist()}, f)

        print("Camera calibration parameters are saved in the directory.")

    def undistort_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        x, y, w, h = roi
        # set 0 to the area outside the ROI
        new_img = np.zeros_like(dst)
        new_img[y:y + h, x:x + w] = dst[y:y + h, x:x + w]
        return new_img


if __name__ == "__main__":
    # get args from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, default="./", help="the directory of the calibration images")
    parser.add_argument("--save_path", type=str, default="./", help="the directory to save the calibration parameters")
    parser.add_argument("--draw_corners", type=bool, default=True, help="whether to draw the corners")
    args = parser.parse_args()

    # calibrate the camera
    calibrator = ImageCalibrator(args.img_root, save_path=args.save_path, draw_corners=args.draw_corners)
    calibrator.calibrate_camera()
    undistort_image = calibrator.undistort_image("/home/SENSETIME/xulixin2/RJ_demo/images/calib/vlcsnap-2023-02-24-13h38m02s325.png")
    undistort_image = cv2.cvtColor(undistort_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/data/undistort.png", undistort_image)
