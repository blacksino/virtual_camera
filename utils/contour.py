import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology


def extract_specific_color_region(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    key_points_mask = np.zeros(image.shape[:-1])
    key_points_mask[(image[:,:,0]>150)&(image[:,:,1]<100)&(image[:,:,-1]<100)] = 1
    return key_points_mask.astype(np.uint8)

def skeleton_demo(image):
    gray = image
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton


if __name__ == '__main__':
    mask = extract_specific_color_region('/data/endoscope/simulation_data/contour.jpg')
