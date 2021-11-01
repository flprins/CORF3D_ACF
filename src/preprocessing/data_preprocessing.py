import matlab.engine
import numpy as np
import os
from sklearn import preprocessing
import cv2
#
m = matlab.engine.start_matlab()
# m.addpath('.\CORFpushpull', nargout=0)


def batch_segmentation(dataset):
    """

    Function to return an list of segmented RGB images and their respective masks

    :param dataset: Folder with class name and all the pre-segmentation images

    :return: list of segmented RGB images and their respective masks in an array form

    """

    list_of_image_rgb= []
    list_of_image_mask = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for image in img_list:
            thermal_image_path = dataset + "/" + folder_name + "/" + image
            rgb_image_path = dataset + "/" + folder_name + "/" + image + 1
            thermal_image = m.imread(thermal_image_path)
            rgb_image = m.imread(rgb_image_path)
            RGB, mask = m.segmentation(thermal_image,rgb_image, nargout=2)
            list_of_image_rgb.append(RGB)
            list_of_image_rgb.append(mask)

    return np.array(list_of_image_rgb), np.array(list_of_image_mask)


def batch_inpaint(segmented_rgb, mask):
    """

    Function to return an list of inpainted images

    :param segmented_rgb: segmented RGB images
    :param mask : masks of the segmented RGB images

    :return: list of inpainted images in an array form

    """

    list_of_inpainted_images= []

    segmented_rgb_images = cv2.imread(segmented_rgb)
    inpainting_mask = cv2.imread(mask)
    inpainted_image = cv2.inpaint(segmented_rgb_images, inpainting_mask, 3, cv2.INPAINT_TELEA)
    list_of_inpainted_images.append(inpainted_image)

    return np.array(list_of_inpainted_images)