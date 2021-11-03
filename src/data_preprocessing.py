import matlab.engine
import numpy as np
import os
import src.preprocessing
from matplotlib.pyplot import imshow
from sklearn import preprocessing
import cv2
#
m = matlab.engine.start_matlab()
m.addpath('.\src\preprocessing', nargout=0)


def batch_data_preprocessing(dataset):
    """

    Function to return an list of inpainted images

    :param dataset: Folder with class name and all the pre-segmentation images

    :return: list of inpainted images in an array form

    """

    list_of_inpainted_images = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        # img_list_range = range(1, len(img_list), 2)
        for image in img_list:
            thermal_image_path = dataset + "/" + folder_name + "/" + image
            # print(type(thermal_image_path))
            rgb_image_path = dataset + "/" + folder_name + "/" + image
            thermal_image = m.imread(thermal_image_path)
            rgb_image = m.imread(rgb_image_path)
            print(rgb_image_path)
            print(thermal_image_path)
            RGB, mask = m.segmentation(thermal_image, rgb_image, nargout=2)
            RGB = np.array(RGB)
            RGB = RGB.astype('uint8')
            mask = np.array(mask)
            mask = mask.astype('uint8')
            inpainted_image = cv2.inpaint(RGB, mask, 3, cv2.INPAINT_TELEA)
            imshow(inpainted_image)
            list_of_inpainted_images.append(inpainted_image)

    return np.array(list_of_inpainted_images)
