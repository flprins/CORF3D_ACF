import matlab.engine
import numpy as np
import os
import cv2
#
m = matlab.engine.start_matlab()
m.addpath('.\src\preprocessing', nargout=0)


def batch_data_preprocessing(dataset):
    """

    Function to return an list of inpainted images

    :param dataset: Folder with class name and all the pre-segmentation images containing RGB
    and themral images

    :return: list of inpainted images in an array form

    """

    list_of_inpainted_images = []
    labels = []
    labels_list = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for idx in range(0, len(img_list), 2):
            rgb_image_idx = img_list[idx]
            rgb_image_path = dataset + "/" + folder_name + "/" + rgb_image_idx
            thermal_image_idx = img_list[idx + 1]
            thermal_image_path = dataset + "/" + folder_name + "/" + thermal_image_idx
            thermal_image = m.imread(thermal_image_path)
            rgb_image = m.imread(rgb_image_path)
            RGB, mask = m.segmentation(thermal_image, rgb_image, nargout=2)
            RGB = np.array(RGB)
            RGB = RGB.astype('uint8')
            mask = np.array(mask)
            mask = mask.astype('uint8')
            inpainted_image = cv2.inpaint(RGB, mask, 3, cv2.INPAINT_TELEA)
            inpainted_image = cv2.resize(inpainted_image, (224, 224))
            list_of_inpainted_images.append(inpainted_image)
        labels_list.append(folder_name)
        labels.append([folder_name] * len(img_list))

    return np.array(list_of_inpainted_images), labels, labels_list
