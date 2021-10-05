import matlab.engine
import numpy as np
import os

m = matlab.engine.start_matlab()
m.addpath('.\CORFpushpull', nargout=0)


def corf_feature_maps(dataset):

    """

    Function to return an list of images and its labels

    :param dataset: Folder with class name and all the images in the dataset
    :return: Lists of CORF feature maps in an array form

    """

    list_of_image_paths = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for image in img_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + image
            images = m.imread(retrieve_dir)
            binarymap, corfresponse = m.CORFContourDetection(images, 2.2, 4, 1.8, 0.005, nargout=2)
            list_of_image_paths.append(corfresponse)

    return np.array(list_of_image_paths)
