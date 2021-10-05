import matlab.engine
import numpy as np
import os

m = matlab.engine.start_matlab()
m.addpath('.\CORFpushpull', nargout=0)


def corf_feature_maps(dataset, sigma, beta, inhibitionFactor, highthresh ):

    """

    Function to return an list of CORF feature maps

    :param dataset: Folder with class name and all the pre-processed images
    :param sigma: The standard deviation of the DoG functions used
    :param beta: The increase in the distance between the sets of center-on
                 and ceter-off DoG receptive fields
    :param inhibitionFactor: The factor by which the response exhibited in the
                             inhibitory receptive field suppresses the response exhibited in the
                             excitatory receptive field
    :param highthresh: The high threshold of the non-maximum suppression used for binarization

    :return: Lists of the response map of the rotation-invariant CORF operator (CORF feature maps) in an array form

    """

    list_of_image_paths = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for image in img_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + image
            images = m.imread(retrieve_dir)
            binarymap, corfresponse = m.CORFContourDetection(images, sigma, beta, inhibitionFactor, highthresh, nargout=2)
            list_of_image_paths.append(corfresponse)

    return np.array(list_of_image_paths)


