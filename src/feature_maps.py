import matlab.engine
import numpy as np
import os
import cv2
from keras import Model
from sklearn import preprocessing

m = matlab.engine.start_matlab()


def corf_feature_maps(dataset, sigma, beta, inhibitionFactor, highthresh):

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

    :return: List of the response map of the rotation-invariant CORF operator (CORF feature maps) in an array form

    """

    list_of_image_paths = []

    m.addpath('.\src\CORFpushpull', nargout=0)

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for image in img_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + image
            images = m.imread(retrieve_dir)
            binarymap, corfresponse = m.CORFContourDetection(images, sigma, beta, inhibitionFactor,
                                                             highthresh, nargout=2)
            list_of_image_paths.append(corfresponse)

    return np.array(list_of_image_paths)


def temp_feature_maps(dataset):

    """

    Function to return an list of temperature feature maps

    :param dataset: Folder with class name and all the thermal images

    :return: List of the response map of the temperature in an array form

    """

    list_of_inpainted_temp_map = []

    m.addpath('.\src\Temp_extraction', nargout=0)
    m.addpath('.\src\preprocessing', nargout=0)

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for image in img_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + image
            temp_map = m.temp_segmentation(retrieve_dir)
            temp_map = np.array(temp_map)
            temp_map = cv2.resize(temp_map, (224,224))
            list_of_inpainted_temp_map.append(temp_map)

    return np.array(list_of_inpainted_temp_map)


def rgb_msx_feature_map(model, model_name, counter):

    """

      Function to return an models feature maps

      :param model_name: Name of the trained model
      :param model: Trained Convolutional neural network model
      :param counter: Number of folds

      :return: List of the response map of the temperature in an array form

    """

    feature_map = Model(model.input,
                        model.layers[-2].output)
    feature_map.save("./feature_maps/" + model_name + "_" + str(counter) + ".h5")

    return feature_map


def normalization(feature_maps):

    """

    Function to return an list of scaled feature maps

    :param feature_maps: List of feature maps

    :return: List of scaled feature maps in an array form

    """

    list_scaled_feature_maps = []

    for i in range(0, len(feature_maps)):
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_feature_maps = min_max_scaler.fit_transform(feature_maps[i])
        list_scaled_feature_maps.append(scaled_feature_maps)

    return np.array(list_scaled_feature_maps)


def feature_stack(feature_maps_one, feature_map_two, feature_map_three):

    """

    Function to return an list of stacked feature maps

    :param feature_maps_one: List of feature maps of first channel
    :param feature_map_two: List of feature maps of second channel
    :param feature_map_three: List of feature maps of thrid channel

    :return: List of concatenate feature maps in an array form

    """

    list_concatenate = []

    for i in range(0, len(feature_maps_one)):
        concatenate_features = np.stack([feature_maps_one[i], feature_map_two[i],
                                         feature_map_three[i]],
                                        axis=-1)
        list_concatenate.append(concatenate_features)

    return np.array(list_concatenate)


def feature_fusion(feature_maps_one, feature_map_two):

    """

    Function to return an list of fused feature maps

    :param feature_maps_one: Feature maps of model 1
    :param feature_map_two: Feature maps of model 2

    :return: Fused feature maps in an array form

    """

    features_fused = np.concatenate((feature_maps_one, feature_map_two), axis=1)

    return features_fused
