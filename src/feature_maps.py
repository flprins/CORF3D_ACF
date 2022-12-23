import os

# import matlab.engine
import numpy as np
import cv2
from sklearn import preprocessing

# m = matlab.engine.start_matlab()


def corf_feature_map(image_path, sigma, beta, inhibition_factor, highthresh):

    """

    Function to return an list of CORF feature maps

    :param image_path: File path to preprocessed RGB image
    :param sigma: The standard deviation of the DoG functions used
    :param beta: The increase in the distance between the sets of center-on
                 and center-off DoG receptive fields
    :param inhibition_factor: The factor by which the response exhibited in the
                             inhibitory receptive field suppresses the response exhibited in the
                             excitatory receptive field
    :param highthresh: The high threshold of the non-maximum suppression used for binarization

    :return: List of the response map of the rotation-invariant CORF operator (CORF feature maps) in an array form

    """

    m.addpath(os.path.join(os.path.dirname(__file__), "CORFpushpull"), nargout=0)

    img = m.imread(image_path)
    binarymap, corfresponse = m.CORFContourDetection(img, sigma, beta, inhibition_factor, highthresh, nargout=2)
    return np.array(corfresponse)


def corf_feature_maps(dataset_path, sigma, beta, inhibitionFactor, highthresh):

    """

    Function to return an list of CORF feature maps

    :param dataset_path: Folder with class name and all the pre-processed images
    :param sigma: The standard deviation of the DoG functions used
    :param beta: The increase in the distance between the sets of center-on
                 and center-off DoG receptive fields
    :param inhibitionFactor: The factor by which the response exhibited in the
                             inhibitory receptive field suppresses the response exhibited in the
                             excitatory receptive field
    :param highthresh: The high threshold of the non-maximum suppression used for binarization

    :return: List of the response map of the rotation-invariant CORF operator (CORF feature maps) in an array form

    """

    response_maps = []

    m.addpath(os.path.join(os.path.dirname(__file__), "CORFpushpull"), nargout=0)

    n_files = len(os.listdir(dataset_path))
    log_steps = np.linspace(0, n_files, 11, dtype=np.uint)[1:-1]
    log_step = 0
    for i, image in enumerate(os.listdir(dataset_path)):
        if os.path.splitext(image)[1].lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        if i == log_steps[log_step]:
            print(f'Done {i} out of {n_files}...')
            log_step += 1
        retrieve_dir = os.path.join(dataset_path, image)
        image = m.imread(retrieve_dir)
        binarymap, corfresponse = m.CORFContourDetection(image, sigma, beta, inhibitionFactor,
                                                         highthresh, nargout=2)
        response_maps.append(corfresponse)

    return np.array(response_maps)


def temp_feature_maps(dataset):

    """

    Function to return an list of temperature feature maps

    :param dataset: Folder with class name and all the thermal images

    :return: List of the response map of the temperature in an array form

    """

    list_of_inpainted_temp_map = []
    labels = []
    labels_list = []

    m.addpath(os.path.join(__file__, "src", "Temp_extraction"), nargout=0)
    m.addpath(os.path.join(__file__, "src", "Preprocessing"), nargout=0)

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for image in img_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + image
            temp_map = m.temp_segmentation(retrieve_dir)
            temp_map = np.array(temp_map)
            temp_map = cv2.resize(temp_map, (224, 224))
            list_of_inpainted_temp_map.append(temp_map)
        labels_list.append(folder_name)
        labels.append([folder_name] * (len(img_list)))

    return np.array(list_of_inpainted_temp_map), labels_list, labels


def feature_normalization(feature_maps):

    """

    Function to return a list of scaled feature maps

    :param feature_maps: List of feature maps

    :return: List of scaled feature maps in an array form

    """

    list_scaled_feature_maps = []

    for feature_map in feature_maps:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled_feature_maps = min_max_scaler.fit_transform(feature_map)
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

    for i in range(len(feature_maps_one)):
        concatenate_features = np.stack([feature_maps_one[i], feature_map_two[i], feature_map_three[i]], axis=-1)
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
