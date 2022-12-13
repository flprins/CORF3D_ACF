import os

import cv2
import numpy as np
from sklearn import preprocessing


def load_images(dataset, resize):
    """
    Function to return a list of images and its labels

    :param resize: resize the image
    :param dataset: Folder with class name and all the images in the dataset
    :return: Lists of image in an array form
    """

    labels = []
    images = []
    labels_list = []  # All unique labels
    filenames = []

    print("Reading images...")
    for file in os.listdir(dataset):
        if os.path.splitext(file)[1] != ".png":
            continue
        full_path = os.path.join(dataset, file)
        filename_elements = file.split('_')
        label = f'{filename_elements[1]}_{filename_elements[2]}'  # frogX_tankX
        img = cv2.imread(full_path, 3)
        img = cv2.resize(img, (resize, resize))
        images.append(img)
        if label not in labels_list:
            labels_list.append(label)
        labels.append(label)
        filenames.append(file)

    print("Done reading images.")

    return np.array(images), labels, labels_list, filenames


def load_corf_arrays(dataset, resize):
    """
    Function to return an list of CORF arrays and its labels

    :param resize: resize the image
    :param dataset: Folder with class name and all the images in the dataset
    :return: Lists of image in an array form
    """

    labels = []
    corf_arrays = []
    labels_list = []  # All unique labels
    filenames = []

    corf_dict = np.load(dataset)
    for filename, data in corf_dict.items():
        if data.dtype == "uint8":
            img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor((data*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (resize, resize))
        # img = np.zeros([*data.shape, 3])
        # img[:, :, 0] = data*255.0
        # img[:, :, 1] = data*255.0
        # img[:, :, 2] = data*255.0
        corf_arrays.append(img)
        filename_elements = filename.split('_')
        label = f'{filename_elements[1]}_{filename_elements[2]}'  # frogX_tankX
        if label not in labels_list:
            labels_list.append(label)
        labels.append(label)
        filenames.append(filename)

    return np.array(corf_arrays), labels, labels_list, filenames


def load_feature_maps(dataset, resize):
    """
    Function to return an list of feature maps and its labels

    :param resize: resize the feature maps
    :param dataset: Folder with class name and all the feature maps in the dataset
    :return: Lists of feature maps in an array form
    """

    feature_data_list = []
    labels = []
    list_of_feature_paths = []
    labels_list = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        feature_list = os.listdir(dataset + '/' + folder_name)
        for feature_maps in feature_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + feature_maps
            feature_maps = np.load(retrieve_dir)
            images = cv2.resize(feature_maps, (resize, resize))
            list_of_feature_paths.append(images)
        feature_data_list.append(feature_list)
        labels_list.append(folder_name)
        labels.append([folder_name] * len(feature_list))

    return np.array(list_of_feature_paths), labels, labels_list


def binarize_labels(labels):
    """
    Function to return an list of binarized labels

    :param labels: List of labels
    :return: Lists of binarized labels
    """

    flattened_list = np.asarray(labels, dtype="str")
    lb = preprocessing.LabelBinarizer().fit(flattened_list)
    labels = lb.transform(flattened_list)

    return labels
