import os

import cv2
import numpy as np
from sklearn import preprocessing

from src.feature_maps import feature_stack


def filename2label(filename: str):
    filename_elements = filename.split('_')
    return f'{filename_elements[1]}_{filename_elements[2]}'  # frogX_tankX


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
        img = cv2.imread(full_path, 3)
        img = cv2.resize(img, (resize, resize))
        images.append(img)

        label = filename2label(file)
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
    :param dataset: Folder containing CORF datasets and labels as .npz
    :return: Lists of image in an array form
    """

    corf_0 = np.load(os.path.join(dataset, "corf_0.0.npz"))['arr_0']
    corf_1 = np.load(os.path.join(dataset, "corf_1.8.npz"))['arr_0']
    corf_2 = np.load(os.path.join(dataset, "corf_3.6.npz"))['arr_0']

    corf_stack = feature_stack(corf_0, corf_1, corf_2)

    labels = []
    labels_list = []
    filenames = np.load(os.path.join(dataset, "labels.npz"))['arr_0']
    for filename in filenames:
        label = filename2label(filename)
        if label not in labels_list:
            labels_list.append(label)
        labels.append(label)

    return corf_stack, labels, labels_list, filenames


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
    Function to return a list of binarized labels

    :param labels: List of labels
    :return: Lists of binarized labels
    """

    flattened_list = np.asarray(labels, dtype="str")
    lb = preprocessing.LabelBinarizer().fit(flattened_list)
    labels = lb.transform(flattened_list)

    return labels, lb
