import matlab.engine
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
m = matlab.engine.start_matlab()
m.addpath('.\src\Preprocessing', nargout=0)
m.addpath('.\src\Temp_extraction', nargout=0)


def batch_data_preprocessing(dataset_1, dataset_2, dataset_3, feature_map):

    """

    Function to return an list of inpainted images

    :param dataset_3: Path to the folder where the preprocessed images need to be stored
    :param dataset_2: Folder with class name and RGB/MSX images
    :param dataset_1: Folder with class name and Thermal images
    :param feature_map: Feature map to be pre-processed
    :param dataset: Folder with class name and all the pre-segmentation images containing RGB/MSX
    and thermal images

    :return: list of inpainted images in an array form and labels

    """

    list_of_inpainted_images = []
    labels = []
    labels_list = []


    data_dir_list = os.listdir(dataset_1)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset_1 + '/' + folder_name)
        for idx in range(0, len(img_list)):
            rgb_image_idx = img_list[idx]
            rgb_image_path = dataset_2 + "/" + folder_name + "/" + rgb_image_idx
            thermal_image_idx = img_list[idx]
            thermal_image_path = dataset_1 + "/" + folder_name + "/" + thermal_image_idx
            thermal_image = m.imread(thermal_image_path)
            rgb_image = m.imread(rgb_image_path)
            if feature_map == "RGB":
                scratched_image, rod_mask = m.segmentation(thermal_image_path, rgb_image,
                                                           nargout=2)
            elif feature_map == "MSX":
                scratched_image, rod_mask = m.segmentation(thermal_image, thermal_image, nargout=2)
            scratched_image = np.array(scratched_image)
            scratched_image = scratched_image.astype('uint8')
            rod_mask = np.array(rod_mask)
            rod_mask = rod_mask.astype('uint8')
            inpainted_image = cv2.inpaint(scratched_image, rod_mask, 3, cv2.INPAINT_TELEA)
            inpainted_image = cv2.resize(inpainted_image, (224, 224))
            preprocessed_path = dataset_3
            if os.path.exists(dataset_3 + "/" + folder_name):
                class_folder_path = dataset_3 + "/" + folder_name
                cv2.imwrite(class_folder_path + "/" + str(rgb_image_idx), inpainted_image)
            else:
                if os.path.exists(preprocessed_path):
                    os.mkdir(dataset_3 + "/" + folder_name)
                    class_folder_path = dataset_3 + "/" + folder_name
                    cv2.imwrite(
                        class_folder_path + "/" + str(rgb_image_idx), inpainted_image)
                else:
                    os.mkdir(preprocessed_path)
                    os.mkdir(dataset_3 + "/" + folder_name)
                    class_folder_path = dataset_3 + "/" + folder_name
                    cv2.imwrite(
                        class_folder_path + "/" + str(rgb_image_idx), inpainted_image)
            list_of_inpainted_images.append(inpainted_image)
        labels_list.append(folder_name)
        labels.append([folder_name] * len(img_list))

    return np.array(list_of_inpainted_images), labels, labels_list
