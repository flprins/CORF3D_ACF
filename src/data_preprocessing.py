import matlab.engine
import numpy as np
import os

import matplotlib.pyplot as plt

m = matlab.engine.start_matlab()
m.addpath('.\CORFpushpull', nargout=0)

img_data_list = []
labels = []
list_of_image_paths = []
labels_list = []

data_dir_list = os.listdir("D:\Amey\Masters\Projects\CORF3D_HCR\data\processed\Preprocessed_RGB")
for folder_name in data_dir_list:
    img_list = os.listdir("D:\Amey\Masters\Projects\CORF3D_HCR\data\processed\Preprocessed_RGB" + '/' + folder_name)
    for image in img_list:
        retrieve_dir = "D:\Amey\Masters\Projects\CORF3D_HCR\data\processed\Preprocessed_RGB" + "/" + folder_name + "/" + image
        print("Reading image"+retrieve_dir)
        images = m.imread(retrieve_dir)
        binarymap, corfresponse = m.CORFContourDetection(images, 2.2, 4, 1.8, 0.005, nargout=2)
        list_of_image_paths.append(corfresponse)
        print(np.array(list_of_image_paths).shape)
        plt.imshow(corfresponse)
        plt.show()


