import matlab.engine
from imageio import imread

m = matlab.engine.start_matlab()
m.CORFContourDetection(imread('D:\Amey\Masters\Projects\CORF3D_HCR\data\processed'
                               '\Preprocessed_RGB\0086\0086_1_1.jpg'), 2.2,4,1.8,0.005)