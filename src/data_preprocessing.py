import matlab.engine
import matplotlib.pyplot as plt

m = matlab.engine.start_matlab()
m.addpath('.\CORFpushpull', nargout=0)
data = m.imread('D:\Amey\Masters\Projects\CORF3D_HCR\data\processed\Preprocessed_RGB\\0086\\0086_1_1.jpg')
binarymap, corfresponse = m.CORFContourDetection(data, 2.2, 4, 1.8, 0.005, nargout=2)

plt.imshow(corfresponse)
plt.show()
