from matplotlib import pyplot as plt
import numpy
import os
cov_time=[260.40123319625854, 252.17480206489563, 256.3235068321228, 271.26123309135437, 257.0146381855011, 244.69290804862976, 238.45690393447876, 238.11059308052063]
img_time=[]
# with open("/Users/puyuandong613/Downloads/Parallel-convolution/Parallel-convolution-Thread/cov_time") as f1:
#     for f in f1:
#         for num in f:
#             cov_time.append(float(num))
#     f1.close()
# with open("/Users/puyuandong613/Downloads/Parallel-convolution/Parallel-convolution-Thread/img_time") as f2:
#     for f in f2:
#         img_time=f
#     f2.close()
plt.xlabel("number of p")
plt.ylabel("Sp")
x=[1,2,4,8,16,32,64,128]
y=[]
for i in range(len(cov_time)):
    y.append(cov_time[0]/cov_time[i])
plt.plot(x,y)
plt.show()