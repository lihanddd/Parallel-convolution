from matplotlib import pyplot as plt
import numpy
import os

from numpy.lib.npyio import save
cov_time=[260.1444070339203, 255.93577766418457, 256.12489914894104, 257.5032413005829, 257.9831428527832, 254.40561985969543, 261.99464321136475, 261.0039863586426]
img_time=[]
save_path="/Users/puyuandong613/Downloads/Parallel-convolution/Parallel-convolution-Thread/images_of_cov"
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
x=[1,2,3,4,5,6,7,8]
y=[]
for i in range(len(cov_time)):
    y.append(cov_time[0]/(cov_time[i]))
plt.plot(x,y)
plt.savefig(save_path)
plt.show()

