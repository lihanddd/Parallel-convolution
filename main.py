from get_img import *
import numpy as np
from conv import Conv2d
from mysum import *
import time
from time import ctime
import threading
import cv2
import matplotlib
import tqdm
from conv_withPARALLEL import Conv2d_plus
IMAGE_H = 200
IMAGE_W = 300
class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        # time.sleep(2)
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None
if __name__ == '__main__':
    file_ = load_Img("/Users/puyuandong613/Downloads/Parallel-convolution/Parallel-convolution-Thread/images")
    start_imread1=time.time()
    #非并行化读文件
    img_batches1= np.zeros((len(file_), 3, IMAGE_H, IMAGE_W))
    for i in range(len(file_)):
        img = cv2.imread(file_[i])
        if img is None:
            continue
        img_ = np.resize(img, (IMAGE_H, IMAGE_W, 3))/255
        img_ = img_.transpose((2, 0, 1))
        img_batches1[i]=(img_)
    end_imread1=time.time()
    print("without parallel: imread time",end_imread1-start_imread1)
    # # 非并行卷积
    conv2d=Conv2d(in_plains=3,out_plains=3)
    # start_imgConv1=time.time()
    # rec_imgs1=[]
    # # for index in range(len(img_batches1)):
    # rec_img=conv2d.convolution(img_batches1)
    # rec_imgs1.append(rec_img)
    # end_imgConv1=time.time()
    # print("without parallel: conv time",end_imgConv1-start_imgConv1)
    #并行化读文件
    img_save_dir="/Users/puyuandong613/Downloads/Parallel-convolution/Parallel-convolution-Thread/img_time"
    cov_save_dir="/Users/puyuandong613/Downloads/Parallel-convolution/Parallel-convolution-Thread/cov_time"
    img_dir="/Users/puyuandong613/Downloads/Parallel-convolution/Parallel-convolution-Thread/convoluted_images/"
    img_time=[]
    cov_time=[]
    process_nums=[1,2,3,4,5,6,7,8]
    conv2d_plus=Conv2d_plus(in_plains=3,out_plains=3)
    for process_num in tqdm.tqdm(process_nums):
        threads=[]
        img_batches2=[]
        print("process_num:", process_num)
        file_list = div_list(file_, process_num)
        start_imread2=time.time()
        for index in range(len(file_list)):
            t = MyThread(
            func=load_img_batch,args=(file_list[index],)
            )
            # print(file_list[index])
            threads.append(t)
        for thread in threads:
            thread.start()
        for thread in threads:
            threading.Thread.join(thread)
            # thread.join()
        for thread in threads:
            img_batches2.append(thread.result)
        end_imread2=time.time()
        print("with parallel: imread time",end_imread2-start_imread2)
        img_time.append(end_imread2-start_imread2)
 #并行卷积
        start_imgConv2=time.time()
        rec_imgs2=[]
        threads2=[]
        for i in range(process_num):
            # t=MyThread(func=conv2d.convolution,args=(img_batches2[i],))
            t=MyThread(func=conv2d_plus.convolution,args=(img_batches2[i],))
            threads2.append(t)
        for thread in threads2:
            thread.start()
        for thread in threads2:
            thread.join()
        for thread in threads2:
            rec_imgs2.append(thread.get_result())
        end_imgConv2=time.time()
        print("with parallel: conv time", end_imgConv2-start_imgConv2)
        cov_time.append(end_imgConv2-start_imgConv2)

with open(img_save_dir,"a") as img_f:
    img_f.write(str(img_time)+"\n")
    img_f.close()
with open(cov_save_dir,'a') as cov_f:
    cov_f.write(str(cov_time)+"\n")
    cov_f.close()