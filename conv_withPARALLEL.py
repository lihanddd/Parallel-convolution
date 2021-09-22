import numpy as np
import threading
from time import time
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
class Conv2d_plus:
    def __init__(self, in_plains, out_plains, kernel_size=(3, 3), stride=1, padding=0):
        kernel_param_num = kernel_size[0] * kernel_size[1]
        # self.kernel = np.ones(kernel_size) / kernel_param_num
        self.kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        self.kernel_size = kernel_size
        self.in_plains = in_plains
        self.out_plains = out_plains
        self.stride = stride
        self.padding = padding
        self.H = 0
        self.W = 0
        self.C = out_plains
    def sub_conv(self,img_padded,img_new,channel):
        for i in range(self.H):
                    for j in range(self.W):
                        i_ = i*self.stride
                        j_ = j*self.stride
                        a = img_padded[channel, i_:i_ +
                                       self.kernel_size[0], j_:j_+self.kernel_size[1]]
                        img_new[channel, i, j] = np.sum(a*self.kernel)
    def convolution(self, image_batch):
        # start_time=time()
        n, c, h, w = image_batch.shape
        self.H = (h - self.kernel_size[0] + 2*self.padding)//self.stride + 1
        self.W = (w - self.kernel_size[1] + 2*self.padding)//self.stride + 1
        tensor = np.zeros([n, self.C, self.H, self.W])
        for num in range(n):
            img = image_batch[num]
            h_ = h+2*self.padding
            w_ = w+2*self.padding
            img_padded = np.zeros([c, h_, w_])
            img_padded[:, self.padding:h_-self.padding,
                       self.padding:w_-self.padding] = img
            global img_new 
            img_new = np.zeros([self.C, self.H, self.W])
            threads=[]
            for channel in range(c):
                t=MyThread(func=self.sub_conv,args=(img_padded,img_new,channel))
                threads.append(t)
            for thread in threads:
                thread.start()
            for thread in threads:
                threading.Thread.join(thread)
            tensor[num] = img_new
        return tensor
