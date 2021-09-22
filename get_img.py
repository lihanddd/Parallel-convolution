import os
from mpi4py import MPI
import numpy as np
import cv2
IMAGE_H = 200
IMAGE_W = 300
def div_list(ls, n):
    if not isinstance(ls, list) or not isinstance(n, int):
        return []
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return []
    if n > ls_len:
        return []
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len//n
        k = ls_len % n
        # j,j,j,...(前面有n-1个j),j+k
        # 步长j,次数n-1
        ls_return = []
        for i in range(0, (n-1)*j, j):
            ls_return.append(ls[i:i+j])
        # 算上末尾的j+k
        ls_return.append(ls[(n-1)*j:])
        return ls_return
def load_Img(filedir):
    file_ = []
    for fname in os.listdir(filedir):
        file_path = filedir + '/' + fname
        file_.append(file_path)
    return file_
def load_img_batch(img_name_list):
    num = len(img_name_list)
    img_list = np.zeros((num, 3, IMAGE_H, IMAGE_W))
    for i in range(num):
        img = cv2.imread(img_name_list[i])
        if img is None:
            continue
        img_ = np.resize(img, (IMAGE_H, IMAGE_W, 3))/255
        img_ = img_.transpose((2, 0, 1))
        img_list[i] = (img_)
    return img_list
