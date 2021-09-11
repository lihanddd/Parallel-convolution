from get_img import *
from mpi4py import MPI
import numpy as np
from conv import Conv2d
from mysum import *
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':

    comm.barrier()
    start = MPI.Wtime()

    if rank == 0:
        file_ = load_Img("../images")
        file_list = div_list(file_, size)
    else:
        file_list = None

    file_batch = comm.scatter(file_list, root=0)
    image_batch = load_img_batch(file_batch)

    comm.barrier()
    end = MPI.Wtime()
    load_img_time = end-start

    comm.barrier()
    start = MPI.Wtime()

    n, c, h, w = image_batch.shape
    conv_batch = Conv2d(c, 3)
    tensor_batch = conv_batch.convolution(image_batch)

    commute = True
    myop = MPI.Op.Create(mysum, commute)
    recv_obj = comm.reduce(tensor_batch, op=myop, root=0)
    myop.Free()

    comm.barrier()
    end = MPI.Wtime()
    convolution_time = end-start
    if rank == 0:
        print("loading images use time:", load_img_time, "s")
        print("convolution use time:", convolution_time, "s")
        print("total time:", load_img_time+convolution_time)
        print(recv_obj.shape)
        img = recv_obj[0].transpose((1, 2, 0))
        cv2.imshow("img",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
