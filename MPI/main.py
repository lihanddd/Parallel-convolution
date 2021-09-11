from get_img import *
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start = time.process_time()

file_ = load_Img("./images")
file_list = div_list(file_, size)
file_batch = file_list[rank]
image_batch = load_img_batch(file_batch)

end = time.process_time()
if rank == 0:
    print("image loaded")
    print("use time:",end-start,"s")
