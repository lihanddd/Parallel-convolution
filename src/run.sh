g++ -o squ_conv ./sequential/matrix_add.cpp 

./squ_conv


nvcc -o gpu_info ./parallel/outputGPUinfo.cu -run


nvcc -o cuda_conv ./parallel/convolution.cu  -run