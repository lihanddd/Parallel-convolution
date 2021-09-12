#include <iostream>
#include <cstdlib>
#include <sys/time.h>
using namespace std;

void conv2d(double **array, double **kernel, double **result, int array_size, int kernel_size)
{
    if (array_size < kernel_size)
        return;
    for (int i = 0; i < array_size - kernel_size + 1; i++)
        for (int j = 0; j < array_size - kernel_size + 1; j++) {
            result[i][j] = 0;
            for (int n = 0; n < kernel_size; n++)
                for (int m = 0; m < kernel_size; m++)
                    result[i][j] += array[i + n][j + m] * kernel[n][m];
        }
}


__global__  void conv2d_cuda(double **array, double **kernel, double **result, int array_size, int kernel_size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int result_size = array_size - kernel_size + 1;

    if (idx < result_size && idy < result_size) {
        result[idx][idy] = 0;
        for (int i = 0; i < kernel_size; i++)
            for (int j = 0; j < kernel_size; j++)
                result[idx][idy] += array[idx + i][idy + j] * kernel[i][j];
    }
}


void testConvTime(int array_size, int kernel_size)
{
    double **a, **res_cpu, *res_gpu, **kernel, *a_line, *kernel_line;
    
    // allocate memory for array
    int result_size = array_size - kernel_size + 1;
    a = new double *[array_size];
    res_cpu = new double *[result_size];
    res_gpu = new double [result_size * result_size];
    kernel = new double *[kernel_size];
    a_line = new double [array_size * array_size];
    kernel_line = new double [kernel_size * kernel_size];

    for (int i = 0; i < array_size; i++)
        // a[i] = (double *)malloc(array_size * sizeof(double));
        a[i] = new double[array_size];
    for (int i = 0; i < result_size; i++) {
        // res_cpu[i] = (double *)malloc(result_size * sizeof(double));
        res_cpu[i] = new double[result_size];
    }
    for (int i = 0; i < kernel_size; i++) {
        // kernel[i] = (double *)malloc(kernel_size * sizeof(double));
        kernel[i] = new double [kernel_size];
    }
    for (int i = 0; i < array_size; i++)
        for (int j = 0; j < array_size; j++) {
            a[i][j] = 1.0;
            a_line[i + j * array_size] = 1.0;
        }
    for (int i = 0; i < kernel_size; i++)
        for (int j = 0; j < kernel_size; j++) {
            kernel[i][j] = 1.0;
            kernel_line[i + j * kernel_size] = 1.0;
        }

    double **d_a, **d_res_gpu, **d_kernel, *d_data_a, *d_data_res_gpu, *d_data_kernel, **temp;
    temp = new double *[array_size];

    cudaMalloc((void**)&d_data_a, array_size * array_size * sizeof(double));
    cudaMalloc((void**)&d_data_res_gpu, result_size * result_size * sizeof(double));
    cudaMalloc((void**)&d_data_kernel, kernel_size * kernel_size * sizeof(double));
    cudaMalloc((void**)&d_a, array_size * sizeof(double *));
    cudaMalloc((void**)&d_res_gpu, result_size * sizeof(double*));
    cudaMalloc((void**)&d_kernel, kernel_size * sizeof(double*));

    for (int i = 0; i < array_size; i++)
        temp[i] = d_data_a + i * array_size;
    cudaMemcpy(d_a, temp, array_size * sizeof(double*), cudaMemcpyHostToDevice);
    for (int i = 0; i < result_size; i++)
        temp[i] = d_data_res_gpu + i * result_size;
    cudaMemcpy(d_res_gpu, temp, result_size * sizeof(double*), cudaMemcpyHostToDevice);
    for (int i = 0; i < kernel_size; i++)
        temp[i] = d_data_kernel + i * kernel_size;
    cudaMemcpy(d_kernel, temp, kernel_size * sizeof(double*), cudaMemcpyHostToDevice);

    cudaMemcpy(d_data_a, a_line, array_size * array_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data_kernel, kernel_line, kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blocksize(32, 32);
    dim3 gridsize((result_size + blocksize.x - 1) / blocksize.x, (result_size + blocksize.x - 1) / blocksize.x);
    // cout << gridsize.x << " " << gridsize.y << endl;

    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;

    // test cpu convolution time
    gettimeofday( &cpu_start, NULL );
    conv2d(a, kernel, res_cpu, array_size, kernel_size);
    gettimeofday(&cpu_end, NULL);
    cout << "cpu convolution time: " << (cpu_end.tv_sec - cpu_start.tv_sec) * 1000 + (cpu_end.tv_usec - cpu_start.tv_usec) / 1000.0 << "ms" << endl;

    // test gpu convolution time
    gettimeofday( &gpu_start, NULL );
    conv2d_cuda <<<gridsize, blocksize>>> (d_a, d_kernel, d_res_gpu, array_size, kernel_size);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaDeviceSynchronize();
    gettimeofday(&gpu_end, NULL);
    cout << "gpu convolution time: " << (gpu_end.tv_sec - gpu_start.tv_sec) * 1000 + (gpu_end.tv_usec - gpu_start.tv_usec) / 1000.0 << "ms" << endl;

    cudaMemcpy(res_gpu, d_data_res_gpu, result_size * result_size * sizeof(double), cudaMemcpyDeviceToHost);

    double residual = 0;
    for (int i = 0; i < result_size * result_size; i++) {
        residual += (res_cpu[i / result_size][i % result_size] - res_gpu[i]);
    }
    cout << "residual of cpu and gpu: " << residual << endl;

    for (int i = 0; i < array_size; i++)
        delete a[i];
    delete a;
    for (int i = 0 ; i < result_size; i++)
        delete res_cpu[i];
    delete res_cpu;
    delete res_gpu;
    for (int i = 0; i < kernel_size; i++)
        delete kernel[i];
    delete kernel;
    delete a_line;
    delete kernel_line;

    cudaFree(d_data_res_gpu);
    cudaFree(d_data_kernel);
    cudaFree(d_a);
    cudaFree(d_res_gpu);
    cudaFree(d_kernel);
    cudaFree(d_data_a);
    delete(temp);
}

int main()
{
    int array_size[5] = {512, 1024, 2048, 4096, 8192}, kernel_size[5] = {3, 5, 9, 12, 15};
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) {
            cout << "===========array size: " << array_size[i] << " kernel size: " << kernel_size[j] << "============" << endl;
            testConvTime(array_size[i], kernel_size[j]);
        }
}