#include "device_launch_parameters.h"
#include <iostream>
using namespace std;

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int i=0;i<deviceCount;i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        cout << "使用GPU device " << i << ": " << devProp.name << endl;
        cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << endl;
        cout << "SM的数量：" << devProp.multiProcessorCount << endl;
        cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << endl;
        cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << endl;
        cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << endl;
        cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << endl;
        cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << endl;
        cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << endl;
        cout << "======================================================" << endl;
        cout << "最高频率：" << devProp.clockRate / 1000000.0 << "GHz" << endl;
        cout << "显存位宽：" << devProp.memoryBusWidth << "bit" << endl;
        cout << "显存频率：" << devProp.memoryClockRate / 1000000.0 << "GHz" << endl;
    }
    return 0;
}