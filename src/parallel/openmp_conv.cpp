#include <iostream>
#include <omp.h>

#define W 1920
#define H 1080

using namespace std;
 
int Limit(int num)
{
	if (num < 0)
		num = 0;
	else if (num > 255)
		num = 255;
 
	return num;
}

float imageKernel[3][3] = {0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111};



int main()
{	char *name[3] = {"1.yuv","2.yuv","3.yuv"};
	char *tar[3] = {"conv_1.yuv","conv_2.yuv","conv_3.yuv"};
	FILE *fp[3];

	double t1 = omp_get_wtime();

	#pragma omp parallel for
	for(int i = 0; i < 3; i++){
		fp[i] = fopen(name[i], "rb");
	}
	
	for(int i = 0; i < 3; i++){
		fseek(fp[i], 0, SEEK_END);
		int len = ftell(fp[i]);
		fseek(fp[i], 0, SEEK_SET);
	
		unsigned char* inputImage = new unsigned char[len]();
		unsigned char* outputImage = new unsigned char[len]();
		fread(inputImage, 1, len, fp[i]);
		fclose(fp[i]);
	
		memcpy(outputImage, inputImage, len);
		


		#pragma omp parallel for
		for (int i = 1; i < H - 1; i++)
		{   
			// int thread_num = omp_get_thread_num();
			// cout << "thread num: " << thread_num << endl;
			for (int j = 1; j < W - 1; j++)
			{   
				float sum = 0;
				for(int a = -1; a < 2; a++){
					for(int b = -1; b < 2; b++){
						sum += inputImage[(i+a)*W+j+b]*imageKernel[a+1][b+1];
					}
				}
				int num = Limit(int(sum));
	
				outputImage[i*W + j] = num;
				
			}
		}
		
	
		fp[i] = fopen(tar[i], "wb+");
		fwrite(outputImage, 1, len, fp[i]);
		fclose(fp[i]);
	
	}
	double t2 = omp_get_wtime();
	cout <<"并行时间:"<< t2-t1 << endl;

    system("pause");
	return 0;
}