#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/cl.h"
#include <windows.h>
#include "read_source.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <string>
#include <windows.h>

using namespace std;

float GetMean(float * number, int count);
float GetStandardDeviation(float *number, float average, int count);

int main(int argc, char** argv)
{
	//add declarations for necessary OpenCl API variables
	cl_platform_id *platforms = NULL;
	cl_uint num_platforms = 0;
	cl_device_id *device_list = NULL;
	cl_uint num_devices = 0;
	char *name = NULL;
	char *device_name = NULL;
	size_t size;
	size_t device_size;
	size_t file_size;
	cl_int err;

	//image setting
	cl_mem InputImage;
	cl_mem DownSampleImage;
	cl_mem GaussianBlurImage;
	cl_sampler ImgSampler;
	cl_mem Filter;
	cl_mem FSum;
	cl_mem Sigma;
	cl_mem DoGImage;
	cl_mem Extrema;
	cl_mem Index;
	int width, height, channels;

	float sigma[5] = {0.7,1.4,2.1,2.8,3.5};
	cl_int sizex = 1;//downsample size
	cl_int sizey = 1;
	cl_int filterWidth = 5/sizex;
	cl_int filterSize = filterWidth * filterWidth*5;
	int * idx = (int*)_aligned_malloc(sizeof(int)*1, 4096);
	float * gaussBlurFilter = (float*)_aligned_malloc(sizeof(float)*filterSize, 4096);
	float * filtsum = (float*)_aligned_malloc(sizeof(float)*filterSize, 4096);
	//filtsum[0] = 0;
	cl_uint2 extremapoints[100];
	size_t filterworksize[] = { filterWidth*filterWidth,5,0 };
	size_t filterlocalworksize[] = { filterWidth*filterWidth,1,0 };
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	string fileIdx[] = { "0","1","2","3","4","5" };
	cl_ulong start_time;
	cl_ulong end_time;
	int iteration = 1;
	float time_ave = 0;
	float time_standard_deviation = 0;
	float time_series[100];

	cl_context context;
	cl_program program;
	cl_command_queue command;
	cl_kernel kernel[5] = {NULL, NULL,NULL,NULL,NULL};

	int dimention = 2;
	char *KernelSource = read_source("device.cl", &file_size);

	unsigned char *indata = NULL;
	unsigned char *outdata = NULL;

	string inputfile = "C:/SIFT/Final/images/512_512.png";
	string outputfile = "C:/SIFT/Final/images/";
	

	indata = stbi_load(inputfile.c_str(), &width, &height, &channels, 0);

	size_t globalworksize[] = { width,height,5 };
	size_t globalworksize2[] = { width / sizex,height / sizey,5 };
	size_t localworksize[] = { 1,1,1};
	size_t origin[] = { 0, 0, 0};
	size_t region[] = { width,height,1}; // 1 for 2D image
	size_t region2[] = { width / sizex, height / sizey, 1};

	cl_image_desc desc;
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = width;
	desc.image_height = height;
	desc.image_depth = 0;
	desc.image_array_size = 0;
	desc.image_row_pitch = 0;
	desc.image_slice_pitch = 0;
	desc.num_mip_levels = 0;
	desc.num_samples = 0;
	desc.buffer = NULL;
	cl_image_desc desc2;
	desc2.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc2.image_width = width / sizex;
	desc2.image_height = height / sizey;
	desc2.image_depth = 0;
	desc2.image_array_size = 0;
	desc2.image_row_pitch = 0;
	desc2.image_slice_pitch = 0;
	desc2.num_mip_levels = 0;
	desc2.num_samples = 0;
	desc2.buffer = NULL;
	cl_image_desc desc3;
	desc3.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
	desc3.image_width = width ;
	desc3.image_height = height;
	desc3.image_depth = 0;
	desc3.image_array_size = 5;
	desc3.image_row_pitch = 0;
	desc3.image_slice_pitch = 0;
	desc3.num_mip_levels = 0;
	desc3.num_samples = 0;
	desc3.buffer = NULL;
	cl_image_format format;
	format.image_channel_order = CL_RGBA;
	format.image_channel_data_type = CL_UNSIGNED_INT8;

	outdata = (unsigned char*)malloc(width*height*channels * sizeof(unsigned char));

	//Get platform
	clGetPlatformIDs(0, NULL, &num_platforms);//first call,get the num_platforms
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms);//memory allocation
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a platform group!\n");
		return EXIT_FAILURE;
	}

	//choose the first platform,i.e. intel(R)
	clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, NULL, NULL, &size);
	name = (char*)malloc(size);
	clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, size, name, NULL);
	printf(name);
	printf("\n");

	//get device information
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}

	//choose the first device
	clGetDeviceInfo(device_list[0], CL_DEVICE_NAME, NULL, NULL, &device_size);
	device_name = (char*)malloc(device_size);
	clGetDeviceInfo(device_list[0], CL_DEVICE_NAME, device_size, device_name, NULL);
	printf(device_name);
	printf("\n");

	//creat context
	context = clCreateContext(0, 1, &device_list[0], NULL, NULL, &err);
	if (!context)
	{
		printf("Error:Failed to creat a context!\n");
		return EXIT_FAILURE;
	}

	//create program
	program = clCreateProgramWithSource(context, 1, (const char **)& KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error:Failed to creat a program!\n");
		return EXIT_FAILURE;
	}

	//build program
	err = clBuildProgram(program, 1, &device_list[0], "-cl-std=CL2.0", NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to build program!\n");
		return EXIT_FAILURE;
	}

	//create command
	command = clCreateCommandQueueWithProperties(context, device_list[0], 0, NULL);
	if (!command)
	{
		printf("Error:Failed to creat a command queue!\n");
		return EXIT_FAILURE;
	}

	//create kernel
	kernel[0] = clCreateKernel(program, "DownSample", &err);
	if (!kernel[0])
	{
		printf("Error:Failed to creat kernel0!\n");
		return EXIT_FAILURE;
	}

	kernel[1] = clCreateKernel(program, "GaussianFilter", &err);
	if (!kernel[1])
	{
		printf("Error:Failed to creat kernel1!\n");
		return EXIT_FAILURE;
	}

	kernel[2] = clCreateKernel(program, "GaussianBlur", &err);
	if (!kernel[2])
	{
		printf("Error:Failed to creat kernel2!\n");
		return EXIT_FAILURE;
	}

	kernel[3] = clCreateKernel(program, "DoG", &err);
	if (!kernel[3])
	{
		printf("Error:Failed to creat kernel3!\n");
		return EXIT_FAILURE;
	}

	kernel[4] = clCreateKernel(program, "Extrema", &err);
	if (!kernel[4])
	{
		printf("Error:Failed to creat kernel4!\n");
		return EXIT_FAILURE;
	}

	//create input and output buffer
	InputImage = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, indata, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create inputimage!\n");
		return EXIT_FAILURE;
	}

	DownSampleImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc2, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create DownSampleImage!\n");
		return EXIT_FAILURE;
	}

	GaussianBlurImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc3, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create GaussianBlurImage!\n");
		return EXIT_FAILURE;
	}

	DoGImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc2, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create DoGImage!\n");
		return EXIT_FAILURE;
	}

	Extrema = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(cl_uint2) * 100, extremapoints, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer for Extrema!\n");
		return EXIT_FAILURE;
	}

	Filter = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(float)*filterSize, gaussBlurFilter, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	FSum = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(float)*filterSize, filtsum, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	Index = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(int)*1, idx, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	Sigma = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, sizeof(float)*5, sigma, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create buffer!\n");
		return EXIT_FAILURE;
	}

	ImgSampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_LINEAR, &err);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to create sampler!\n");
		return EXIT_FAILURE;
	}

	err = clEnqueueWriteImage(command, InputImage, CL_TRUE, origin, region, 0, 0, indata, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write image to device!\n");
		return EXIT_FAILURE;
	}

	//Set the kernel arguments

	err = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &DownSampleImage);
	err |= clSetKernelArg(kernel[0], 2, sizeof(cl_mem), &sizex);
	err |= clSetKernelArg(kernel[0], 3, sizeof(cl_mem), &sizey);
	err |= clSetKernelArg(kernel[0], 4, sizeof(cl_sampler), &ImgSampler);
	err |= clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &filterWidth);
	err |= clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &Sigma);
	err |= clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &Filter);
	err |= clSetKernelArg(kernel[1], 3, sizeof(cl_mem), &FSum);
	err |= clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &GaussianBlurImage);
	err |= clSetKernelArg(kernel[2], 2, sizeof(cl_sampler), &ImgSampler);
	err |= clSetKernelArg(kernel[2], 3, sizeof(cl_mem), &filterWidth);
	err |= clSetKernelArg(kernel[2], 4, sizeof(cl_mem), &Filter);
	err |= clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[3], 2, sizeof(cl_mem), &DoGImage); 
	err |= clSetKernelArg(kernel[3], 3, sizeof(cl_sampler), &ImgSampler);
	err |= clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[4], 1, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[4], 2, sizeof(cl_mem), &InputImage);
	err |= clSetKernelArg(kernel[4], 3, sizeof(cl_mem), &Extrema);
	err |= clSetKernelArg(kernel[4], 4, sizeof(cl_sampler), &ImgSampler);
	err |= clSetKernelArg(kernel[4], 5, sizeof(cl_mem), &Index);

	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel argument!\n");
		return EXIT_FAILURE;
	}

	QueryPerformanceFrequency(&frequency);

	for (int i = 0; i < iteration; i++)
	{
		QueryPerformanceCounter(&start);

		//Execute the kernel

		//Downsample
		err = clEnqueueNDRangeKernel(command, kernel[0], dimention, NULL, globalworksize, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}

		//Building Gaussian Matrix
		err = clEnqueueNDRangeKernel(command, kernel[1], 2, NULL, filterworksize, filterlocalworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}

		//Gaussian Blur
		err = clEnqueueNDRangeKernel(command, kernel[2], 3, NULL, globalworksize, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		//DoG Image
		err = clEnqueueNDRangeKernel(command, kernel[3], dimention, NULL, globalworksize, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		//Extrema Points
		err = clEnqueueNDRangeKernel(command, kernel[4], dimention, NULL, globalworksize, localworksize, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to enqueue NDRange Kernel!\n");
			return EXIT_FAILURE;
		}

		err = clFinish(command);
		if (CL_SUCCESS != err)
		{
			printf("Error: clFinish Failed!\n");
			return EXIT_FAILURE;
		}
		

		QueryPerformanceCounter(&end);

		time_series[i] = (float)(end.QuadPart - start.QuadPart) * 1000 / frequency.QuadPart;

	}

	printf("\n***** NDRange is finished ***** \n");

	time_ave = GetMean(time_series, iteration);
	time_standard_deviation = GetStandardDeviation(time_series, time_ave, iteration);
	printf("\tImageWidth: %d, ImageHeight: %d \n ", width, height);
	printf("\tAverage Execution Time: %f ms\n ", time_ave);
	printf("\tStandard Deviation of Time: %f\n ", time_standard_deviation);


	
	for (int i = 0; i < 5; i++)
	{
		size_t origin2[] = { 0, 0 ,i };
		err = clEnqueueReadImage(command, GaussianBlurImage, CL_TRUE, origin2, region, 0, 0, outdata, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to read image from device!\n");
			return EXIT_FAILURE;
		}
		string fileName="";
		fileName += fileIdx[i];
		fileName += ".jpg";
		string output = outputfile;
		output.append(fileName);
		stbi_write_jpg(output.c_str(), width, height, channels, outdata, 100);
	}
	


	//Release Memory
	clReleaseMemObject(InputImage);
	clReleaseMemObject(DownSampleImage);
	clReleaseMemObject(GaussianBlurImage);
	clReleaseMemObject(DoGImage);

	clReleaseProgram(program);
	clReleaseKernel(kernel[0]);
	clReleaseKernel(kernel[1]);
	clReleaseKernel(kernel[2]);
	clReleaseKernel(kernel[3]);
	clReleaseKernel(kernel[4]);
	clReleaseCommandQueue(command);
	clReleaseContext(context);

	stbi_image_free(indata);
	stbi_image_free(outdata);

	return 0;
}

float GetMean(float* number, int count)
{
	float sum = 0.0;
	for (int i = 0; i < count; i++)
	{
		sum += *number;
		number++;

	}
	return (float)(sum / count);
}

float GetStandardDeviation(float* number, float average, int count)
{
	float sum = 0.0;
	for (int i = 0; i < count; i++)
	{
		sum += (*number - average)*(*number - average);
		number++;
	}
	return (float)sqrt((sum / (count - 1)));
}
