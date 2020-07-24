#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#define random(x) rand() % (x)
//#define IMAGE


int convertToString(const char* filename, std::string& s)
{
	size_t size;
	char* str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));
	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return NULL;
		}
		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	printf("Error: Failed to open file %s\n", filename);
	return -1;
}
int main()
{
	cl_int status = 0;
	cl_platform_id platform;
	//查询平台
	status = clGetPlatformIDs(1, &platform, nullptr);
	std::vector<cl_platform_info> param_names = { CL_PLATFORM_NAME,CL_PLATFORM_VENDOR,CL_PLATFORM_VERSION,CL_PLATFORM_PROFILE };
	for (size_t i = 0; i < param_names.size(); i++)
	{
		size_t size;
		char* Ndata;
		clGetPlatformInfo(platform, param_names[i], 0, NULL, &size);
		Ndata = (char*)malloc(size);
		clGetPlatformInfo(platform, param_names[i], size, Ndata, NULL);
		std::cout << Ndata << std::endl;
		free(Ndata);
	}
	std::cout << "\n" << std::endl;
	//查询平台上的设备
	//clGetDeviceIDs();
	cl_device_id device;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
	std::vector<cl_device_info> device_param = { CL_DEVICE_NAME,CL_DEVICE_VENDOR };
	for (size_t i = 0; i < device_param.size(); i++)
	{
		size_t size;
		char* Ndata;
		clGetDeviceInfo(device, device_param[i], 0, NULL, &size);
		Ndata = (char*)malloc(size);
		clGetDeviceInfo(device, device_param[i], size, Ndata, NULL);
		std::cout << Ndata << std::endl;
		free(Ndata);
	}
	//创建上下文
	//clCreateContext();
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
	//创建命令队列
	//clCreateCommandQueue();
	cl_command_queue commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, nullptr);
	if (commandQueue == NULL)
	{
		perror("Failed to create commandQueue for device 0.");
	}
	//建立要传入从机的数据
	const float filter[] = { 0.2f,0.2f,0.2f,0.2f,0.2f,0.2f,0.2f,0.2f,0.2f };

	const int Ndim = 12;
	const int Mdim = 10;
	int szA = Ndim * Ndim;
	int szC = Mdim * Mdim;

	float* input;
	float* output;
	// void *malloc(size_t size) 分配所需的内存空间，并返回一个指向它的指针
	input = (float*)malloc(szA * sizeof(float));
	output = (float*)malloc(szC * sizeof(float));
	for (int i = 0; i < szA; i++)
		input[i] = static_cast<float>(random(5));

	int filterWidth = 3;
	int filterSize = filterWidth * filterWidth;
#ifdef IMAGE
	// image
	cl_image_format format;
	format.image_channel_order = CL_R;
	format.image_channel_data_type = CL_FLOAT;

	cl_image_desc Inputdesc;
	memset(&Inputdesc, 0, sizeof(Inputdesc));
	Inputdesc.image_type = CL_MEM_OBJECT_IMAGE2D;
	Inputdesc.image_width = Ndim;
	Inputdesc.image_height = Ndim;
	Inputdesc.image_depth = 0;
	Inputdesc.image_array_size = 0;
	Inputdesc.image_row_pitch = 0;
	Inputdesc.image_slice_pitch = 0;
	Inputdesc.num_mip_levels = 0;
	Inputdesc.num_samples = 0;
	Inputdesc.buffer = NULL;
	cl_mem inputImage = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &Inputdesc, input, &status);
	if (status < 0) {
		printf("Couldn't clCreateinputImage");
		exit(1);
	};
	cl_image_desc Outputdesc;
	memset(&Outputdesc, 0, sizeof(Outputdesc));
	Outputdesc.image_type = CL_MEM_OBJECT_IMAGE2D;
	Outputdesc.image_width = Mdim;
	Outputdesc.image_height = Mdim;
	Outputdesc.image_depth = 0;
	Outputdesc.image_array_size = 0;
	Outputdesc.image_row_pitch = 0;
	Outputdesc.image_slice_pitch = 0;
	Outputdesc.num_mip_levels = 0;
	Outputdesc.num_samples = 0;
	Outputdesc.buffer = NULL;
	cl_mem outputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &Outputdesc, output, &status);
	if (status < 0) {
		printf("Couldn't clCreateoutputImage");
		exit(1);
	};
	cl_mem bufferFilter = clCreateBuffer(context, 0, sizeof(float) * filterSize, nullptr, &status);
	if (status < 0) {
		printf("Couldn't clCreateBuffer");
		exit(1);
	};
	size_t in_origin[3] = { 0, 0, 0 }, in_region[3] = { Ndim, Ndim, 1 };
	status = clEnqueueWriteImage(commandQueue, inputImage, CL_TRUE, in_origin, in_region, 0, 0, input, 0, nullptr, nullptr);
	if (status < 0) {
		printf("Couldn't clEnqueueWriteImage");
		exit(1);
	};
	status = clEnqueueWriteBuffer(commandQueue, bufferFilter, CL_FALSE, 0, sizeof(float) * filterSize, filter, 0, nullptr, nullptr);
	if (status < 0) {
		printf("Couldn't clEnqueueWriteBuffer");
		exit(1);
	};

#else
	//clCreateBuffer();
	//创建三个 OpenCL 内存对象，并把buf1 的内容通过隐式拷贝的方式
	//拷贝到clbuf1, buf2 的内容通过显示拷贝的方式拷贝到clbuf2

	cl_mem memObjects[3] = { 0, 0, 0 };
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * szA, input, &status);
	if (status < 0) {
		printf("Couldn't clCreateBuffer0");
		exit(1);
	}
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * szC, output, &status);
	if (status < 0) {
		printf("Couldn't clCreateBuffer1");
		exit(1);
	}
	memObjects[2] = clCreateBuffer(context, 0, sizeof(float) * filterSize, nullptr, &status);
	if (status < 0) {
		printf("Couldn't clCreateBuffer2");
		exit(1);
	}
	status = clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_FALSE, 0, sizeof(float) * szA, input, 0, nullptr, nullptr);
	if (status < 0) {
		printf("Couldn't clEnqueueWriteBuffer");
		exit(1);
	}
	status = clEnqueueWriteBuffer(commandQueue, memObjects[2], CL_FALSE, 0, sizeof(float) * filterSize, filter, 0, nullptr, nullptr);
	if (status < 0) {
		printf("Couldn't clEnqueueWriteBuffer");
		exit(1);
	}

#endif
	
#ifdef IMAGE
	//读取CL程序源码
	const char* filename = "conv_img.cl";
#else
	//读取CL程序源码
	const char* filename = "conv.cl";
#endif // IMAGE

	
	std::string sourceStr;
	status = convertToString(filename, sourceStr);
	if (status)
		std::cout << status << "  !!!!!!!!" << std::endl;

	const char* source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };

	//创建CL程序
	//clCreateProgramWithSource();
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);
	if (status < 0) {
		printf("Couldn't clCreateProgramWithSource");
		exit(1);
	};
	//编译CL程序
	//clBuildProgram();
	status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	if (status < 0) {
		printf("Couldn't clBuildProgram");
		exit(1);
	};
	if (status != 0)
	{
		printf("clBuild failed:%d\n", status);
		char tbuf[0x10000];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf, nullptr);
		printf("\n%s\n", tbuf);
	}
#ifdef IMAGE
	//创建CL内核
	//clCreateKernel();
	cl_kernel kernel = clCreateKernel(program, "conv2d", &status);
	if (status < 0) {
		printf("Couldn't clCreateKernel");
		exit(1);
	};
#else
	//创建CL内核
	//clCreateKernel();
	cl_kernel kernel = clCreateKernel(program, "matrix_mult", &status);
	if (status < 0) {
		printf("Couldn't clCreateKernel");
		exit(1);
	};
#endif // IMAGE

	
#ifdef IMAGE
	//内存推送到GPU里面
	//clSetKernelArg();
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&inputImage);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&outputImage);
	status = clSetKernelArg(kernel, 2, sizeof(int), (void*)&Ndim);
	status = clSetKernelArg(kernel, 3, sizeof(int), (void*)&Mdim);
	status = clSetKernelArg(kernel, 4, sizeof(int), (void*)&filterWidth);
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufferFilter);
	if (status)
		std::cout << "参数设置错误" << std::endl;
#else
	status = clSetKernelArg(kernel, 0, sizeof(int), &Ndim);
	status = clSetKernelArg(kernel, 1, sizeof(int), &Mdim);
	status = clSetKernelArg(kernel, 2, sizeof(int), &filterWidth);
	status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[0]);
	status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[1]);
	status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[2]);
#endif // IMAGE
	
	if (status)
		std::cout << "参数设置错误" << std::endl;
	//计算
	//clEnqueueNDRangeKernel();
	size_t global[2] = { Mdim, Mdim };
	cl_event prof_event;
	cl_ulong ev_start_time = (cl_ulong)0;
	cl_ulong ev_end_time = (cl_ulong)0;
	double rum_time;
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, nullptr, global, nullptr, 0, nullptr, &prof_event);

	if (status)
		std::cout << "执行内核时错误" << std::endl;

	clFinish(commandQueue);

	//读取时间
	status = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, nullptr);
	status = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, nullptr);

	if (status)
		perror("读取时间的时候发生错误\n");

	rum_time = (double)(ev_end_time - ev_start_time)/1000000;

#ifdef IMAGE
	std::cout << "image2D执行时间为:" << rum_time << "ms" << std::endl;
	//读取结果
	//clEnqueueReadImage
	cl_event prof_event_map;
	cl_ulong map_start_time = (cl_ulong)0;
	cl_ulong map_end_time = (cl_ulong)0;
	double map_rum_time;
	size_t out_origin[3] = { 0, 0, 0 }, out_region[3] = { Mdim, Mdim, 1 };
	//status = clEnqueueReadImage(commandQueue, outputImage, CL_TRUE, out_origin, out_region, 0, 0, output, 0, nullptr, &prof_event_map);
	auto map_memory = clEnqueueMapImage(commandQueue, outputImage, CL_TRUE, CL_MAP_READ, out_origin, out_region, 0, 0, 0, nullptr, &prof_event_map, &status);
	if (status)
		perror("clEnqueueMapImage读回数据的时候发生错误\n");
	clGetEventProfilingInfo(prof_event_map, CL_PROFILING_COMMAND_START, sizeof(map_start_time), &map_start_time, nullptr);
	clGetEventProfilingInfo(prof_event_map, CL_PROFILING_COMMAND_END, sizeof(map_end_time), &map_end_time, nullptr);
	memcpy(output, map_memory, sizeof(float) * szC);

	map_rum_time = (double)(map_end_time - map_start_time) / 1000000;
	std::cout << "image2D读取时间为:" << map_rum_time << "ms" << std::endl;
	status = clEnqueueUnmapMemObject(commandQueue, outputImage, map_memory, 0, nullptr, nullptr);
	if (status)
		perror("clEnqueueUnmapMemObject读回数据的时候发生错误\n");
#else
	std::cout << "buffer执行时间为:" << rum_time << "ms" << std::endl;
	//clEnqueueReadBuffer();
	cl_event prof_event_map;
	cl_ulong map_start_time = (cl_ulong)0;
	cl_ulong map_end_time = (cl_ulong)0;
	double map_rum_time;
	//status = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, sizeof(float) * szC, output, 0, nullptr, &prof_event_map);
	auto map_memory =  clEnqueueMapBuffer(commandQueue, memObjects[1],CL_TRUE,CL_MAP_READ,0, sizeof(float) * szC,0,nullptr, &prof_event_map,&status);
	if (status)
		perror("clEnqueueMapBuffer读回数据的时候发生错误\n");

	clGetEventProfilingInfo(prof_event_map, CL_PROFILING_COMMAND_START, sizeof(map_start_time), &map_start_time, nullptr);
	clGetEventProfilingInfo(prof_event_map, CL_PROFILING_COMMAND_END, sizeof(map_end_time), &map_end_time, nullptr);
	map_rum_time = (double)(map_end_time - map_start_time) / 1000000;
	std::cout << "buffer读取时间为:" << map_rum_time << "ms" << std::endl;
	memcpy(output, map_memory, sizeof(float) * szC);
	status = clEnqueueUnmapMemObject(commandQueue, memObjects[1], map_memory, 0, nullptr, nullptr);
	
	if (status)
		perror("clEnqueueUnmapMemObject读回数据的时候发生错误\n");
#endif // IMAGE
	

	//结果显示
	printf("\nArray input:\n");
	for (int i = 0; i < Ndim; i++) {
		for (int j = 0; j < Ndim; j++)
			printf("%.3f\t", input[i * Ndim + j]);
		printf("\n");
	}
	printf("\nArray mask:\n");
	for (int i = 0; i < filterWidth; i++) {
		for (int j = 0; j < filterWidth; j++)
			printf("%.3f\t", filter [i * filterWidth + j] );
		printf("\n");
	}

	printf("\nArray output:\n");
	for (int i = 0; i < Mdim; i++) {
		for (int j = 0; j < Mdim; j++)
			printf("%.3f\t", output[i * Mdim + j]);
		printf("\n");
	}

	std::cout << std::endl;
	if (input != NULL)
		free(input);

	if (output != NULL)
		free(output);
#ifdef IMAGE
	clReleaseMemObject(inputImage);
	clReleaseMemObject(outputImage);
	clReleaseMemObject(bufferFilter);
#else
		clReleaseMemObject(memObjects[2]);
		clReleaseMemObject(memObjects[1]);
		clReleaseMemObject(memObjects[0]);
#endif // IMAGE

	clReleaseProgram(program);
	clReleaseCommandQueue(commandQueue);
	clReleaseContext(context);
	system("pause");


	return 0;

}
