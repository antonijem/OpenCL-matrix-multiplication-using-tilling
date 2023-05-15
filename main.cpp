#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <iostream>
#include <CL/cl.hpp>
#include <stdio.h>

#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

void print_device_info(cl_device_id device) {
	char name[128];
	cl_uint compute_units;
	cl_ulong global_mem_size, local_mem_size;
	size_t max_workgroup_size;

	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_workgroup_size), &max_workgroup_size, NULL);
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);

	printf("Device name: %s\n", name);
	printf("Compute units: %u\n", compute_units);
	printf("Max workgroup size: %zu\n", max_workgroup_size);
	printf("Global memory size: %llu bytes\n", global_mem_size);
	printf("Local memory size : % llu bytes\n", local_mem_size);
}

void multiply_matrices(float* A, float* B, float* C, int m, int n, int k) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float sum = 0.0f;
			for (int l = 0; l < k; l++) {
				sum += A[i * k + l] * B[l * n + j];
			}
			C[i * n + j] = sum;
		}
	}
}


int main(int argc, char* argv[]) {

	if (argc != 6) {
		return -1;
	}

	//Collecting arguments 
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	int n_runs = atoi(argv[4]);
	int n_works = atoi(argv[5]);

	if (m <= 0 || n <= 0 || k <= 0 || n_runs <= 0) {
		printf("Invalid input size or number of runs\n");
		return -1;
	}

	//Read kernel code from file
	FILE* fp;
	char* source_str;
	size_t source_size;

	fopen_s(&fp, "matrix_mult.cl", "r");
	if (!fp) {
		printf("Failed to open kernel file.\n");
		exit(1);
	}

	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	//Allocating memory  
	float* A = (float*)malloc(m * k * sizeof(float));
	float* B = (float*)malloc(k * n * sizeof(float));
	float* C = (float*)malloc(m * n * sizeof(float));
	float* C_ref = (float*)malloc(m * n * sizeof(float));

	//Random matrix 
	srand(time(NULL));
	for (int i = 0; i < m * k; i++) {
		A[i] = rand() / (float)RAND_MAX;
	}

	for (int i = 0; i < k * n; i++) {
		B[i] = rand() / (float)RAND_MAX;
	}

	cl_int err = 0;

	std::vector < cl::Platform > platforms;
	cl::Platform::get(&platforms);

	//Select the first platform
	cl::Platform platform = platforms.front();

	//Get OpenCl devices
	std::vector< cl::Device > devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	cl::Device device = devices.front();

	print_device_info(device());

	cl_context context = clCreateContext(NULL, 1, &device(), NULL, NULL, &err);

	

	/*cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device(), &properties, &err);*/
	cl_command_queue command_queue = clCreateCommandQueue(context, device(), CL_QUEUE_PROFILING_ENABLE, &err);

	//Create memory buffer on the device
	cl_mem mem_A = clCreateBuffer(context, CL_MEM_READ_ONLY, m * k * sizeof(float), NULL, &err);
	cl_mem mem_B = clCreateBuffer(context, CL_MEM_READ_ONLY, k * n * sizeof(float), NULL, &err);
	cl_mem mem_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m * n * sizeof(float), NULL, &err);

	//Copy matrices A and B to the device memory
	err = clEnqueueWriteBuffer(command_queue, mem_A, CL_TRUE, 0, m * k * sizeof(float), A, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, mem_B, CL_TRUE, 0, k * n * sizeof(float), B, 0, NULL, NULL);

	//Create a program from kernel source code
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &err);

	//Compile the program
	err = clBuildProgram(program, 1, &device(), NULL, NULL, NULL);

	if (err != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(program, device(), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = (char*)malloc(log_size);
		clGetProgramBuildInfo(program, device(), CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("Build log:\n%s\n", log);
		free(log);
		// Handle the error appropriately
	}

	//Create the kernel
	cl_kernel kernel = clCreateKernel(program, "matrix_mult", &err);

	//Set the arguments of the kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_A);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_B);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mem_C);
	err = clSetKernelArg(kernel, 3, sizeof(int), (void*)&m);
	err = clSetKernelArg(kernel, 4, sizeof(int), (void*)&n);
	err = clSetKernelArg(kernel, 5, sizeof(int), (void*)&k);
	err = clSetKernelArg(kernel, 6, sizeof(int), (void*)&n_works);

	//Execute the kernel and measure the execution time
	cl_event event;
	size_t global_size[2] = { m,n / n_works };
	size_t local_size[2] = { 64 ,64 / n_works }; //Specific to the device running on
	double total_time = 0.0;
	double* time_arr = new double[n_runs];
	for (int i = 0; i < n_runs; i++) {
		err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to execute kernel\n");
			return -2;
		}
		err = clWaitForEvents(1, &event);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to wait for event\n");
			return -2;
		}
		cl_ulong start_time = 0, end_time = 0;
		err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
		err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
		double execution_time = (double)(end_time - start_time) * 1.0e-6;
		total_time += execution_time;
		time_arr[i] = execution_time;
	}

	//Compute the mean and standard deviation of the execution time
	double mean_time = total_time / n_runs;
	double std_dev_time = 0.0;
	for (int i = 0; i < n_runs; i++) {
		std_dev_time += pow(time_arr[i] - mean_time, 2);
	}
	std_dev_time = sqrt(std_dev_time / n_runs);

	printf("Mean execution time: %f ms\n", mean_time);
	printf("Standard deviation of execution time: %f ms\n", std_dev_time);

	//Copying the result matrix C back to the host
	err = clEnqueueReadBuffer(command_queue, mem_C, CL_TRUE, 0, m * n * sizeof(float), C, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error reading result buffer\n");
		return -3;
	}

	multiply_matrices(A, B, C_ref, m, n, k);

	//Check the results
	int errors = 0;
	for (int i = 0; i < m * n; i++) {
		if (fabs(C[i] - C_ref[i]) > 1) {
			errors++;
		}
	}
	if (errors == 0) {
		printf("Result matrix is correct\n");
	}
	else {
		printf("Result matrix is incorrect, %d errors\n", errors);
	}

	//Free memory
	clReleaseMemObject(mem_A);
	clReleaseMemObject(mem_B);
	clReleaseMemObject(mem_C);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(A);
	free(B);
	free(C);
	free(C_ref);
	delete time_arr;

	return 0;
}