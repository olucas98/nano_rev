/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <ctime>
#include <cmath>

typedef float DTYPE;
//const int SIZE = 512;

#define WINDOW 13
#define KERNEL_SIZE 3
#define CHANNELS 8
#define N WINDOW+KERNEL_SIZE-1

/*void mm_sw( std::vector<DTYPE, aligned_allocator<DTYPE> > A, std::vector<DTYPE, aligned_allocator<DTYPE> > B, std::vector<DTYPE, aligned_allocator<DTYPE> > & AB){
//void mm_sw( std::vector<DTYPE, aligned_allocator<DTYPE> > At, std::vector<DTYPE, aligned_allocator<DTYPE> > B, std::vector<DTYPE, aligned_allocator<DTYPE> > & AB){

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if( tid == 0 ){
            int nthreads = omp_get_num_threads();
            std::cout << "Running OpenMP with " << nthreads << " threads...\n";
        }
    }

    DTYPE sum = 0;
#pragma omp parallel for private(sum)
    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j<SIZE; j++){
            sum = 0;
            for(int k = 0; k < SIZE; k++){
                sum = sum + A[i*SIZE+k] * B[k*SIZE+j];
                //sum = sum + At[k*SIZE+i] * B[k*SIZE+j];
            }
            AB[i*SIZE+j] = sum;
        }
    }
}*/

void relu(float* v){
    for (int j = 0; j < WINDOW; j++){
        if (v[j] < 0){
            v[j] = 0;    
        }
    }
}

void conv_sw(float* f, float* g, float* out) {
  //int const n  = WINDOW + KERNEL_SIZE - 1;
  //float out[N] = {0};
  for(int i = 0; i < N; ++i) {
	  out[i] = 0;
    int const jmn = (i >= KERNEL_SIZE - 1)? i - (KERNEL_SIZE - 1) : 0;
    int const jmx = (i <  WINDOW - 1)? i            : WINDOW - 1;
    for(int j = jmn; j <= jmx; ++j) {
      out[i] += (f[j] * g[i - j]);
    }
  }
  //return out; 
}

void res_block(float win[], std::vector<DTYPE, aligned_allocator<DTYPE> > kern, float* v1, float v2[][WINDOW], float* v3, float v4[][WINDOW]){
	for (int k =0; k < CHANNELS; k++){
		
		
		conv_sw(&win[WINDOW*k], &kern[KERNEL_SIZE*k], v1);
		for(int i = 1; i < N - 1; i++){
			v2[k][i-1] = v1[i];
		}
		relu(v2[k]);
		
    }
	
	for (int k =0; k < CHANNELS; k++){
		
		
		conv_sw(v2[k], &kern[KERNEL_SIZE*k], v3);
		for(int i = 1; i < N - 1; i++){
			v4[k][i-1] = v3[i];
		}
		relu(v4[k]);
		
    }
	
	for(int i = 0; i < CHANNELS; i++){
		for (int j = 0; j < WINDOW; j++){
			v4[i][j] += win[i*WINDOW+j];
		}
	}
	
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string binaryFile = argv[1];

    cl_int err;
    cl::Context context;
    cl::Kernel krnl_nano_rev;
    cl::CommandQueue q;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
    // hood user ptr
    // is used if it is properly aligned. when not aligned, runtime had no choice
    // but to create
    // its own host side buffer. So it is recommended to use this allocator if
    // user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
    // boundary. It will
    // ensure that user buffer is used when user create Buffer/Mem object with
    // CL_MEM_USE_HOST_PTR

    /*std::vector<DTYPE, aligned_allocator<DTYPE> > A(SIZE*SIZE); 
    //std::vector<DTYPE, aligned_allocator<DTYPE> > At(SIZE*SIZE); 
    std::vector<DTYPE, aligned_allocator<DTYPE> > B(SIZE*SIZE); 
    std::vector<DTYPE, aligned_allocator<DTYPE> > AB_sw(SIZE*SIZE); 
    std::vector<DTYPE, aligned_allocator<DTYPE> > AB_hw(SIZE*SIZE); 

    srand(time(NULL));

    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j < SIZE; j++){
                A[i*SIZE+j] = rand() % 8;
                //At[i*SIZE+j] = rand() % 8;
                B[i*SIZE+j] = rand() % 8;
                //A[i*SIZE+j] = 1;
                //B[i*SIZE+j] = 1;

                AB_sw[i*SIZE+j] = 0;
                AB_hw[i*SIZE+j] = 0;
        }
    }
    printf("Done initializing vectors\n");

    std::cout << "Running SW MM...\n";
    mm_sw(A, B, AB_sw);
    //mm_sw(At, B, AB_sw);
    printf("Done\n");
	*/
	
	//Now start my new host code
	printf("\nStarting software conv\n");
	srand(unsigned(time(NULL)));
	//std::vector<DTYPE, aligned_allocator<DTYPE> > At(SIZE*SIZE); 
    std::vector<DTYPE, aligned_allocator<DTYPE> > f(WINDOW*CHANNELS);
	
    std::vector<DTYPE, aligned_allocator<DTYPE> > kern(CHANNELS * KERNEL_SIZE);
	std::vector<DTYPE, aligned_allocator<DTYPE> > out_hw(CHANNELS * WINDOW);
    std::generate(f.begin(), f.end(), std::rand);
    /*for (int k =0; k < CHANNELS; k++){
        std::generate(kern[k].begin(), kern[k].end(), std::rand);
    }
    for (int j = 0; j < WINDOW; j++){
        f[j] = (float)std::fmod(f[j], 200) - 100;
    }
    for (int k =0; k < CHANNELS; k++){
        for (int j = 0; j < KERNEL_SIZE; j++){
            kern[k][j] = (float)(std::fmod(kern[k][j],20)) - 10;
        }
    }*/
	
	for(int i = 0; i < CHANNELS; i++){
        for(int j = 0; j < KERNEL_SIZE; j++){
                kern[i * KERNEL_SIZE + j] = fmod(rand(), 20) - 5;
        }
    }
	for(int i = 0; i < CHANNELS; i++){
		for (int j = 0; j < WINDOW; j++){
			f[i*WINDOW+j] = (float)std::fmod(rand(), 200) - 100;
		}
	}
    //std::vector<std::vector<float>> v1( CHANNELS , std::vector<float> (WINDOW + KERNEL_SIZE - 1, 0));
    //std::vector<std::vector<float>> v2( CHANNELS , std::vector<float> (WINDOW, 0));
	//std::vector<std::vector<DTYPE, aligned_allocator<DTYPE> > out_hw( CHANNELS , std::vector<float> (WINDOW, 0));
	float v1[N];
	float v2[CHANNELS][WINDOW];
	float win[WINDOW*CHANNELS];
	float v3[N];
	float v4[CHANNELS][WINDOW];
	
	for(int i = 0; i < CHANNELS; i++){
		for(int j = 0; j < WINDOW; j++){
			win[i*WINDOW+j] = f[i*WINDOW+j];
		}
	}
	
	res_block(win, kern, v1, v2, v3, v4);
	
	float v5[N];
	float v6[CHANNELS][WINDOW];
	float win2[WINDOW*CHANNELS];
	float v7[N];
	float v8[CHANNELS][WINDOW];
	for(int i = 0; i < CHANNELS; i++){
		for(int j = 0; j < WINDOW; j++){
			win2[i*WINDOW+j] = v4[i][j];
		}
	}
	res_block(win2, kern, v5, v6, v7, v8); 
	
	float v9[N];
	float v10[CHANNELS][WINDOW];
	float win3[WINDOW*CHANNELS];
	float v11[N];
	float v12[CHANNELS][WINDOW];
	for(int i = 0; i < CHANNELS; i++){
		for(int j = 0; j < WINDOW; j++){
			win3[i*WINDOW+j] = v8[i][j];
		}
	}
	res_block(win3, kern, v9, v10, v11, v12); 
	/*
    for (int k =0; k < CHANNELS; k++){
		
		
		conv_sw(&win[WINDOW*k], &kern[KERNEL_SIZE*k], v1);
		for(int i = 1; i < N - 1; i++){
			v2[k][i-1] = v1[i];
		}
		relu(v2[k]);
		
    }
	
	for (int k =0; k < CHANNELS; k++){
		
		
		conv_sw(v2[k], &kern[KERNEL_SIZE*k], v3);
		for(int i = 1; i < N - 1; i++){
			v4[k][i-1] = v3[i];
		}
		relu(v4[k]);
		
    }
	
	for(int i = 0; i < CHANNELS; i++){
		for (int j = 0; j < WINDOW; j++){
			v4[i][j] += win[i*WINDOW+j];
		}
	}*/

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_nano_rev = cl::Kernel(program, "nano_rev", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_win(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DTYPE)*WINDOW*CHANNELS, f.data(), &err));
    //OCL_CHECK(err, cl::Buffer buffer_At(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DTYPE)*SIZE*SIZE, At.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_kern(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DTYPE)*CHANNELS*KERNEL_SIZE, kern.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(DTYPE)*WINDOW*CHANNELS, out_hw.data(), &err));

    //int matrix_size = SIZE;
    OCL_CHECK(err, err = krnl_nano_rev.setArg(0, buffer_win));
    //OCL_CHECK(err, err = krnl_mm.setArg(0, buffer_At));
    OCL_CHECK(err, err = krnl_nano_rev.setArg(1, buffer_kern));
    OCL_CHECK(err, err = krnl_nano_rev.setArg(2, buffer_out));
    //OCL_CHECK(err, err = krnl_mm.setArg(3, matrix_size));

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_win, buffer_kern}, 0 /* 0 means from host*/));
    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_At, buffer_B}, 0 /* 0 means from host*/));
    q.finish();
    
    std::cout << "Running FPGA MM...\n";
    auto start = std::chrono::steady_clock::now();

    OCL_CHECK(err, err = q.enqueueTask(krnl_nano_rev));
    q.finish();

    auto end = std::chrono::steady_clock::now();
    std::cout << "Done.\n";
    double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //double gops = double(SIZE) * SIZE * SIZE * 2 / (exec_time);
    std::cout << "Time: " << exec_time*1e-9 << " sec"<< std::endl;

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();


    int err_cnt = 0;
    for(int i = 0; i<CHANNELS; i++){ //check for errors later
        for(int j = 0; j<WINDOW; j++){
			printf("%.1f ", out_hw[i*WINDOW+j]);
            if(std::abs((float)(v12[i][j] - out_hw[i*WINDOW+j])) > 1) {
               err_cnt++;
                /*if( err_cnt == 1 ){
                    printf("i:%d j:%d sw:%d hw:%d\n", i, j, AB_sw[i*SIZE+j], AB_hw[i*SIZE+j] );
                }*/
            }
        }
		std::cout << std::endl;
    }

    if(err_cnt != 0){
        printf("FAILED! Error count : %d\n", err_cnt);
        return EXIT_FAILURE;
    }
    else{
        printf("PASSED!\n");
    }

    return EXIT_SUCCESS;
}

