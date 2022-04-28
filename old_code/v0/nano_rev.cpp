/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce
latency and
    device resource utilization of the resulting RTL code
    This is vector addition example to demonstrate how HLS optimizations are
used in kernel.
*******************************************************************************/

#include "hls_stream.h"
#include "ap_int.h"

typedef float DTYPE;
//const int M = 256;

#define WINDOW 13
#define KERNEL_SIZE 3
#define CHANNELS 8
#define N WINDOW+KERNEL_SIZE-1

void conv(float* f, float* g, float* out) {
  //int const n  = WINDOW + KERNEL_SIZE - 1;
  //float out[N];
  for(int i = 0; i < N; ++i) {
	  out[i] = 0;
    int const jmn = (i >= KERNEL_SIZE - 1)? i - (KERNEL_SIZE - 1) : 0;
    int const jmx = (i <  WINDOW - 1)? i            : WINDOW - 1;
    for(int j = jmn; j <= jmx; ++j) {
      out[i] += (f[j] * g[i - j]);
    }
  }
  //return &out; 
}

void relu(float* v){
    for (int j = 0; j < WINDOW; j++){
        if (v[j] < 0){
            v[j] = 0;    
        }
    }
}

void res_block(float win[], float* kern, float* v1, float v2[][WINDOW], float* v3, float v4[][WINDOW]){
	for (int k =0; k < CHANNELS; k++){
		
		
		conv(&win[WINDOW*k], &kern[KERNEL_SIZE*k], v1);
		for(int i = 1; i < N - 1; i++){
			v2[k][i-1] = v1[i];
		}
		relu(v2[k]);
		
    }
	
	for (int k =0; k < CHANNELS; k++){
		
		
		conv(v2[k], &kern[KERNEL_SIZE*k], v3);
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

extern "C" {

void nano_rev(DTYPE *win,  DTYPE *kern, DTYPE out[CHANNELS][WINDOW])
{
//#pragma HLS INTERFACE mode=m_axi bundle=m0 port=A 
//#pragma HLS INTERFACE mode=m_axi bundle=m1 port=B 
//#pragma HLS INTERFACE mode=m_axi bundle=m1 port=AB 

#pragma HLS INTERFACE m_axi port = win offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = kern offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = win bundle = control
#pragma HLS INTERFACE s_axilite port = kern bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    float v1[N];
	float v2[CHANNELS][WINDOW];
	float v3[N];
	float v4[CHANNELS][WINDOW];
	
	
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
	
	for (int k =0; k < CHANNELS; k++){
		for (int i = 0; i < WINDOW; i++){
			out[k][i] = v12[k][i];
		}
	}

	
	
	/*for (int k =0; k < CHANNELS; k++){
		
		
		conv(&win[WINDOW*k], &kern[KERNEL_SIZE*k], v1);
		for(int i = 1; i < N - 1; i++){
			v2[k][i-1] = v1[i];
		}
		relu(v2[k]);
	}
	
	for (int k =0; k < CHANNELS; k++){
		
		
		conv(v2[k], &kern[KERNEL_SIZE*k], v3);
		for(int i = 1; i < N - 1; i++){
			v4[k][i-1] = v3[i];
		}
		relu(v4[k]);
		

		
    
		
		for (int i = 0; i < WINDOW; i++){
			out[k][i] = v4[k][i] + win[k*WINDOW + i];
		}
    
		//for(float i : v2[k]) std::cout << i << " ";
		//std::cout << std::endl;
    }*/
	
	

    
}
}
