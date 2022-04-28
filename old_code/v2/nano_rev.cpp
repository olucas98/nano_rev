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

typedef ap_int<416> win_line;

void conv(float* f, float* g, float* out) {
	#pragma HLS INLINE
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
	#pragma HLS INLINE
	
    for (int j = 0; j < WINDOW; j++){
		#pragma HLS unroll
        if (v[j] < 0){
            v[j] = 0;    
        }
    }
}

void res_block(float win[], float* kern, float* v1, float v2[][WINDOW], float* v3, float v4[][WINDOW]){
	#pragma HLS INLINE
	
	for (int k =0; k < CHANNELS; k++){
		#pragma HLS unroll
		
		conv(&win[WINDOW*k], &kern[KERNEL_SIZE*k], v1);
		for(int i = 1; i < N - 1; i++){
			#pragma HLS unroll
			v2[k][i-1] = v1[i];
		}
		relu(v2[k]);
		
    }
	
	for (int k =0; k < CHANNELS; k++){
		#pragma HLS unroll
		
		conv(v2[k], &kern[KERNEL_SIZE*k], v3);
		for(int i = 1; i < N - 1; i++){
			#pragma HLS unroll
			v4[k][i-1] = v3[i];
		}
		relu(v4[k]);
		
    }
	#pragma HLS unroll
	for(int i = 0; i < CHANNELS; i++){
		#pragma HLS unroll
		for (int j = 0; j < WINDOW; j++){
			#pragma HLS unroll
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
	float win_buf[WINDOW*CHANNELS];
	float kern_buf[WINDOW*CHANNELS];
	
	#pragma HLS ARRAY_PARTITION variable = win_buf complete
	#pragma HLS ARRAY_PARTITION variable = kern_buf complete
	
	for(int j = 0; j < CHANNELS; j++){
		for(int k = 0; k < WINDOW; k++){
			win_buf[j*WINDOW+k] = win[j*WINDOW+k];
		}
	}
	
	for(int j = 0; j < CHANNELS; j++){
		for(int k = 0; k < KERNEL_SIZE; k++){
			kern_buf[j*KERNEL_SIZE+k] = kern[j*KERNEL_SIZE+k];
		}
	}
	
	
	
	#pragma HLS ARRAY_PARTITION variable = v1 complete
	#pragma HLS ARRAY_PARTITION variable = v2 complete
	#pragma HLS ARRAY_PARTITION variable = v3 complete
	#pragma HLS ARRAY_PARTITION variable = v4 complete

	
	
	
	res_block(win_buf, kern_buf, v1, v2, v3, v4);
	
	float v5[N];
	float v6[CHANNELS][WINDOW];
	float win2[WINDOW*CHANNELS];
	float v7[N];
	float v8[CHANNELS][WINDOW];
	
	#pragma HLS ARRAY_PARTITION variable = v5 complete
	#pragma HLS ARRAY_PARTITION variable = v6 complete
	#pragma HLS ARRAY_PARTITION variable = win2 complete
	#pragma HLS ARRAY_PARTITION variable = v7 complete
	#pragma HLS ARRAY_PARTITION variable = v8 complete
	
	for(int i = 0; i < CHANNELS; i++){
		#pragma HLS unroll
		for(int j = 0; j < WINDOW; j++){
			#pragma HLS unroll
			win2[i*WINDOW+j] = v4[i][j];
		}
	}
	
	res_block(win2, kern_buf, v5, v6, v7, v8); 
	
	float v9[N];
	float v10[CHANNELS][WINDOW];
	float win3[WINDOW*CHANNELS];
	float v11[N];
	float v12[CHANNELS][WINDOW];
	
	#pragma HLS ARRAY_PARTITION variable = v9 complete
	#pragma HLS ARRAY_PARTITION variable = v10 complete
	#pragma HLS ARRAY_PARTITION variable = win3 complete
	#pragma HLS ARRAY_PARTITION variable = v11 complete
	#pragma HLS ARRAY_PARTITION variable = v12 complete
	
	for(int i = 0; i < CHANNELS; i++){
		#pragma HLS unroll
		for(int j = 0; j < WINDOW; j++){
			#pragma HLS unroll
			win3[i*WINDOW+j] = v8[i][j];
		}
	}
	
	res_block(win3, kern_buf, v9, v10, v11, v12);
	
	for (int k =0; k < CHANNELS; k++){
		#pragma HLS unroll
		for (int i = 0; i < WINDOW; i++){
			#pragma HLS unroll
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
