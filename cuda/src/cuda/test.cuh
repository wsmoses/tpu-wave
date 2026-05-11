#ifndef TEST_CUH
#define TEST_CUH

#include <iostream>

__global__ void print_message () 
{
    printf( "\n\n\nThis is from device on block %d thread %d.\n\n\n", blockIdx.x, threadIdx.x ); 
}

__global__ void print_element ( double * ptr )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf( "block %d thread %d : % 16.15e\n", blockIdx.x, threadIdx.x, ptr[i] );
}



#endif