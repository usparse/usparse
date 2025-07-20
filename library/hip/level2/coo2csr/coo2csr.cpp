#include "coo2csr.h"

__global__ void coo2csr_part1(const int m, const int nnz, const int* coo_row_ind, int* csr_row_ptr)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % warpSize;
    if (tid == 0) {
        csr_row_ptr[0] = 0;
        csr_row_ptr[m] = nnz;
    }
    if (tid >= nnz) {
        return;
    }
    int current_row = coo_row_ind[tid];
    int next_row = __shfl_down(current_row, 1);
    if(lane == warpSize-1 && tid != nnz-1) {
        next_row = coo_row_ind[tid+1];
    }
    if(tid == nnz - 1) {
        next_row = current_row;   
    }
    if(current_row != next_row) {
	    csr_row_ptr[next_row] = tid+1;
    }
}
/*---------end----------*/
__global__ void coo2csr_part2(const int m, int* csr_row_ptr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if((tid == 0) || (tid == 1) || (tid > m)){
        return;
    }
    if((csr_row_ptr[tid-1] == 0) && (csr_row_ptr[tid] != 0)){
        int flag = tid - 1;
        int value = csr_row_ptr[tid];
        while(flag > 0 && csr_row_ptr[flag] == 0){
            csr_row_ptr[flag] = value;
            flag--;
        }
    }
}


void coo2csr(const int m, const int nnz, const int* coo_row_ind, int* csr_row_ptr){
    int threads=512;
    int blocks=(nnz-1)/threads+1;
    coo2csr_part1<<<blocks, threads>>>(m, nnz, coo_row_ind, csr_row_ptr);
    coo2csr_part2<<<blocks, threads>>>(m,csr_row_ptr);
}