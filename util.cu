/*
这个文件主要是放一些例如排序，合并的公用函数
*/

#include "util.h"

// ddd 1:生序 0: 降序
__device__ void swap(PPQ_ITEM * buffer, int source, int target, int ddd){
	PPQ_ITEM temp;
	if(((buffer[source].key>buffer[target].key)&&(ddd)) || ((buffer[source].key<buffer[target].key)&&(1-ddd))){
		temp=buffer[source];
		buffer[source]=buffer[target];
		buffer[target]=temp;
	}
}

/*
使用bitonic排序法 对并发堆节点数组进行排序
dir 排序的方向 1:生序 0: 降序
*/
__device__ void bitonic_sort(PPQ_ITEM * buffer,int arrayLength,int dir){
	for(int size = 2; size < arrayLength; size <<= 1){
        //Bitonic merge
        int ddd = dir ^ ( (threadIdx.x & (size / 2)) != 0 );
        for(int stride = size / 2; stride > 0; stride >>= 1){
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        	if(threadIdx.x<(NUM_THREADS>>1)){
        		swap(buffer, pos, (pos+stride),ddd);
        	}
        }
    }

    //ddd == dir for the last bitonic merge step
    {
        for(int stride = arrayLength / 2; stride > 0; stride >>= 1){
            __syncthreads();
            int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        	if(threadIdx.x<(NUM_THREADS>>1)){
            	swap(buffer, pos, (pos+stride),dir);
        	}
        }
    }
    __syncthreads();
}

// 对buffer里的数进行排序
__device__ void sort_one(PPQ_ITEM * buffer,int num,int * count){
	
	int tx=threadIdx.x;
	int pout = 0, pin = NUM_THREADS;
	int out=NUM_THREADS,in=0;
	int i,j;
	int offset;
	int totalOne;
	int b,d,f;
	
	for(i=0;i<BITS;i++){
		// 取buffer里元素的i位 存到count数组里
		for(j=tx;j<num;j=j+blockDim.x){
			count[pout+j]=(1-((buffer[in+j].key>>i)&0x00000001));
		}
		__syncthreads();
		// 计算count数组里1的个数
		for (offset = 1; offset < num; offset *= 2) {
			pout = NUM_THREADS - pout; // swap double buffer indices 
			pin = NUM_THREADS - pout;
			for(j=tx;j<num;j=j+blockDim.x){
				if (j >= offset) {
					count[pout+j] = count[pin+j-offset]+count[pin+j];
				}
				else {
					count[pout+j] = count[pin+j];
				}
			}
			__syncthreads();
		}
		totalOne=count[pout+num-1];
		// 改变buffer
		for(j=tx;j<num;j=j+blockDim.x){
			b=((buffer[in+j].key>>i)&0x00000001);
			f=count[pout+j]-(1-b);
			if(b==0){
				d=f;
			}
			else{
				d=j+totalOne-f;
			}
			buffer[out+d]=buffer[in+j];
		}
		out = NUM_THREADS - out; // swap double buffer indices 
		in = NUM_THREADS - out;
		__syncthreads(); 
	}
}

// 把数组tempA和数组tempB的数合并起来，存到数组C里.
__device__ void merge(PPQ_ITEM * tempA, int numA, PPQ_ITEM * tempB, int numB,PPQ_ITEM * C){
	int tx=threadIdx.x;
	int size=blockDim.x;
	
	int i,j,k,mid;
	
	for(i=tx;i<numA;i+=size){
		j=0;
		k=numB-1;
		if(tempA[i].key>=tempB[0].key){
			C[i]=tempA[i];
		}
		else if(tempA[i].key<tempB[numB-1].key){
			C[i+numB]=tempA[i];
		}
		else{
			while(j<k-1){
				mid=(j+k)/2;
				if(tempB[mid].key>tempA[i].key){
					j=mid;
				}
				else{
					k=mid;
				}
			}
			//printf("i=%d,j=%d,C=%d\n",i,j,tempA[i]);
			C[i+j+1]=tempA[i];
		}
	}	
	for(i=tx;i<numB;i+=size){
		j=0;
		k=numA-1;
		if(tempB[i].key>tempA[0].key){
			C[i]=tempB[i];
		}
		else if(tempB[i].key<=tempA[numA-1].key){
			C[i+numA]=tempB[i];
		}
		else{
			while(j<k-1){
				mid=(j+k)/2;
				if(tempA[mid].key>=tempB[i].key){
					j=mid;
				}
				else{
					k=mid;
				}
			}
			//printf("i=%d,j=%d,C=%d\n",i,j,tempB[i]);
			C[i+j+1]=tempB[i];
		}
	}	 
	__syncthreads();
}  

// Utils...
// http://www.moderngpu.com/intro/scan.html
__device__ void scan_backup(const int* values, int* exclusive) {

	// Reserve a half warp of extra space plus one per warp in the block.
	// This is exactly enough space to avoid comparisons in the multiscan
	// and to avoid bank conflicts.
	__shared__ volatile int scan[NUM_WARPS * SCAN_STRIDE];
	int tid = threadIdx.x;
	int warp = tid / WARP_SIZE;
	int lane = (WARP_SIZE - 1) & tid;

	volatile int* s = scan + SCAN_STRIDE * warp + lane + WARP_SIZE / 2;
	s[-16] = 0;

	// Read from global memory.
	int x = values[tid];
	s[0] = x;

	// Run inclusive scan on each warp's data.
	int sum = x;    

#pragma unroll
	for(int i = 0; i < 5; ++i) {
		int offset = 1<< i;
		sum += s[-offset];
		s[0] = sum;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	__shared__ volatile int totals[NUM_WARPS + NUM_WARPS / 2];
	if(tid < NUM_WARPS) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		int total = scan[SCAN_STRIDE * tid + WARP_SIZE / 2 + WARP_SIZE - 1];

		totals[tid] = 0;
		volatile int* s2 = totals + NUM_WARPS / 2 + tid;
		int totalsSum = total;
		s2[0] = total;

#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
			int offset = 1<< i;
			totalsSum += s2[-offset];
			s2[0] = totalsSum;  
		}

		// Subtract total from totalsSum for an exclusive scan.
		totals[tid] = totalsSum - total;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	// Add the block scan to the inclusive sum for the block.
	sum += totals[warp];

	// Write the inclusive and exclusive scans to global memory.
//	inclusive[tid] = sum;
	exclusive[tid] = sum - x;
}