// work efficient algorithm with optimization
// when working with float data, output are not correct. 
// the maximum number that can be handled is 1024*65535=67107840.
#include <stdio.h>
#include "scan4.h"
#define datatype int

#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS  
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else 
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) 
#endif

__global__ void scan1(datatype *g_data, datatype *output,int size,int flag,int numOfBlocks) {
	extern __shared__ datatype temp[];// allocated on invocation 
	int offset = 1;
	int tx=threadIdx.x;
	int bx=blockIdx.x+blockIdx.y*1024;

	if(bx<numOfBlocks){
		int n=size;		
		int a1 = tx; 
		int b1 = tx + (size/2);
		int bankOffsetA = CONFLICT_FREE_OFFSET(a1); 
		int bankOffsetB = CONFLICT_FREE_OFFSET(b1);
		temp[a1 + bankOffsetA] = g_data[tx+bx*size]; 
		temp[b1 + bankOffsetB] = g_data[tx+bx*size+n/2];
		
		for (int d = n>>1; d > 0; d >>= 1){ // build sum in place up the tree {
			__syncthreads();
			if (tx < d) {
				int a2 =offset*(2*tx+1)-1;
				int b2 =offset*(2*tx+2)-1;
				a2+=CONFLICT_FREE_OFFSET(a2);
				b2+=CONFLICT_FREE_OFFSET(b2);
				temp[b2]+=temp[a2];
			}
			offset*=2;
		}
		if(tx==0){
			temp[n-1+CONFLICT_FREE_OFFSET(n-1)]=0; 
		}
		for(int d=1;d<n;d*=2){
			offset>>=1;
			__syncthreads();
			if(tx<d){
				int a3 =offset*(2*tx+1)-1;
				int b3 =offset*(2*tx+2)-1;
				a3+=CONFLICT_FREE_OFFSET(a3);
				b3+=CONFLICT_FREE_OFFSET(b3);
				datatype t=temp[a3];
				temp[a3]=temp[b3];
				temp[b3]+=t;
			}
		}	
		__syncthreads();
		
		temp[a1 + bankOffsetA]=temp[a1 + bankOffsetA]+g_data[bx*size+tx];
		temp[b1 + bankOffsetB]=temp[b1 + bankOffsetB]+g_data[bx*size+tx+n/2]; 
		g_data[bx*size+tx]=temp[a1 + bankOffsetA];
		g_data[bx*size+tx+n/2]=temp[b1 + bankOffsetB];   
		if((tx==size/2-1) && flag) {
			output[bx]=temp[b1 + bankOffsetB];
		}
	}	
}

__global__ void scan2(datatype *g_data, datatype *output,int size,int numOfBlocks) {
	int tx=threadIdx.x;
	//int bx=blockIdx.x;
	int bx=blockIdx.x+blockIdx.y*1024;
	int base=bx*size/2;
	int thid = tx+base;
	if((bx!=0)&&(bx<numOfBlocks)){ 
		g_data[thid*2]=g_data[thid*2]+output[bx-1];	
		g_data[thid*2+1]=g_data[thid*2+1]+output[bx-1];	
	}
}

void scan(int *g_data,int length,int size){
    int i; 
    int diff;
    int firstBlocks=(length-1)/size+1;
	int len1=firstBlocks*size;
    int secondBlocks=(firstBlocks-1)/size+1;
    int len2=secondBlocks*size;
    cudaError_t err;
	int firstBlocksX;
	int firstBlocksY;
    datatype* firstOutput=0;
    datatype* secondOutput=0;
    datatype* data = (datatype*) malloc((len1-length)*sizeof(datatype));
    datatype* data_out = (datatype*) malloc(length*sizeof(datatype));
    
    firstBlocksX=1024;
    firstBlocksY=(firstBlocks-1)/1024+1;
    /*err=((cudaMalloc( (void**) &g_data, len1*sizeof(datatype))));
    if( err != cudaSuccess)
	{
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
     	return -1;
	}*/
    err=((cudaMalloc( (void**) &firstOutput, len2*sizeof(datatype))));
	if( err != cudaSuccess)
	{
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
     	exit(-1);
	}    
    //srand(time(NULL));
    /*for(i = 0; i < length; i++) 
    {
        data[i] = (int)(10 * rand()/32768.f);
    }
    */
    for(i = 0; i < (len1-length); i++) 
    {
        data[i] = 0;//(int)(10 * rand()/32768.f);
    }
	//for(i=0;i<1025;i++){
    	//printf("data[%d]=%d\n",i,data[i]);
    //}
    err=(cudaMemcpy( g_data+length, data, sizeof(datatype)*(len1-length), cudaMemcpyHostToDevice));
	if( err != cudaSuccess)
	{
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
     	exit(-1);
	}
    dim3 dimBlock(size/2,1,1);
    //dim3 dimGrid(firstBlocks,1);
    dim3 dimGrid(firstBlocksX,firstBlocksY);
    dim3 dimGrid2(secondBlocks,1);
    dim3 singleGrid(1,1);
    //warmup();
	cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event) ;
    cudaEventRecord(start_event, 0);
	//	
	//checkErrors("test0");
	//printf("firstblocks=%d\n",firstBlocks);
	scan1<<<dimGrid, dimBlock,(size+CONFLICT_FREE_OFFSET(size-1))*sizeof(datatype)>>>(g_data,firstOutput,size,1,firstBlocks);
	//checkErrors("test1");

	if(firstBlocks!=len2){
		diff=len2-firstBlocks;
		datatype* padding = (datatype*) malloc((diff)*sizeof(datatype));
		for(i=0;i<diff;i++){
			padding[i]=0;
		}
		err=((cudaMemcpy((firstOutput+firstBlocks), padding, sizeof(datatype)*diff, cudaMemcpyHostToDevice)));
	    if( err != cudaSuccess){
    		printf("CUDA error: %s\n", cudaGetErrorString(err));
     		exit(-1);
		} 
	}
	//checkErrors("test2");
	if(firstBlocks>size){ 
	    err=((cudaMalloc( (void**) &secondOutput, size*sizeof(datatype))));
    	if( err != cudaSuccess){
    		printf("CUDA error: %s\n", cudaGetErrorString(err));
     		exit(-1);
		} 
    	if(secondBlocks<size){
    		diff=size-secondBlocks;
    		datatype* padding1 = (datatype*) malloc(diff*sizeof(datatype));
    		for(i=0;i<diff;i++){
				padding1[i]=0;
			}
    		err=((cudaMemcpy((secondOutput+secondBlocks), padding1, sizeof(datatype)*diff, cudaMemcpyHostToDevice)));
    		if( err != cudaSuccess)
			{
    			printf("CUDA error: %s\n", cudaGetErrorString(err));
     			exit(-1);
			}  
    	}
    	scan1<<<dimGrid2, dimBlock,(size+CONFLICT_FREE_OFFSET(size-1))*sizeof(datatype)>>>(firstOutput,secondOutput,size,1,secondBlocks);
 	   	scan1<<<singleGrid, dimBlock,(size+CONFLICT_FREE_OFFSET(size-1))*sizeof(datatype)>>>(secondOutput,secondOutput,size,0,1);
    	scan2<<<dimGrid2, dimBlock>>>(firstOutput,secondOutput,size,secondBlocks);
    	scan2<<<dimGrid, dimBlock>>>(g_data,firstOutput,size,firstBlocks);
    }
    else{
    	scan1<<<singleGrid, dimBlock,(size+CONFLICT_FREE_OFFSET(size-1))*sizeof(datatype)>>>(firstOutput,firstOutput,size,0,1);
    	scan2<<<dimGrid, dimBlock>>>(g_data,firstOutput,size,firstBlocks);
    }

	free(data); 
	//free(data_out);
    /*err=cudaFree(g_data);
    if( err != cudaSuccess){
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
     	return -1;
	} */ 
    err=cudaFree(firstOutput);
    if( err != cudaSuccess){
    	printf("CUDA error: %s\n", cudaGetErrorString(err));
     	exit(-1);
	} 
}
