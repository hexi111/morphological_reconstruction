// Efficient radix sort implementation on CUDA

// Satish, N., Harris, M., and Garland, M. "Designing Efficient Sorting 
// Algorithms for Manycore GPUs". In Proceedings of IEEE International
// Parallel & Distributed Processing Symposium 2009 (IPDPS 2009).


#include <stdio.h>
#include <stdlib.h>
#include "scan4.h"
#include "MorphologicReconGPU.h"
#include "radixSort.h"
typedef unsigned int uint;


#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS 
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else 
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS) 
#endif
	
//#define NUM_OF_ELEMENTS 512 //equal to size
//#define CONFLICT_FREE_OFFSET_N 31 

#define NUM_OF_ELEMENTS 1024
#define NUM_OF_ELEMENTS_2 2048
#define CONFLICT_FREE_OFFSET_N 63

//changes needs to make for different structures include the amount of shared memory allocated in global functions
// and the structure member name.
// should allocate enough memory for temp2 in localSort function

//#define CONFLICT_FREE_OFFSET(n) 0

void checkErrors(char *label)
{
  // we need to synchronise first to catch errors due to
  // asynchroneous operations that would otherwise
  // potentially go unnoticed

  cudaError_t err;

  err = cudaThreadSynchronize();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    char *e = (char*) cudaGetErrorString(err);
    fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
  }
}

__global__ void LocalSort(datatype_search *g_data,int *histogram,int *position,int size,int round,int numOfBlocks){
	
	//extern __shared__ char temp[]; //allocated on invocation 
	__shared__ datatype_search in_c[NUM_OF_ELEMENTS];
	__shared__ datatype_search out_c[NUM_OF_ELEMENTS];
	__shared__ int temp2[NUM_OF_ELEMENTS_2];
	__shared__ int temp3[NUM_OF_ELEMENTS+CONFLICT_FREE_OFFSET_N];
	int tx=threadIdx.x;
	datatype_search * in=in_c;		
	datatype_search * out=out_c; 
	//datatype_search * in=(datatype_search *)temp;		
	//datatype_search * out=(datatype_search *)(temp+8*size); 
	//for exchange
	datatype_search * temp1;	
	// for histogram
	//int * temp2=(int*)(temp+8*size);	
	// for scan
	//int * temp3=(int *)(temp+16*size);
	//int bx=blockIdx.x;
	int bx=blockIdx.x+blockIdx.y*1024;
	if(bx<numOfBlocks){
		int base=bx*size;
		int thid=tx+base;
		int i;
		//int out=size,in=0;
		int m=size/2;
		int offset;
		in[tx] = g_data[thid];
		//temp[tx+m/2] = g_data[thid+m/2];
		in[tx+m]=g_data[thid+m];
		//temp[tx+3*m/2]=g_data[thid+3*m/2];
		int total;
		int b;
		for(i=0;i<BIT;i++){
			int a1 = tx; 
			int b1 = tx + m;
			int bankOffsetA = CONFLICT_FREE_OFFSET(a1); 
			int bankOffsetB = CONFLICT_FREE_OFFSET(b1);
			temp3[tx+bankOffsetA]=(1-((in[tx].FIELD1>>(i+round))&0x00000001));
			temp3[tx+m+bankOffsetB]=(1-((in[tx+m].FIELD1>>(i+round))&0x00000001));
			offset=1;
			for (int d = size>>1; d > 0; d >>= 1){ // build sum in place up the tree {
				__syncthreads();
				if (tx < d) {
					int a2 =offset*(2*tx+1)-1;
					int b2 =offset*(2*tx+2)-1;
					a2+=CONFLICT_FREE_OFFSET(a2);
					b2+=CONFLICT_FREE_OFFSET(b2);
					temp3[b2]+=temp3[a2];
				}
				offset*=2;
			}
			if(tx==0){
				temp3[size-1+CONFLICT_FREE_OFFSET(size-1)]=0; 
			}
			for(int d=1;d<size;d*=2){
				offset>>=1;
				__syncthreads();
				if(tx<d){
					int a3 =offset*(2*tx+1)-1;
					int b3 =offset*(2*tx+2)-1;
					a3+=CONFLICT_FREE_OFFSET(a3);
					b3+=CONFLICT_FREE_OFFSET(b3);
					int t=temp3[a3];
					temp3[a3]=temp3[b3];
					temp3[b3]+=t;
				}
			}	
			__syncthreads();
			total=temp3[size-1+CONFLICT_FREE_OFFSET(size-1)]+(1-((in[size-1].FIELD1>>(i+round))&0x00000001));
			__syncthreads();
			b=((in[tx].FIELD1>>(i+round))&0x00000001);
			temp3[tx+bankOffsetA]=b?(tx-temp3[tx+bankOffsetA]+total):(temp3[tx+bankOffsetA]);
			b=((in[tx+m].FIELD1>>(i+round))&0x00000001);
			temp3[tx+m+bankOffsetB]=b?(tx+m-temp3[tx+m+bankOffsetB]+total):(temp3[tx+m+bankOffsetB]);
			out[temp3[tx+bankOffsetA]]=in[tx];
			out[temp3[tx+m+bankOffsetB]]=in[tx+m];
			temp1=in;
			in=out;
			out=temp1;
			__syncthreads(); 
		}
		g_data[thid]=in[tx];
		g_data[thid+m]=in[tx+m];
		
		// computer the in-block histogram: 128 groups * 16
		temp2[tx]=0;
		temp2[tx+m]=0;
		temp2[tx+size]=0;
		temp2[tx+size+m]=0;	
		__syncthreads(); 	
		
		if(tx<(size/8)){
			for(i=0;i<8;i++){
				offset=in[tx*8+i].FIELD1;
				temp2[tx+(size/8)*((offset>>round)&0x0000000F)]++;
			}
		}
		__syncthreads(); 
		int threadEachRow=(size/32);
		int row=tx/threadEachRow;
		int index=tx%threadEachRow;
		int loc=(size/8)*row;
		temp2[index+loc]+=temp2[index+loc+threadEachRow];
		temp2[index+loc]+=temp2[index+loc+threadEachRow+threadEachRow];
		temp2[index+loc]+=temp2[index+loc+threadEachRow+threadEachRow+threadEachRow];
		__syncthreads(); 
		offset=size/32;		
		while(offset>1){
			offset=(offset>>1);
			if(index<offset){
				temp2[loc+index]=temp2[loc+index+offset]+temp2[loc+index];
			}
			__syncthreads(); 
		}
		if(index==0){
			histogram[row*numOfBlocks+bx]=temp2[loc];
			position[row*numOfBlocks+bx]=temp2[loc];
		}
	}
}
__global__ void Reorder(datatype_search *g_data,datatype_search *g_data_out,int *histogram,int *position,int size,int round,int numOfBlocks){
	//extern __shared__ char tempr[]; // allocated on invocation 
	__shared__ datatype_search tempr1[NUM_OF_ELEMENTS];
	__shared__ int tempr2[16];
	int tx=threadIdx.x;
	//datatype_search * tempr1=(datatype_search *)tempr;
	//int * tempr2=(int *)(tempr+8*size); 
	//int bx=blockIdx.x;
	int bx=blockIdx.x+blockIdx.y*1024;
	if(bx<numOfBlocks){
		int base=bx*size;
		int thid = tx+base;
		tempr1[tx] = g_data[thid];
		if(tx<16){
			tempr2[tx]=histogram[tx*numOfBlocks+bx];
		}
		if(tx==0){
			for(int i=1;i<16;i++){
				tempr2[i]=tempr2[i]+tempr2[i-1];
			}
		}
		__syncthreads();
		int index=(tempr1[tx].FIELD1>>round)&0x0000000F;
		int pos=position[index*numOfBlocks+bx]-(tempr2[index]-tx);
		g_data_out[pos]=tempr1[tx];
	}
}

// 排序
// g_data: gpu内的数据  length: 数据个数  size: 线程数目
void sortGM_struct(datatype_search *g_data,int length,int size){
	int i; 
    int sizeOfScan=1024;
    int numOfBlocks=(length-1)/size+1;
    int numOfBlocksX=1024;
    //int numOfBlocksY=(numOfBlocks-1)/1024+1;
    //int numOfBlocksX=1024;
    int numOfBlocksY;
    //=(numOfBlocks-1)/1024+1;
    if(numOfBlocks!=1){
		numOfBlocksY=(numOfBlocks-1)/1024+1;
    }
    else{
    	numOfBlocksY=1;
    }
    int len=numOfBlocks*size;
    //unsigned long long max=0xffffffffffffffff;
	unsigned char max=255;

    datatype_search* data = (datatype_search*) malloc((len-length)*sizeof(datatype_search));
    //printf("size=%d\n",sizeof(datatype_search));
    datatype_search* g_data_out=0;
    int* histogram=0;
    int* position=0;
    int numOfHistogram=numOfBlocks*16;
    int numOfActualHistogram=((numOfHistogram-1)/sizeOfScan+1)*sizeOfScan;
    (cudaMalloc( (void**) &g_data_out, len*sizeof(datatype_search)));
    (cudaMalloc( (void**) &histogram, numOfActualHistogram*sizeof(int)));
    (cudaMalloc( (void**) &position, numOfActualHistogram*sizeof(int)));
    //checkErrors("Memory copy 222");	
    //srand(time(NULL));
    //srand(10000);

    for(i = length; i < len; i++) 
    {
        data[i-length].FIELD1 = max;//(int)(10 * rand()/32768.f);
    }
    //#ifdef TEST
    	//printf("length=%d,len=%d\n",length,len);
    //#endif
    //checkErrors("Memory copy 12-1");
    (cudaMemcpy( g_data+length, data, sizeof(datatype_search)*(len-length), cudaMemcpyHostToDevice));
    checkErrors("Memory copy 12");
    dim3 dimBlock(size,1,1); 
    dim3 dimBlock_localsort(size/2,1,1);

    dim3 dimGrid(numOfBlocksX,numOfBlocksY);
	
	
    for(i=0;i<(BYTES/8);i=i+2){
		LocalSort<<<dimGrid,dimBlock_localsort>>>(g_data,histogram,position,size,i*4,numOfBlocks);
		scan(position,numOfHistogram,sizeOfScan);
		//scan1<<<dimGridScan, dimBlockScan,(2*size*sizeof(datatype_search))>>>(histogram,position,firstOutput,numOfBlocks*BYTES,size,1);
		//scan1<<<singleGrid, dimBlockScan,(2*size*sizeof(datatype_search))>>>(firstOutput,firstOutput,firstOutput,firstBlocks,size,0);
    	//scan2<<<dimGridScan, dimBlockScan>>>(position,firstOutput,len,size,firstBlocks);
		Reorder<<<dimGrid, dimBlock>>>(g_data,g_data_out,histogram,position,size,i*4,numOfBlocks);
	
		LocalSort<<<dimGrid,dimBlock_localsort>>>(g_data_out,histogram,position,size,(i+1)*4,numOfBlocks);
		scan(position,numOfHistogram,sizeOfScan);
		//scan1<<<dimGridScan, dimBlockScan,(2*size*sizeof(datatype_search))>>>(histogram,position,firstOutput,numOfBlocks*BYTES,size,1);
		//scan1<<<singleGrid, dimBlockScan,(2*size*sizeof(datatype_search))>>>(firstOutput,firstOutput,firstOutput,firstBlocks,size,0);
    	//scan2<<<dimGridScan, dimBlockScan>>>(position,firstOutput,len,size,firstBlocks);
		Reorder<<<dimGrid, dimBlock>>>(g_data_out,g_data,histogram,position,size,(i+1)*4,numOfBlocks);
	}
	
	free(data); 
    cudaFree(g_data_out);
    cudaFree(position);
    cudaFree(histogram);
}
