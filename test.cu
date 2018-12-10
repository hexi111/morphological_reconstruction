#include <iostream>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

#include "test.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

void testScan();
__device__ void scan_include(int * values,int * exclusive);
__global__ void testScan_include_kernel(int * in, int * out);
void testMerge();
__global__ void testMerge_kernel(PPQ_ITEM * A, int numA,PPQ_ITEM * B,int numB,PPQ_ITEM * C);
__device__ void merge(PPQ_ITEM * tempA, int numA, PPQ_ITEM * tempB, int numB,PPQ_ITEM * C);
void testSort();
__global__ void testSort_kernel_bitonic(PPQ_ITEM * A,PPQ_ITEM * B, int num );
__device__ void swap(PPQ_ITEM * buffer, int source, int target, int ddd);
__device__ void bitonic_sort(PPQ_ITEM * buffer,int arrayLength,int dir);

int main (int argc, char **argv){
	//testScan();
	//testMerge();
	testSort();
	return 0;
}

void testScan(){
	srand(time(NULL));
    int i;
    int * in=(int * )malloc(NUM_THREADS*sizeof(int));
    int * prefix_in=(int * )malloc(NUM_THREADS*sizeof(int));

    int * out=(int * )malloc(NUM_THREADS*sizeof(int));
    for(i = 0; i < NUM_THREADS; i++) {
        in[i]=(rand() % 256);
    }
    for(i=0;i<NUM_THREADS;i++){
    	printf("%d ",in[i]);
    }
    printf("\n");

    prefix_in[0]=in[0];
    for(i=1;i<NUM_THREADS;i++){
    	prefix_in[i]=prefix_in[i-1]+in[i];
    }
    for(i=0;i<NUM_THREADS;i++){
    	printf("%d ",prefix_in[i]);
    }
    printf("\n");
    
    int * in_gpu;
    int * out_gpu;
    (cudaMalloc( (void**) &in_gpu, NUM_THREADS*sizeof(int)));
    (cudaMemcpy( in_gpu, in, sizeof(int)*NUM_THREADS, cudaMemcpyHostToDevice));
    (cudaMalloc( (void**) &out_gpu, NUM_THREADS*sizeof(int)));
	testScan_include_kernel<<<1,NUM_THREADS>>>(in_gpu,out_gpu);
    (cudaMemcpy( out, out_gpu, sizeof(int)*NUM_THREADS, cudaMemcpyDeviceToHost));
	
	int count=0;
	for(i=0;i<NUM_THREADS;i++){
		if(out[i]!=prefix_in[i]){
     		count++;
     	}
	}

	printf("count=%d\n",count);
	printf("\n");
	for(i=0;i<NUM_THREADS;i++){
		printf("%d ",out[i]);    	
	}
}

__device__ void scan_include(int * values,int * inclusive) {
	int tx = threadIdx.x;
	int pout = 0, pin = 1;
	int * p[2];
	p[0]=values;
	p[1]=inclusive;

	for (int offset = 1; offset < NUM_THREADS; offset *= 2) {
		pout = 1 - pout; // swap double buffer indices 
		pin = 1 - pout;
		if (tx >= offset) {
			p[pout][tx] = p[pin][tx - offset]+ p[pin][tx];
		}
		else {
			p[pout][tx] = p[pin][tx];
		}
		__syncthreads();
	}
	if(p[pout]!=inclusive){
		p[pin][tx]=p[pout][tx];
	}
	__syncthreads();
}

__global__ void testScan_include_kernel(int * in, int * out){
	int tx=threadIdx.x;
	__shared__ int values[NUM_THREADS];
	__shared__ int inclusive[NUM_THREADS];
	values[tx]=in[tx];
	__syncthreads();
	//void scan(int *g_data,int length,int size){

	scan_include(values,inclusive);
	out[tx]=inclusive[tx];	
}

void testMerge(){
	int i;
	int numA=NUM_THREADS;
	int numB=NUM_THREADS;
	PPQ_ITEM * data1 = (PPQ_ITEM*) malloc(numA*sizeof(PPQ_ITEM));
	PPQ_ITEM * data2 = (PPQ_ITEM*) malloc(numB*sizeof(PPQ_ITEM));
	PPQ_ITEM * data3 = (PPQ_ITEM*) malloc((numA+numB)*sizeof(PPQ_ITEM));
	PPQ_ITEM * A;
	PPQ_ITEM * B;
	PPQ_ITEM * C;
	cudaError_t err;
	
	for(i = 0; i <= 255; i++) {
        data1[i].key= (255-i);
    }
    for(i = 256; i < numA; i++){
        data1[i].key= 0;
    }
	for(i = 0; i <= 255; i++) {
        data2[i].key= (255-i);
    }
    for(i = 256; i < numB; i++){
        data2[i].key= 0;
    }

    (cudaMalloc( (void**) &A, numA*sizeof(PPQ_ITEM)));
    (cudaMalloc( (void**) &B, numB*sizeof(PPQ_ITEM)));
    (cudaMalloc( (void**) &C, (numA+numB)*sizeof(PPQ_ITEM)));
    err=cudaMemcpy( A, data1, sizeof(PPQ_ITEM)*numA, cudaMemcpyHostToDevice) ;
    err=cudaMemcpy( B, data2, sizeof(PPQ_ITEM)*numB, cudaMemcpyHostToDevice) ;
    if( err != cudaSuccess){
    	printf("CUDA error t1: %s\n", cudaGetErrorString(err));
     	exit(1);
     }
    uint64_t t31, t32;
	t31 = cci::common::event::timestampInUS();
    testMerge_kernel<<<1,NUM_THREADS>>>(A, numA,B,numB,C);
    cudaDeviceSynchronize();
    t32 = cci::common::event::timestampInUS(); 
	cout << " Mering "<< t32 - t31<< "us" <<endl;
    
    err=cudaMemcpy( data3, C, sizeof(PPQ_ITEM)*(numA+numB), cudaMemcpyDeviceToHost) ;
    if( err != cudaSuccess){
    	printf("CUDA error t2: %s\n", cudaGetErrorString(err));
     	exit(1);
	}
    for(i=0;i<numA;i++){
    	printf("%d ",data1[i].key);
    }
    printf("\n");
    for(i=0;i<numB;i++){
    	printf("%d ",data2[i].key);
    }
	printf("\n");
	printf("***********\n");
    for(i=0;i<(numA+numB);i++){
    	printf("%d ",data3[i].key);
    }
	printf("\n");
	printf("***********\n");
    for(i=1;i<(numA+numB);i++){
    	if(data3[i].key>data3[i-1].key){
    		printf("data[%d](%d)>data[%d-1](%d)\n",i,data3[i].key,i,data3[i-1].key);
    		break;
    	}
    }
}

__global__ void testMerge_kernel(PPQ_ITEM * A, int numA,PPQ_ITEM * B,int numB,PPQ_ITEM * C){
	__shared__ PPQ_ITEM first[NUM_THREADS_2];
	__shared__ PPQ_ITEM second[NUM_THREADS_2];
	__shared__ PPQ_ITEM third[NUM_THREADS_4];
	int i;
	int tx=threadIdx.x;
	for(i=tx;i<numA;i+=NUM_THREADS){
		first[i]=A[i];
	}
	for(i=tx;i<numB;i+=NUM_THREADS){
		second[i]=B[i];
	}	
	__syncthreads();
	merge(first, numA, second, numB, third);
	for(i=tx;i<(numA+numB);i++){
		C[i]=third[i];
	}
}

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
			C[i+j+1]=tempB[i];
		}
	}	 
	__syncthreads();
} 

void testSort(){

    int i; 
	int num=NUM_THREADS;
    PPQ_ITEM * in = (PPQ_ITEM*) malloc(num*sizeof(PPQ_ITEM));
    PPQ_ITEM * out = (PPQ_ITEM*) malloc(num*sizeof(PPQ_ITEM));

	PPQ_ITEM * g_data_in;
	PPQ_ITEM * g_data_out;

    srand(time(NULL));
    for(i = 0; i < num; i++) {
        in[i].key=(rand() % 256);
    }

    for(i=0;i<num;i++){	
    	printf("%d ",in[i].key);
    }	
    
    (cudaMalloc( (void**) &g_data_in, num*sizeof(PPQ_ITEM)));
    (cudaMemcpy( g_data_in, in, sizeof(PPQ_ITEM)*num, cudaMemcpyHostToDevice));
    (cudaMalloc( (void**) &g_data_out, num*sizeof(PPQ_ITEM)));
    	
    uint64_t t11, t12;	
    t11 = cci::common::event::timestampInUS();
	for(i=0;i<1000;i++){
 	   testSort_kernel_bitonic<<<1,(NUM_THREADS)>>>(g_data_in,g_data_out,num);
    }
    cudaDeviceSynchronize();
	t12 = cci::common::event::timestampInUS(); 
	cout << " bitonic sort took "<< t12 - t11<< "us" <<endl;

		
	(cudaMemcpy( out,g_data_out, sizeof(PPQ_ITEM)*num, cudaMemcpyDeviceToHost));

	printf("\n");
    for(i=0;i<num;i++){
    	printf("%d ",out[i].key);
    }
	printf("\n");
    for(i=1;i<num;i++){   	
    	if(out[i].key<out[i-1].key){
    		printf("data_out[%d](%d)<data_out[%d](%d)\n",i,out[i].key,(i-1),out[i-1].key);
			break;
    	}
    }    

}

__global__ void testSort_kernel_bitonic(PPQ_ITEM * A,PPQ_ITEM * B, int num ){
	int i;
	int tx=threadIdx.x;
	__shared__ PPQ_ITEM buffer[NUM_THREADS];
	for(i=tx;i<num;i+=(NUM_THREADS)){
		buffer[i]=A[i];
	}
	__syncthreads();
	bitonic_sort(buffer,num,1);
	for(i=tx;i<num;i+=(NUM_THREADS)){
		B[i]=buffer[i];	
	}
}

// ddd 1:生序 0: 降序
__device__ void swap(PPQ_ITEM * buffer, int source, int target, int ddd){
	PPQ_ITEM temp;
	if(((buffer[source].key>buffer[target].key)&&(ddd)) || ((buffer[source].key<buffer[target].key)&&(1-ddd))){
		temp=buffer[source];
		buffer[source]=buffer[target];
		buffer[target]=temp;
	}
}

// dir 1:生序 0: 降序
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

void testInsertUpdate(){

	PPQ_TYPE *ppqs=(PPQ_TYPE *)malloc(NUM_OF_BLOCKS*sizeof(PPQ_TYPE));
	int i;
	cudaMalloc((void**)&(ppqs[0].parHeap),sizeof(PPQ_ITEM)*200);	
	
	PPQ_ITEM *d_queue_cpu = NULL;	
	d_queue_cpu=(PPQ_ITEM *)malloc(sizeof(PPQ_ITEM) * 30);
	
	unsigned char keys[]={255,254,251,243,237,198,178,177,176,162,158,155,154,153,140,130,109,99,92,80,82,87,92,98,106,115,170,190,211,234};
	int indices[]={23,45,14,16,35,27,38,29,30,19,17,16,14,21,63,65,72,75,83,81,123,421,563,732,64,89,40,28,91,130};
	
	for(i=0;i<30;i++){
		d_queue_cpu[i].index=indices[i];
		d_queue_cpu[i].key=keys[i];
	}
		
	cudaMemcpy( ppqs[0].parHeap, d_queue_cpu, sizeof(PPQ_ITEM) * 20, cudaMemcpyHostToDevice) ;
	PPQ_ITEM *d_queue = NULL;
	cudaMalloc( (void**) &d_queue, sizeof(PPQ_ITEM) * 8) ;
	cudaMemcpy( d_queue, d_queue_cpu+20, sizeof(PPQ_ITEM) * 8, cudaMemcpyHostToDevice) ;

	PPQ_TYPE *ppqs_gpu;
	cudaMalloc((void**)&ppqs_gpu,sizeof(PPQ_TYPE));	
	cudaMemcpy( ppqs_gpu, ppqs, sizeof(PPQ_TYPE) , cudaMemcpyHostToDevice ) ;
	
	testInsertUpdate_kernel<<<1,NUM_THREADS>>>(ppqs_gpu, 8, 20, d_queue);
	PPQ_ITEM * data_output;
	data_output=(PPQ_ITEM *)malloc(sizeof(PPQ_ITEM) * 28);
	cudaMemcpy( data_output, ppqs[0].parHeap, sizeof(PPQ_ITEM) * 28, cudaMemcpyDeviceToHost) ;
}

__global__ void testInsertUpdate_kernel(PPQ_TYPE * ppqs, int leftover, int numOfItems, PPQ_ITEM *d_queue){
	
	int tx=threadIdx.x;
	int i;
	__shared__ PPQ_ITEM buffer[10*NUM_THREADS];
	__shared__ PPQ_ITEM *p1;
	__shared__ PPQ_ITEM *p2;
	__shared__ PPQ_ITEM *p3;
	__shared__ PPQ_ITEM *p4;
	__shared__ PPQ_ITEM *p5;
	__shared__ PPQ_ITEM *p6;
 
	if(tx==0){
		p1=buffer+NUM_THREADS_4;
		p2=p1+NUM_THREADS;
		p3=p2+NUM_THREADS;
		p4=p3+NUM_THREADS;
		p5=p4+NUM_THREADS;
		p6=p5+NUM_THREADS;
	}
	
	for(i=tx;i<leftover;i+=NUM_THREADS){
		buffer[i]=d_queue[i];
	}
	
	__syncthreads();
	insertUpdate(ppqs,leftover,numOfItems,buffer,p1,p2,p3,p4,p5,p6,count);
	numOfItems+=total;	
}

// 假设要插入的元素都在buffer里.
// 变量leftover是要插入元素的数目.
// 变量numOfItems是优先队列里元素的总数.
// 每个节点里元素的数目等于线程块里线程的数目.
__device__ void insertUpdate(PPQ_TYPE * ppqs,int leftover,int numOfItems,PPQ_ITEM * buffer,PPQ_ITEM *p1,PPQ_ITEM *p2,PPQ_ITEM *p3,PPQ_ITEM *p4,PPQ_ITEM *p5,int * count){
	
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int numOfNodes = numOfItems / NUM_THREADS; // 节点树
	int offset = numOfItems % NUM_THREADS; // 最后一个节点里元素数目
	int num; //最后一个节点要插入新元素的数目
	int parent,current;
	int next=numOfNodes+1;
	
	// 如果最后一个节点不是空的
	if(offset!=0){
		if(leftover>=(NUM_THREADS-offset)){ //如果最后一个节点不能容纳要插入元素
			num=NUM_THREADS-offset;
		}
		else{ //如果可以容纳
			num=leftover;
		}
		/* newly added start */
		if(tx < num){
			p3[tx]=buffer[leftover-1-tx];
		}
		__syncthreads();
		sort_one(p3,num,count);		
		if(tx < num){
			p1[tx] = p3[num-1-tx];	
		}
		/* newly added end */
		leftover-=num;
		if(tx<offset){
			p2[tx]=ppqs[bx].parHeap[numOfItems-offset+tx];
		}

		merge(p1,num,p2,offset,p4);	//合并最后一个节点之前的元素和新加入的元素
		current=next;
		parent=current/2;
		// 调整
		if(parent>0){
			num=num+offset;
			p5[tx]=ppqs[bx].parHeap[((parent-1)<<SHIFT_DIGITS)+tx];
			merge(p4,num,p5,NUM_THREADS,p1);	
			if(tx<num){
				ppqs[bx].parHeap[((current-1)<<SHIFT_DIGITS)+tx]=p2[tx];
			}
			current=parent;
			parent=current/2;
			p4[tx]=p1[tx];	
		}
		while(parent>0){
			p5[tx]=ppqs[bx].parHeap[((parent-1)<<SHIFT_DIGITS)+tx];
			merge(p4,NUM_THREADS,p5,NUM_THREADS,p1);	
			ppqs[bx].parHeap[((current-1)<<SHIFT_DIGITS)+tx]=p2[tx];
			p4[tx]=p1[tx];
			current=parent;
			parent=current/2;
		}
		ppqs[bx].parHeap[tx]=p1[tx];
		//continue to the next node
		next++;
		//leftover-=num;
	}
	num=(leftover>=NUM_THREADS?NUM_THREADS:leftover);
	while(leftover>0){
		current=next;
		parent=current/2;
		if(parent>0){
			//if(tx<num){
				//p1[tx]=buffer[leftover-1-tx];	
			//}
			/* newly added start */
			if(num<NUM_THREADS){
				if(tx<num){
					p3[tx]=buffer[leftover-1-tx];
				}
				#ifdef WITH_SYNC
				__syncthreads();
				#endif
				sort_one(p3,num,count);		
				//sort(p3,num);		
				if(tx<num){
					p1[tx]=p3[num-1-tx];	
				}
			}
			else{
				p1[tx]=buffer[leftover-1-tx];
				#ifdef WITH_SYNC
				__syncthreads();
				#endif
				bitonic_sort(p1,num,0);
			}
			/* newly added end */
			p2[tx]=ppqs[bx].parHeap[((parent-1)<<SHIFT_DIGITS)+tx];	
			merge(p1,num,p2,NUM_THREADS,p4);	
			if(tx<num){
				ppqs[bx].parHeap[((current-1)<<SHIFT_DIGITS)+tx]=p5[tx];
			}
			current=parent;
			parent=current/2;
			p1[tx]=p4[tx];	
		}
		while(parent>0){
			p2[tx]=ppqs[bx].parHeap[((parent-1)<<SHIFT_DIGITS)+tx];
			merge(p1,NUM_THREADS,p2,NUM_THREADS,p4);

			ppqs[bx].parHeap[((current-1)<<SHIFT_DIGITS)+tx]=p5[tx];
			current=parent;
			parent=current/2;
			p1[tx]=p4[tx];	
		}
		ppqs[bx].parHeap[tx]=p1[tx];
		next++;
		leftover-=num;	
		num=leftover>NUM_THREADS?NUM_THREADS:leftover;
	}
	__syncthreads();
}

