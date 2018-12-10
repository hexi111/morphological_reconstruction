#include <iostream>
#include "opencv2/cudaarithm.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

#include "MorphologicReconGPU.h"
#include "util.h"
#include "radixSort.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

__device__ void scan(int * values,int * exclusive);
__device__ void sort_one(int * buffer_index,unsigned char * buffer_key,int num,int * count);
__device__ void merge(int * tempA_index,unsigned char * tempA_key, int numA, int * tempB_index, unsigned char * tempB_key,int numB,int * C_index,unsigned char * C_key);

template <typename T> 
void preprocessing(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, int connectivity, int num_iterations, cudaStream_t stream);
template <typename T>
GpuMat imreconstruct(GpuMat &marker, GpuMat &mask, int connectivity,int nItFirstPass, Stream& stream);
template <typename T>
PPQ_TYPE * initHeap(T *marker, const T *mask, int sx, int sy, bool conn8,cudaStream_t stream);
__global__ void strucTransfer(PPQ_ITEM * d_queue_fit,int * d_queue_fit_index,unsigned char * d_queue_fit_key,int d_queue_size);
template <typename T>
__global__ void initQueuePixels(T *marker, const T *mask, int sx, int sy, bool conn8, PPQ_ITEM *d_queue, int *d_queue_size);
__global__ void distInitialItems(PPQ_TYPE * ppqs, int * d_queue_fit_index,unsigned char * d_queue_fit_key, int d_queue_size);
__global__ void reconstruction(PPQ_TYPE * ppqs, int * seeds, unsigned char * image, int ncols, int nrows);
template <typename T> 
__global__ void iRec1DForward_X_dilation ( T* marker, const T* mask, const unsigned int sx, const unsigned int sy, bool* change );
template <typename T>
__global__ void iRec1DBackward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change );
template <typename T>
__global__ void iRec1DForward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change );
template <typename T>
__global__ void iRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy, bool* __restrict__ change );
template <typename T>
__global__ void iRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy, bool* __restrict__ change );
template <typename T>
__global__ void iRec1DBackward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change );

void pixelConvertIntToChar(GpuMat& input, GpuMat&result, Stream& stream);
void convertIntToChar(int rows, int cols, int *input, unsigned char *result, cudaStream_t stream);
__global__ void intToCharKernel(int rows, int cols, int *input, unsigned char *result);
template <typename T>
__device__ bool checkCandidateNeighbor4(T *marker, const T *mask, int x, int y, int ncols, int nrows, T pval);
template <typename T>
__device__ bool checkCandidateNeighbor8(T *marker, const T *mask, int x, int y, int ncols, int nrows,T pval);
__device__ unsigned char min(unsigned char pval, unsigned char imageXYval);
__device__ unsigned char max(unsigned char pval, unsigned char imageXYval);
__device__ int propagate(int *seeds, unsigned char *image, int x, int y, int ncols, unsigned char pval,unsigned char * newVal);
__device__ void deleteUpdate(PPQ_TYPE * ppqs,int numOfItems,int *p1_index,int *p2_index,int *p3_index,int *p4_index,int *p5_index,int *p6_index,unsigned char *p1_key,unsigned char *p2_key,unsigned char *p3_key,unsigned char *p4_key,unsigned char *p5_key,unsigned char *p6_key);
__device__ void insertUpdate(PPQ_TYPE * ppqs,int leftover,int numOfItems,int * buffer_index,unsigned char * buffer_key,int *p1_index,int *p2_index,int *p3_index,int *p4_index,int *p5_index,unsigned char *p1_key,unsigned char *p2_key,unsigned char *p3_key,unsigned char *p4_key,unsigned char *p5_key,int * count);

int main (int argc, char **argv){
		
	if(argc != 3){
		printf("Usage: ./imreconTest <marker> <mask>");
		return -1; 
	}
	
	// imread是opencv里一个把图片读进矩阵的函数
	Mat marker = imread(argv[1], -1);
	Mat mask = imread(argv[2], -1);
	// 结果图片矩阵
	Mat recon;

	// 时间变量
	uint64_t t1, t2;
	
	// opencv中封装好的操控CUDA的函数
	Stream stream;
	// GPU中的图片矩阵矩阵
	GpuMat g_marker;
	GpuMat g_mask;
	GpuMat g_recon;

	// 上传图片矩阵到GPU中
	g_marker.upload(marker, stream);
	g_mask.upload(mask, stream);		
	stream.waitForCompletion();
	
	t1 = cci::common::event::timestampInUS();
	g_recon = imreconstruct<unsigned char>(g_marker, g_mask, 4, 8, stream);
	stream.waitForCompletion();
	t2 = cci::common::event::timestampInUS(); 
	cout << " gpu reconstruction took "<< t2-t1<< "us" <<endl;
	
	// 从GPU下载图片矩阵到CPU
	g_recon.download(recon);
	
	// imwrite是opencv里一个把图片矩阵写到文件中函数
	imwrite("img_result/out.jpg", recon);	
	g_recon.release();
	g_marker.release();
	g_mask.release();
}

__device__ void scan(int * values,int * exclusive) {
	int tx = threadIdx.x;
	int pout = 0, pin = 1;
	int * p[2];
	p[0]=values;
	p[1]=exclusive;
	// This is exclusive scan, so shift right by one and set first elt to 0 
	// temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0; 
	// p[pout][tx]=(tx>0)? p[pin][tx-1]:0;

	for (int offset = 1; offset < NUM_THREADS; offset *= 2) {
		pout = 1 - pout; // swap double buffer indices 
		pin = 1 - pout;
		if (tx >= offset) {
			p[pout][tx] = p[pin][tx - offset]+ p[pin][tx];
		}
		else {
			p[pout][tx] = p[pin][tx];
		}
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
	}
	if(p[pout]!=exclusive){
		p[pin][tx]=p[pout][tx];
	}
	#ifdef WITH_SYNC
	__syncthreads();
	#endif
}

__device__ void sort_one(int * buffer_index,unsigned char * buffer_key,int num,int * count){
	
	//volatile __shared__ int count[NUM_THREADS+NUM_THREADS];
	int tx=threadIdx.x;
	int pout = 0, pin = NUM_THREADS;
	int out=NUM_THREADS,in=0;
	int i,j;
	int offset;
	int totalOne;
	int b,d,f;
	
	for(i=0;i<BITS;i++){
	//for(i=0;i<2;i++){
		//Count the number of zeros
		for(j=tx;j<num;j=j+blockDim.x){
			count[pout+j]=(1-((buffer_key[in+j]>>i)&0x00000001));
		}
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
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
			#ifdef WITH_SYNC
			__syncthreads();
			#endif		 
		}
		//printf("num=%d\n",num);
		//printf("loc=%d\n",(pout+num-1));
		totalOne=count[pout+num-1];

		for(j=tx;j<num;j=j+blockDim.x){
			b=((buffer_key[in+j]>>i)&0x00000001);
			f=count[pout+j]-(1-b);
			if(b==0){
				d=f;
			}
			else{
				d=j+totalOne-f;
			}
			buffer_index[out+d]=buffer_index[in+j];
			buffer_key[out+d]=buffer_key[in+j];
		}
		out = NUM_THREADS - out; // swap double buffer indices 
		in = NUM_THREADS - out;
		#ifdef WITH_SYNC
		__syncthreads(); 
		#endif		 
	}
}

// Merging two descending sequences.
__device__ void merge(int * tempA_index,unsigned char * tempA_key, int numA, int * tempB_index, unsigned char * tempB_key,int numB,int * C_index,unsigned char * C_key){
	int tx=threadIdx.x;
	int size=blockDim.x;
	
	int i,j,k,mid;
	
	for(i=tx;i<numA;i+=size){
		j=0;
		k=numB-1;
		if(tempA_key[i]>=tempB_key[0]){
			C_index[i]=tempA_index[i];
			C_key[i]=tempA_key[i];
		}
		else if(tempA_key[i]<tempB_key[numB-1]){
			C_index[i+numB]=tempA_index[i];
			C_key[i+numB]=tempA_key[i];
		}
		else{
			while(j<k-1){
				mid=(j+k)/2;
				if(tempB_key[mid]>tempA_key[i]){
					j=mid;
				}
				else{
					k=mid;
				}
			}
			//printf("i=%d,j=%d,C=%d\n",i,j,tempA[i]);
			C_index[i+j+1]=tempA_index[i];
			C_key[i+j+1]=tempA_key[i];
		}
	}	
	for(i=tx;i<numB;i+=size){
		j=0;
		k=numA-1;
		if(tempB_key[i]>tempA_key[0]){
			C_index[i]=tempB_index[i];
			C_key[i]=tempB_key[i];
		}
		else if(tempB_key[i]<=tempA_key[numA-1]){
			C_index[i+numA]=tempB_index[i];
			C_key[i+numA]=tempB_key[i];
		}
		else{
			while(j<k-1){
				mid=(j+k)/2;
				if(tempA_key[mid]>=tempB_key[i]){
					j=mid;
				}
				else{
					k=mid;
				}
			}
			//printf("i=%d,j=%d,C=%d\n",i,j,tempB[i]);
			C_index[i+j+1]=tempB_index[i];
			C_key[i+j+1]=tempB_key[i];
		}
	}	 
	#ifdef WITH_SYNC
	__syncthreads();
	#endif
}

/*
形态学重建主函数，包括
数据的拷贝，预处理，建堆，迭代扩散
connectivity：是有4个邻居，还是8个邻居
nItFirstPass: 预处理需要扫描多少轮
*/
template <typename T>
GpuMat imreconstruct(GpuMat &marker, GpuMat &mask, int connectivity,int nItFirstPass, Stream& stream ) {
	
	CV_Assert(marker.size() == mask.size());
	CV_Assert(marker.channels() == 1);
	CV_Assert(mask.channels() == 1);
	CV_Assert(marker.type() == CV_8UC1);
	CV_Assert(mask.type() == CV_8UC1); // CV_8UC1 == unsigned char

	// 在gpu里分配两块连续的空间。在上面复制marker和mask
	GpuMat g_marker =  createContinuous(marker.size(), marker.type());
	marker.copyTo(g_marker,stream);
	GpuMat g_mask = createContinuous(mask.size(), mask.type());
	mask.copyTo(g_mask,stream);
	stream.waitForCompletion();

	// 预处理
	preprocessing<T>(g_marker.data, g_mask.data, g_mask.cols, g_mask.rows, connectivity,nItFirstPass, StreamAccessor::getStream(stream));
	stream.waitForCompletion();

	// 建堆
	PPQ_TYPE * ppqs=initHeap<T>(g_marker.data,g_mask.data,g_mask.cols, g_mask.rows, (connectivity==8),StreamAccessor::getStream(stream));
	stream.waitForCompletion();

	// 在gpu里分配两块连续的空间。把marker和mask上的数据从字符转成整数.
	GpuMat g_markerInt;
	g_markerInt = createContinuous(marker.size(), CV_32S);
	GpuMat g_maskInt;
	g_maskInt = createContinuous(mask.size(), CV_32S);
	
	g_marker.convertTo(g_markerInt, CV_32S);
	g_mask.convertTo(g_maskInt, CV_32S);
	
	reconstruction<<<NUM_OF_BLOCKS,NUM_THREADS>>>(ppqs, (int*)g_markerInt.data, g_mask.data, g_mask.cols, g_mask.rows);
	stream.waitForCompletion();

	// 在GPU里转换一个矩阵的整数值为字符值，并赋給另外一个矩阵
	pixelConvertIntToChar(g_markerInt, g_marker, stream);
	
	g_mask.release();
	g_markerInt.release(); 
	g_maskInt.release(); 
	return g_marker;
}

/*
预处理函数 
处理一些异常的点
*/
template <typename T> 
void preprocessing(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, int connectivity, int num_iterations, cudaStream_t stream){
	// setup execution parameters
	bool conn8 = (connectivity == 8);
	
	dim3 threadsx( XX_THREADS, XY_THREADS );
	dim3 blocksx( (sy + threadsx.y - 1) / threadsx.y );
	
	dim3 threadsy( MAX_THREADS );
	dim3 blocksy( (sx + threadsy.x - 1) / threadsy.x );
	
	bool *d_change;
	cudaMalloc( (void**) &d_change, sizeof(bool) ) ;

	// stability detection
	unsigned int iter = 0;
	
	if (conn8) {
		while(iter < num_iterations){
			iter++;
			
			iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
			iRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );
			iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
			iRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			if (stream == 0) cudaDeviceSynchronize();
			else  cudaStreamSynchronize(stream);
		}
	} else {
		while(iter < num_iterations){
			iter++;

			iRec1DForward_X_dilation <<< blocksx, threadsx>>> ( marker, mask, sx, sy, d_change );
			cudaDeviceSynchronize();

			iRec1DForward_Y_dilation <<< blocksy, threadsy>>> ( marker, mask, sx, sy, d_change );
			cudaDeviceSynchronize();

			iRec1DBackward_X_dilation<<< blocksx, threadsx>>> ( marker, mask, sx, sy, d_change );
			cudaDeviceSynchronize();

			iRec1DBackward_Y_dilation<<< blocksy, threadsy>>> ( marker, mask, sx, sy, d_change );
			cudaDeviceSynchronize();
			
			if (stream == 0) cudaDeviceSynchronize();
			else  cudaStreamSynchronize(stream);
		}
	}
	
	cudaDeviceSynchronize();
}

template <typename T>
__global__ void
iRec1DForward_X_dilation ( T* marker, const T* mask, const unsigned int sx, const unsigned int sy, bool* change ){

	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int ychunk = WARP_SIZE / XX_THREADS;
	const unsigned int xstop = sx - WARP_SIZE;
	
	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	volatile unsigned int s_change = 0;
	T s_old, s_new;
	unsigned int startx;
	unsigned int start;

	s_marker[threadIdx.y][WARP_SIZE] = 0;  // only need x=0 to be 0

	// the increment allows overlap by 1 between iterations to move the data to next block.
	for (startx = 0; startx < xstop; startx += WARP_SIZE) {
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
		s_marker[threadIdx.y][0] = s_marker[threadIdx.y][WARP_SIZE];
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x+1] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x+1] = mask[start + i*sx];
		}
		if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
			for (unsigned int i = 1; i <= WARP_SIZE; ++i) {
				s_old = s_marker[threadIdx.y][i];
				s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
				s_change |= s_new ^ s_old;
				s_marker[threadIdx.y][i] = s_new;
			}
		}
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x+1];
		}
	}

	if (startx < sx) {
		s_marker[threadIdx.y][0] = s_marker[threadIdx.y][sx-startx];  // getting ix-1st entry, which has been offsetted by 1 in s_marker
		// shared mem copy
		startx = sx - WARP_SIZE;
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;

		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x+1] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x+1] = mask[start + i*sx];
		}

		if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
			for (unsigned int i = 1; i <= WARP_SIZE; ++i) {
				s_old = s_marker[threadIdx.y][i];
				s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
				s_change |= s_new ^ s_old;
				s_marker[threadIdx.y][i] = s_new;
			}
		}
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x+1];
		}
	}

	if (s_change > 0) *change = true;
}

template <typename T>
__global__ void
iRec1DBackward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change ){

	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int ychunk = WARP_SIZE / XX_THREADS;
	const unsigned int xstop = sx - WARP_SIZE;

	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	volatile unsigned int s_change = 0;
	T s_old, s_new;
	int startx;
	unsigned int start;
	
	s_marker[threadIdx.y][0] = 0;  // only need x=WARPSIZE to be 0

	// the increment allows overlap by 1 between iterations to move the data to next block.
	for (startx = xstop; startx > 0; startx -= WARP_SIZE) {
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
		s_marker[threadIdx.y][WARP_SIZE] = s_marker[threadIdx.y][0];

		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x] = mask[start + i*sx];
		}

		if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
				for (int i = WARP_SIZE - 1; i >= 0; --i) {
					s_old = s_marker[threadIdx.y][i];
					s_new = min( max( s_marker[threadIdx.y][i+1], s_old ), s_mask[threadIdx.y][i] );
					s_change |= s_new ^ s_old;
					s_marker[threadIdx.y][i] = s_new;
				}
		}

		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x];
		}
		
	}

	if (startx <= 0) {
		s_marker[threadIdx.y][WARP_SIZE] = s_marker[threadIdx.y][-startx];  // getting ix-1st entry, which has been offsetted by 1 in s_marker
		startx = 0;
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;

		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x] = mask[start + i*sx];
		}

		if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
			for (int i = WARP_SIZE - 1; i >= 0; --i) {
				s_old = s_marker[threadIdx.y][i];
				s_new = min( max( s_marker[threadIdx.y][i+1], s_old ), s_mask[threadIdx.y][i] );
				s_change |= s_new ^ s_old;
				s_marker[threadIdx.y][i] = s_new;
			}
		}

		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x];
		}
	}

	if (s_change > 0) *change = true;
}

template <typename T>
__global__ void
iRec1DForward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change ){
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	unsigned int  s_change = 0;
	T s_old, s_new, s_prev;
	
	if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = 0; iy < sy; ++iy) {
			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_change |= s_old ^ s_new;
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;

		}
	}	
	if (s_change != 0) *change = true;
}

template <typename T>
__global__ void
iRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy, bool* __restrict__ change ){

	// best thing to do is to use linear arrays.  each warp does a column of 32.

	// parallelize along x.
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_B[MAX_THREADS+2];
	//	volatile T* s_marker = s_marker_B + 1;
	unsigned int s_change = 0;
	int tx1 = tx + 1;

	T s_new, s_old, s_prev;

	if ( bx + tx < sx ) { // make sure number of threads is a divisor of sx.
		s_prev = 0;

		for (int iy = 0; iy < sy; ++iy) {
			// copy part of marker and mask to shared memory
			if (tx == 0) {
				s_marker_B[0] = (bx == 0) ? 0 : marker[iy*sx + bx - 1];
				s_marker_B[MAX_THREADS + 1] = (bx + MAX_THREADS >= sx) ? 0 : marker[iy*sx + bx + MAX_THREADS];
			}
			if (tx < WARP_SIZE) {
				// first warp, get extra stuff
				s_marker_B[tx1] = marker[iy*sx + bx + tx];
			}
			if (tx < MAX_THREADS - WARP_SIZE) {
				s_marker_B[tx1 + WARP_SIZE] = marker[iy*sx + bx + tx + WARP_SIZE];
			}
			__syncthreads();

			// perform iteration
			s_old = s_marker_B[tx1];
			s_new = min( max( s_prev, s_old ),  mask[iy*sx + bx + tx]);
			s_change |= s_old ^ s_new;

			// output result back to global memory
			s_marker_B[tx1] = s_new;
			marker[iy*sx + bx + tx] = s_new;
			__syncthreads();

			s_prev = max( max(s_marker_B[tx1-1], s_marker_B[tx1]), s_marker_B[tx1+1]);
		}
	}
	if (s_change != 0) *change = true;
}

template <typename T>
__global__ void
iRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy, bool* __restrict__ change ){

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	unsigned int s_change=0;
	T s_old, s_new, s_prev;

	if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = sy - 1; iy >= 0; --iy) {

			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_change |= s_old ^ s_new;
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;
		}
	}
	if (s_change != 0) *change = true;
}

template <typename T>
__global__ void
iRec1DBackward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change ){

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_B[MAX_THREADS+2];
	unsigned int s_change = 0;
	int tx1 = tx + 1;  // for accessing s_marker_B

	T s_new, s_old, s_prev;

	if ( bx + tx < sx ) {  //make sure number of threads is a divisor of sx.

		s_prev = 0;

		for (int iy = sy - 1; iy >= 0; --iy) {

			if (tx == 0) {
				s_marker_B[0] = (bx == 0) ? 0 : marker[iy*sx + bx - 1];
				s_marker_B[MAX_THREADS+1] = (bx + MAX_THREADS >= sx) ? 0 : marker[iy*sx + bx + MAX_THREADS];
			}
			if (tx < WARP_SIZE) {
				// first warp, get extra stuff
				s_marker_B[tx1] = marker[iy*sx + bx + tx];
			}
			if (tx < MAX_THREADS - WARP_SIZE) {
				s_marker_B[tx1 + WARP_SIZE] = marker[iy*sx + bx + tx + WARP_SIZE];
			}
			__syncthreads();


			// perform iteration
			s_old = s_marker_B[tx1];
			s_new = min( max( s_prev, s_old ),  mask[iy*sx + bx + tx]);
			s_change |= s_old ^ s_new;

			// output result back to global memory
			s_marker_B[tx1] = s_new;
			marker[iy*sx + bx + tx] = s_new;
			__syncthreads();

			s_prev = max( max(s_marker_B[tx1-1], s_marker_B[tx1]), s_marker_B[tx1+1]);

		}
	}
	if (s_change != 0) *change = true;
}

// 在GPU里转换一个矩阵的整数值为字符值，并赋給另外一个矩阵
void pixelConvertIntToChar(GpuMat& input, GpuMat&result, Stream& stream){
	convertIntToChar(input.rows, input.cols, (int*)input.data, (unsigned char*)result.data,  StreamAccessor::getStream(stream));
}

// 在GPU里转换一个矩阵的整数值为字符值，并赋給另外一个矩阵
void convertIntToChar(int rows, int cols, int *input, unsigned char *result, cudaStream_t stream){
    dim3 threads(16, 16);
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    intToCharKernel<<<grid, threads, 0, stream>>>(rows, cols, input, result);
    cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}

// (kernel) 在GPU里转换一个矩阵的整数值为字符值，并赋給另外一个矩阵
__global__ void intToCharKernel(int rows, int cols, int *input, unsigned char *result){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * cols + x;

	if (y < rows && x < cols)
	{
		result[index] = (unsigned char)input[index];
	}
}

// 检查四个邻居 判断当前点是否会扩散
template <typename T>
__device__ bool checkCandidateNeighbor4(T *marker, const T *mask, int x, int y, int ncols, int nrows, T pval){
	bool isCandidate = false;
	int index = 0;

	T markerXYval;
	T maskXYval;
	if(x < (ncols-1)){
		// check right pixel
		index = y * ncols + (x+1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval, maskXYval)) ){
			isCandidate = true;
		}
	}

	if(y < (nrows-1)){
		// check pixel bellow current
		index = (y+1) * ncols + x;

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	// check left pixel
	if(x > 0){
		index = y * ncols + (x-1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	if(y > 0){
		// check up pixel
		index = (y-1) * ncols + x;

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}
	return isCandidate;
}

// 检查八个邻居 判断当前点是否会扩散
template <typename T>
__device__ bool checkCandidateNeighbor8(T *marker, const T *mask, int x, int y, int ncols, int nrows,T pval){
	int index = 0;
	bool isCandidate = checkCandidateNeighbor4(marker, mask, x, y, ncols, nrows, pval);

	T markerXYval;
	T maskXYval;

	// check up right corner
	if(x < (ncols-1) && y > 0){
		// check right pixel
		index = (y-1) * ncols + (x+1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval, maskXYval)) ){
			isCandidate = true;
		}
	}

	// check up left corner
	if(x> 0 && y > 0){
		// check pixel bellow current
		index = (y-1) * ncols + (x-1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	// check bottom left pixel
	if(x > 0 && y < (nrows-1)){
		index = (y+1) * ncols + (x-1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	// check bottom right
	if(x < (ncols-1) && y < (nrows-1)){
		index = (y+1) * ncols + (x+1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}
	return isCandidate;
}

// 返回两个值的最小值
__device__ unsigned char min(unsigned char pval, unsigned char imageXYval){
	if(pval>= imageXYval) {
		return imageXYval;
	}
	return pval;
}

// 返回两个值的最大值
__device__ unsigned char max(unsigned char pval, unsigned char imageXYval){
	if(pval>= imageXYval) {
		return pval;
	}
	return imageXYval;
}

// 扩散处理函数 				
__device__ int propagate(int *seeds, unsigned char *image, int x, int y, int ncols, unsigned char pval,unsigned char * newVal){
	
	int returnValue;
	int index = y*ncols + x;
	unsigned char seedXYval = seeds[index];
	unsigned char imageXYval = image[index];
	returnValue=-1;
	
	if((seedXYval < pval) && (imageXYval != seedXYval)){
		unsigned char newValue = min(pval, imageXYval);
		//  this should be a max atomic...
		atomicMax(&(seeds[index]), newValue);
		returnValue = index;
		(*newVal)=seeds[index];
	}
	return returnValue;
}


// 建立并行堆
template <typename T>
PPQ_TYPE * initHeap(T *marker, const T *mask, int sx, int sy, bool conn8,cudaStream_t stream){

	int i;
	//PPQ_TYPE *ppqs;
	cudaError_t err;
	// 为多个并发堆分配空间
	PPQ_TYPE *ppqs=(PPQ_TYPE *)malloc(NUM_OF_BLOCKS*sizeof(PPQ_TYPE));
	for(i=0;i<NUM_OF_BLOCKS;i++){
		err=cudaMalloc((void**)&(ppqs[i].parHeap_index),sizeof(int)*sx*sy/NUM_OF_BLOCKS*2);	
		err=cudaMalloc((void**)&(ppqs[i].parHeap_key),sizeof(unsigned char)*sx*sy/NUM_OF_BLOCKS*2);	
		err=cudaMalloc((void**)&(ppqs[i].init_index),sizeof(int)*NUM_THREADS);	
		err=cudaMalloc((void**)&(ppqs[i].init_key),sizeof(unsigned char)*NUM_THREADS);
		// 为每个并发堆的缓冲区分配空间
		if(conn8==true){
			err=(cudaMalloc( (void**) &(ppqs[i].new_items_index), sizeof(int)*NUM_THREADS*(8+1)*2));
			err=(cudaMalloc( (void**) &(ppqs[i].new_items_key), sizeof(unsigned char)*NUM_THREADS*(8+1)*2));
		}
		else{
			err=(cudaMalloc( (void**) &(ppqs[i].new_items_index), sizeof(int)*NUM_THREADS*(4+1)*2));
			err=(cudaMalloc( (void**) &(ppqs[i].new_items_key), sizeof(unsigned char)*NUM_THREADS*(4+1)*2));
		}
		if( err != cudaSuccess){
    		printf("CUDA error allocate memory for new items: %s\n", cudaGetErrorString(err));
     		exit(-1);
		}
	}
	
	// alloc it with the same size as the input image.
	PPQ_ITEM *d_queue = NULL;
	cudaMalloc( (void**) &d_queue, sizeof(PPQ_ITEM) * sx * sy ) ;

	int *d_queue_size;
	cudaMalloc( (void**) &d_queue_size, sizeof(int)) ;
	cudaMemset( (void*) d_queue_size, 0, sizeof(int)) ;
	
	dim3 threads(16, 16);
	dim3 grid((sx + threads.x - 1) / threads.x, (sy + threads.y - 1) / threads.y);
	// 扫描图像 找到潜在扩散的点 存在d_queue里。
	initQueuePixels<T><<< grid, threads, 0, stream >>>(marker, mask, sx, sy, conn8, d_queue, d_queue_size);
	
	//dim3 threads_initPPQ(1, 1);
	//dim3 grid_initPPQ(1,1);
	//initPPQ<<< grid_initPPQ, threads_initPPQ, 0, stream>>>(marker, mask, sx, sy, conn8, d_queue, d_queue_size);

	if (stream == 0) {
		cudaDeviceSynchronize();
	}
	else {
		cudaStreamSynchronize(stream);
	}
	
	// 为了节省空间，重新分配存储空间， 并把并行堆节点数组复制到新的存储空间里
	int *d_queue_size_cpu=(int *)malloc(sizeof(int));
    err=cudaMemcpy( d_queue_size_cpu, d_queue_size, sizeof(int), cudaMemcpyDeviceToHost) ;
    if( err != cudaSuccess){
    	printf("CUDA error t1: %s\n", cudaGetErrorString(err));
     	exit(1);
     }
	//cout<<"Queue size is "<<(*d_queue_size_cpu)<<endl;
	
	int h_compact_queue_size;
	cudaMemcpy( &h_compact_queue_size, d_queue_size, sizeof(int), cudaMemcpyDeviceToHost);

	//cout<<"h_compact_queue_size is "<<h_compact_queue_size<<endl;

	PPQ_ITEM *d_queue_fit = NULL;
	// alloc current size +1000 (magic number)
	cudaMalloc( (void**) &d_queue_fit, sizeof(PPQ_ITEM) * (h_compact_queue_size+1000)*2 ) ;

	// Copy content of the d_queue (which has the size of the image x*y) to a more compact for (d_queue_fit). 
	// This should save a lot of memory, since the compact queue is usually much smaller than the image size
	cudaMemcpy( d_queue_fit, d_queue, sizeof(PPQ_ITEM) * h_compact_queue_size, cudaMemcpyDeviceToDevice ) ;

	// This is the int containing the size of the queue
	cudaFree(d_queue_size) ;
	// Cleanup the temporary memory use to store the queue
	cudaFree(d_queue) ;
		
	// Sort d_queue by the acsending order.
	// 排序
	sortGM_struct(d_queue_fit,h_compact_queue_size,1024);

	int *d_queue_fit_index = NULL;
	cudaMalloc( (void**) &d_queue_fit_index, sizeof(int) * (h_compact_queue_size+1000)*2 ) ;
	unsigned char *d_queue_fit_key = NULL;
	cudaMalloc( (void**) &d_queue_fit_key, sizeof(unsigned char) * (h_compact_queue_size+1000)*2 ) ;
	strucTransfer<<<((h_compact_queue_size-1)/256+1),256>>>(d_queue_fit,d_queue_fit_index,d_queue_fit_key,h_compact_queue_size);

	
	for(i=0;i<NUM_OF_BLOCKS;i++){
		ppqs[i].init_index = d_queue_fit_index+h_compact_queue_size - NUM_OF_BLOCKS*NUM_THREADS + i*NUM_THREADS;
		ppqs[i].init_key = d_queue_fit_key+h_compact_queue_size - NUM_OF_BLOCKS*NUM_THREADS + i*NUM_THREADS;
	}

	//d_queue_fit+=NUM_THREADS*NUM_OF_BLOCKS;
	//h_compact_queue_size-=	NUM_THREADS*NUM_OF_BLOCKS;
	//cout<<"h_compact_queue_size is (later)"<<h_compact_queue_size<<endl;
	//printf("h_compact_queue_size=%d\n",h_compact_queue_size);
	dim3 threads1(32,1);
	dim3 grid1(NUM_OF_BLOCKS,1);

	PPQ_TYPE *ppqs_gpu;
	err=cudaMalloc((void**)&ppqs_gpu,sizeof(PPQ_TYPE)*NUM_OF_BLOCKS);	
	if( err != cudaSuccess){
		printf("CUDA allocation error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaMemcpy( ppqs_gpu, ppqs, sizeof(PPQ_TYPE) * NUM_OF_BLOCKS, cudaMemcpyHostToDevice ) ;

	distInitialItems<<<grid1, threads1, 0, stream >>>(ppqs_gpu, d_queue_fit_index, d_queue_fit_key ,h_compact_queue_size);

	return ppqs_gpu;

}

__global__ void strucTransfer(PPQ_ITEM * d_queue_fit,int * d_queue_fit_index,unsigned char * d_queue_fit_key,int d_queue_size){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int id=bx*blockDim.x+tx;
	if(id<d_queue_size){
		d_queue_fit_index[id]=d_queue_fit[id].index;
		d_queue_fit_key[id]=d_queue_fit[id].key;
	}
}


/*
把潜在扩散的像素点聚到一个缓冲区里
d_queue 并发堆节点数组
d_queue_size 节点个数
*/
template <typename T>
__global__ void initQueuePixels(T *marker, const T *mask, int sx, int sy, bool conn8, PPQ_ITEM *d_queue, int *d_queue_size){
 	int x = blockIdx.x * blockDim.x + threadIdx.x;
 	int y = blockIdx.y * blockDim.y + threadIdx.y;
 
 	// if it is inside image without right/bottom borders
 	if(y < (sy) && x < (sx)){
 		int input_index = y * sy + x;
 		T pval = marker[input_index];
 		bool isCandidate = false;
 		if(conn8){
 			 //connectivity 8
 			isCandidate = checkCandidateNeighbor8(marker, mask, x, y, sx, sy, pval);
 		}else{
 			 //connectivity 4
 			isCandidate = checkCandidateNeighbor4(marker, mask, x, y, sx, sy, pval);
 		}
 		if(isCandidate){
 			int queuePos = atomicAdd((unsigned int*)d_queue_size, 1);
 			d_queue[queuePos].index = input_index;
 			d_queue[queuePos].key = pval;
 		}	
 	}
}

/*
把每个并发堆里初始化区的并发堆节点拷贝到并发堆中
*/
__global__ void distInitialItems(PPQ_TYPE * ppqs, int * d_queue_fit_index,unsigned char * d_queue_fit_key, int d_queue_size){
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int id=bx*blockDim.x+tx;
	int i=0;
	int a,b,c,d;
	int counter=tx;
	for(i=id;i<d_queue_size;i+=(NUM_OF_BLOCKS*blockDim.x)){
		//ppqs[bx].parHeap[counter]=d_queue[d_queue_size-1-i];
		ppqs[bx].parHeap_index[counter]=d_queue_fit_index[d_queue_size-1-i];
		ppqs[bx].parHeap_key[counter]=d_queue_fit_key[d_queue_size-1-i];
		counter+=blockDim.x;
	}
	if(tx==0){
		a=(d_queue_size/(blockDim.x*NUM_OF_BLOCKS))*blockDim.x;
		b=d_queue_size%(blockDim.x*NUM_OF_BLOCKS);
		c=b/blockDim.x;
		d=b%blockDim.x;
		//printf("a=%d,b=%d,c=%d,d=%d\n",a,b,c,d); 
		if((bx+1)<=c){
			a+=blockDim.x;
		}
		else if(c==bx){
			a+=d;
		}
		ppqs[bx].numOfItems=a;
		//printf("block %d has %d items.\n",bx,ppqs[bx].numOfItems);
		//printf("%d\n",ppqs[bx].numOfItems);
		if(bx==0){
		for(i=0;i<(d_queue_size+NUM_THREADS*NUM_OF_BLOCKS);i++){
			//printf("[%d,%d] ",d_queue[i].index,d_queue[i].key);		
		}
		}
	}
}

// 形态学重建
__global__ void reconstruction(PPQ_TYPE * ppqs, int * seeds, unsigned char * image, int ncols, int nrows){
	
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	//ITEM_TYPE output[4];
	int output_index[4];
	unsigned char output_key[4];
	
	 
	int x,y;
	//ITEM_TYPE workUnit;
	int workUnit_index;
	unsigned char workUnit_key;
	
	//unsigned char pval = 0;
	//ITEM_TYPE retWork; 
	int retWork_index;
	unsigned char retWork_key;
		
	__shared__ int writeAddr[NUM_THREADS];
	__shared__ int writeAddr_1[NUM_THREADS];
	__shared__ int exclusiveScan[NUM_THREADS];

	int total=0;
	int numOfWorkUnit=NUM_THREADS;
	int i;
	int numOfItems;
	//__shared__ ITEM_TYPE buffer[NUM_ITEMS_2];
	//__shared__ ITEM_TYPE buffer[6*NUM_THREADS];
	__shared__ int buffer_index[6*NUM_THREADS];
	__shared__ unsigned char buffer_key[6*NUM_THREADS];
	//__shared__ ITEM_TYPE *p1;
	//__shared__ ITEM_TYPE *p2;
	//__shared__ ITEM_TYPE *p3;
	//__shared__ ITEM_TYPE *p4;
	//__shared__ ITEM_TYPE *p5;
	//__shared__ ITEM_TYPE *p6;
	
	__shared__ int *p1_index;
	__shared__ unsigned char *p1_key;
	__shared__ int *p2_index;
	__shared__ unsigned char *p2_key;
	__shared__ int *p3_index;
	__shared__ unsigned char *p3_key;
	__shared__ int *p4_index;
	__shared__ unsigned char *p4_key;
	__shared__ int *p5_index;
	__shared__ unsigned char *p5_key;
	__shared__ int *p6_index;
	__shared__ unsigned char *p6_key;
	
 	__shared__	int * count;
	//__shared__ ITEM_TYPE * new_items;
	__shared__ int * new_items_index;
	__shared__ unsigned char * new_items_key;

	if(tx==0){
		p1_index=buffer_index;	
		p2_index=p1_index+NUM_THREADS;
		p3_index=p2_index+NUM_THREADS;
		p4_index=p3_index+NUM_THREADS;
		p5_index=p4_index+NUM_THREADS; 
		p6_index=p5_index+NUM_THREADS;
		
		p1_key=buffer_key;	
		p2_key=p1_key+NUM_THREADS;
		p3_key=p2_key+NUM_THREADS;
		p4_key=p3_key+NUM_THREADS;
		p5_key=p4_key+NUM_THREADS; 
		p6_key=p5_key+NUM_THREADS;
		count=p5_index;
		new_items_index=ppqs[bx].new_items_index;
		new_items_key=ppqs[bx].new_items_key;
	}
	numOfItems=ppqs[bx].numOfItems;
	workUnit_index=ppqs[bx].init_index[tx];
	workUnit_key=ppqs[bx].init_key[tx];
	#ifdef WITH_SYNC
	__syncthreads();
	#endif
	#ifdef DEBUG_CONSTR
	int iter=0;
	//if(tx==0){
	//	printf("num of items [%d] is %d.\n", bx, numOfItems);
	//}
	#endif	
	while((numOfItems+numOfWorkUnit)>0){

		#ifdef DEBUG_CONSTR
		iter++;
		//if(tx==0){
		//	printf("*******************%d iteration************************\n",iter);
		//}
		#endif	
		/* Propagation */
		writeAddr[tx]=0;
		if(tx<numOfWorkUnit){
			y = (workUnit_index)/ncols;
			x = (workUnit_index)%ncols;		
			if(workUnit_index >= 0){			
				if(workUnit_index >= 0 && y > 0){
					retWork_index = propagate((int*)seeds, image, x, y-1, ncols, workUnit_key,&retWork_key);
					if(retWork_index >= 0){
						output_index[writeAddr[tx]] = retWork_index;
						output_key[writeAddr[tx]] = retWork_key;
						writeAddr[tx]++;
					}
				}		
				if(workUnit_index >= 0 && y < nrows-1){
					retWork_index = propagate((int*)seeds, image, x, y+1, ncols, workUnit_key,&retWork_key);
					if(retWork_index >= 0){
						output_index[writeAddr[tx]] = retWork_index;
						output_key[writeAddr[tx]] = retWork_key;
						writeAddr[tx]++;
					}	
				}
				if(workUnit_index >= 0 && x > 0){
					retWork_index = propagate((int*)seeds, image, x-1, y, ncols, workUnit_key,&retWork_key);
					if(retWork_index >= 0){
						output_index[writeAddr[tx]] = retWork_index;
						output_key[writeAddr[tx]] = retWork_key;
						writeAddr[tx]++;
					}
				}
				if(workUnit_index >= 0 && x < ncols-1){
					retWork_index = propagate((int*)seeds, image, x+1, y, ncols, workUnit_key,&retWork_key);
					if(retWork_index >= 0){
						output_index[writeAddr[tx]] = retWork_index;
						output_key[writeAddr[tx]] = retWork_key;
						writeAddr[tx]++;
					}
				}
			}
		}
		// We switch to inclusive scan.
		writeAddr_1[tx]=writeAddr[tx];
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		// run a prefix-sum on threads inserting data to the queue
		scan(writeAddr, exclusiveScan);
		//total=exclusiveScan[tx]+writeAddr[tx];
		total=exclusiveScan[NUM_THREADS-1];

		for(i=0;i<writeAddr_1[tx];i++){
			new_items_index[exclusiveScan[tx]-writeAddr_1[tx]+i]=output_index[i];
			new_items_key[exclusiveScan[tx]-writeAddr_1[tx]+i]=output_key[i];
			//buffer[exclusiveScan[tx]-writeAddr_1[tx]+i]=output[i];
		}	
	
		/* Compensate() */
		if((total+numOfItems)<(NUM_THREADS_2)){			
			for(i=tx;i<numOfItems;i+=blockDim.x){
				//buffer[total+i]=ppqs[bx].parHeap[i];
				new_items_index[total+i]=ppqs[bx].parHeap_index[i];
				new_items_key[total+i]=ppqs[bx].parHeap_key[i];
			}
			total+=numOfItems;
			numOfItems=total>NUM_THREADS?(total-NUM_THREADS):0;
		}
		else{
			if(total>=NUM_THREADS){
				if(tx<numOfItems){
					//buffer[total+tx]=ppqs[bx].parHeap[tx];
					new_items_index[total+tx]=ppqs[bx].parHeap_index[tx];
					new_items_key[total+tx]=ppqs[bx].parHeap_key[tx];
				}
				if(numOfItems>=NUM_THREADS){
					//numOfItems+=(total- NUM_THREADS);
					total+=NUM_THREADS;
					//numOfItems-=NUM_THREADS;
				}
				else{
					total+=numOfItems;	
					//numOfItems=(total- NUM_THREADS);
				}
			}
			else{
				//buffer[total+tx]=ppqs[bx].parHeap[tx];
				//if(tx<(NUM_THREADS-total)){
				//	buffer[total+NUM_THREADS+tx]=ppqs[bx].parHeap[numOfItems-1-tx];
				//}
				if(tx<(NUM_THREADS-total)){
					new_items_index[total+tx]=ppqs[bx].parHeap_index[numOfItems-1-tx];
					new_items_key[total+tx]=ppqs[bx].parHeap_key[numOfItems-1-tx];
				}
				new_items_index[NUM_THREADS+tx]=ppqs[bx].parHeap_index[tx];
				new_items_key[NUM_THREADS+tx]=ppqs[bx].parHeap_key[tx];
				numOfItems-=(NUM_THREADS-total);
				total=NUM_THREADS_2;
			}
		}

		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		
		/* Sort and make it as an ascending sequence */
		//if(total>0){
			//sort(buffer,total);
			//sort(new_items,total);
		//}
		
		if(total>(NUM_THREADS_2)){
			//workUnit=buffer[total-1-tx];
			workUnit_index=new_items_index[total-1-tx];
			workUnit_key=new_items_key[total-1-tx];
			//p3[tx]=new_items[total-NUM_THREADS-1-tx];
			//p3[tx]=buffer[total-NUM_THREADS-1-tx];
			//p1[tx]=buffer[total-NUM_THREADS-1-tx];
			p1_index[tx]=new_items_index[total-NUM_THREADS-1-tx];
			p1_key[tx]=new_items_key[total-NUM_THREADS-1-tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			sort_one(p1_index,p1_key,NUM_THREADS,count);
			//sort(p1,NUM_THREADS);			
			p3_index[tx]=p1_index[NUM_THREADS-1-tx];
			p3_key[tx]=p1_key[NUM_THREADS-1-tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
		
			/* delete update */
			deleteUpdate(ppqs,numOfItems,p1_index,p2_index,p3_index,p4_index,p5_index,p6_index,p1_key,p2_key,p3_key,p4_key,p5_key,p6_key);	
		
			/* insert update */
			numOfItems=numOfItems>NUM_THREADS?numOfItems:NUM_THREADS;
			insertUpdate(ppqs,(total-NUM_THREADS_2),numOfItems,new_items_index,new_items_key,p1_index,p2_index,p3_index,p4_index,p5_index,p1_key,p2_key,p3_key,p4_key,p5_key,count);
			
			numOfItems+=(total-NUM_THREADS_2);
			numOfWorkUnit=NUM_THREADS;
		}
		else if(total==(NUM_THREADS_2)){
			//workUnit=buffer[NUM_THREADS+tx];
			workUnit_index=new_items_index[NUM_THREADS+tx];
			workUnit_key=new_items_key[NUM_THREADS+tx];
			//p1[tx]=buffer[NUM_THREADS-1-tx];			
			//p3[tx]=buffer[NUM_THREADS-1-tx];
			//p3[tx]=new_items[NUM_THREADS-1-tx];
			p1_index[tx]=new_items_index[NUM_THREADS-1-tx];
			p1_key[tx]=new_items_key[NUM_THREADS-1-tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			sort_one(p1_index,p1_key,NUM_THREADS,count);
			//sort(p1,NUM_THREADS);			
			p3_index[tx]=p1_index[NUM_THREADS-1-tx];
			p3_key[tx]=p1_key[NUM_THREADS-1-tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			
			/* delete update */
			deleteUpdate(ppqs,numOfItems,p1_index,p2_index,p3_index,p4_index,p5_index,p6_index,p1_key,p2_key,p3_key,p4_key,p5_key,p6_key);	
			numOfWorkUnit=NUM_THREADS;		
		}
		else if(total>=NUM_THREADS){
			//workUnit=buffer[total-1-tx];
			workUnit_index=new_items_index[total-1-tx];
			workUnit_key=new_items_key[total-1-tx];
			if(tx<(total-NUM_THREADS)){
				//ppqs[bx].parHeap[tx]=new_items[total-1-NUM_THREADS-tx];	
				//ppqs[bx].parHeap[tx]=buffer[total-1-NUM_THREADS-tx];	
				//p1[tx]=buffer[total-1-NUM_THREADS-tx];	
				p1_index[tx]=new_items_index[total-1-NUM_THREADS-tx];	
				p1_key[tx]=new_items_key[total-1-NUM_THREADS-tx];	
			}
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			sort_one(p1_index,p1_key,(total-NUM_THREADS),count);
			//sort(p1,(total-NUM_THREADS));			
			if(tx<(total-NUM_THREADS)){
				//ppqs[bx].parHeap[tx]=buffer[total-1-NUM_THREADS-tx];	
				ppqs[bx].parHeap_index[tx]=p1_index[total-1-NUM_THREADS-tx];	
				ppqs[bx].parHeap_key[tx]=p1_key[total-1-NUM_THREADS-tx];	
			}
			numOfWorkUnit=NUM_THREADS;	
		}	
		else if(tx<total){
			//workUnit=buffer[tx];
			workUnit_index=new_items_index[tx];			
			workUnit_key=new_items_key[tx];			
			numOfWorkUnit=total;
		}
		if(total<NUM_THREADS){
			numOfWorkUnit=total;
		}
		else{
			numOfWorkUnit=NUM_THREADS;
		}
		#ifdef WITH_SYNC
		__syncthreads();
		#endif	
	}
	#ifdef DEBUG_CONSTR
	//if(tx==0){
	//	printf("iter=%d\n",iter);
	//}
	#endif	
}

// 假设要插入的元素都在buffer里.
// 变量leftover是要插入元素的数目.
// 变量numOfItems是优先队列里元素的总数.
// 每个节点里元素的数目等于线程块里线程的数目.
__device__ void insertUpdate(PPQ_TYPE * ppqs,int leftover,int numOfItems,int * buffer_index,unsigned char * buffer_key,int *p1_index,int *p2_index,int *p3_index,int *p4_index,int *p5_index,unsigned char *p1_key,unsigned char *p2_key,unsigned char *p3_key,unsigned char *p4_key,unsigned char *p5_key,int * count){
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int numOfNodes=numOfItems / NUM_THREADS;
	int offset=numOfItems % NUM_THREADS;
	int num;
	int parent,current;
	int next=numOfNodes+1;
	#ifdef DEBUG_INSERT
	int i;
	if(tx==0){
		printf("************Insert Update******************\n");
		printf("leftover=%d\n",leftover);
		for(i=0;i<leftover;i++){
			printf("[%d,%d] ",buffer_index[i],buffer_key[i]);
		}
		printf("\n");
	}
	#endif	
	//handle the half-full node first.
	if(offset!=0){
		//num=NUM_THREADS-offset;
		if(leftover>=(NUM_THREADS-offset)){
			num=NUM_THREADS-offset;
		}
		else{
			num=leftover;
		}
		//if(tx<num){
		//	p1[tx]=buffer[leftover-1-tx];
		//}
		/* newly added start */
		if(tx<num){
			p3_index[tx]=buffer_index[leftover-1-tx];
			p3_key[tx]=buffer_key[leftover-1-tx];
		}
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		sort_one(p3_index,p3_key,num,count);		
		//sort(p3,num);				
		if(tx<num){
			p1_index[tx]=p3_index[num-1-tx];	
			p1_key[tx]=p3_key[num-1-tx];	
		}
		/* newly added end */
		leftover-=num;
		if(tx<offset){
			p2_index[tx]=ppqs[bx].parHeap_index[numOfItems-offset+tx];
			p2_key[tx]=ppqs[bx].parHeap_key[numOfItems-offset+tx];
		}
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		#ifdef DEBUG_INSERT
		if(tx==0){
			printf("Handling Leftover %d........\n",leftover);
			printf("P1........\n");
			for(i=0;i<num;i++){
				printf("[%d,%d] ",p1_index[i],p1_key[i]);
			}
			printf("\n");
			printf("P2........\n");
			for(i=0;i<offset;i++){
				printf("[%d,%d] ",p2_index[i],p2_key[i]);
			}
			printf("\n");
		}
		#endif	
		
		merge(p1_index,p1_key,num,p2_index,p2_key,offset,p4_index,p4_key);
		#ifdef DEBUG_INSERT
		if(tx==0){
			printf("Merging P4........\n");
			if(tx==0){
				for(i=0;i<(num+offset);i++){
					printf("[%d,%d] ",p4_index[i],p4_key[i]);
				}
				printf("\n");
			}
		}
		#endif	
		current=next;
		parent=current/2;
		if(parent>0){
			num=num+offset;
			p5_index[tx]=ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx];
			p5_key[tx]=ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			#ifdef DEBUG_INSERT
			if(tx==0){
				printf("P4........\n");
				for(i=0;i<num;i++){
					printf("[%d,%d] ",p4_index[i],p4_key[i]);
				}
				printf("\n");
				printf("P5........\n");
				for(i=0;i<NUM_THREADS;i++){
					printf("[%d,%d] ",p5_index[i],p5_key[i]);
				}
				printf("\n");
			}
			#endif	
			merge(p4_index,p4_key,num,p5_index,p5_key,NUM_THREADS,p1_index,p1_key);
			#ifdef DEBUG_INSERT		
			if(tx==0){
				printf("Merging P1........\n");
				for(i=0;i<(NUM_THREADS+num);i++){
					printf("[%d,%d] ",p1_index[i],p1_key[i]);
				}
				printf("\n");
			}
			#endif	
			if(tx<num){
				ppqs[bx].parHeap_index[(current-1)*NUM_THREADS+tx]=p2_index[tx];
				ppqs[bx].parHeap_key[(current-1)*NUM_THREADS+tx]=p2_key[tx];
			}
			current=parent;
			parent=current/2;
			p4_index[tx]=p1_index[tx];	
			p4_key[tx]=p1_key[tx];	
		}
		while(parent>0){
			p5_index[tx]=ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx];
			p5_key[tx]=ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			#ifdef DEBUG_INSERT
			if(tx==0){
				printf("P4........\n");
				for(i=0;i<NUM_THREADS;i++){
					printf("[%d,%d] ",p4_index[i],p4_key[i]);
				}
				printf("\n");
				printf("P5........\n");
				for(i=0;i<NUM_THREADS;i++){
					printf("[%d,%d] ",p5_index[i],p5_key[i]);
				}
				printf("\n");
			}
			#endif	
			merge(p4_index,p4_key,NUM_THREADS,p5_index,p5_key,NUM_THREADS,p1_index,p1_key);
			#ifdef DEBUG_INSERT		
			if(tx==0){
				printf("Merging........\n");
				for(i=0;i<(NUM_THREADS+NUM_THREADS);i++){
					printf("[%d,%d] ",p1_index[i],p1_key[i]);
				}
				printf("\n");
			}
			#endif		
			ppqs[bx].parHeap_index[(current-1)*NUM_THREADS+tx]=p2_index[tx];
			ppqs[bx].parHeap_key[(current-1)*NUM_THREADS+tx]=p2_key[tx];
			p4_index[tx]=p1_index[tx];
			p4_key[tx]=p1_key[tx];
			current=parent;
			parent=current/2;
		}
		ppqs[bx].parHeap_index[tx]=p1_index[tx];
		ppqs[bx].parHeap_key[tx]=p1_key[tx];
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
			if(tx<num){
				p3_index[tx]=buffer_index[leftover-1-tx];
				p3_key[tx]=buffer_key[leftover-1-tx];
			}
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			sort_one(p3_index,p3_key,num,count);		
			//sort(p3,num);		
			if(tx<num){
				p1_index[tx]=p3_index[num-1-tx];	
				p1_key[tx]=p3_key[num-1-tx];	
			}
			/* newly added end */
			p2_index[tx]=ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx];
			p2_key[tx]=ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			#ifdef DEBUG_INSERT
			if(tx==0){
				printf("P1........\n");
				for(i=0;i<num;i++){
					printf("[%d,%d] ",p1_index[i],p1_key[i]);
				}
				printf("\n");
				printf("P2........\n");
				for(i=0;i<NUM_THREADS;i++){
					printf("[%d,%d] ",p2_index[i],p2_key[i]);
				}
				printf("\n");
			}
			#endif		
			merge(p1_index,p1_key,num,p2_index,p2_key,NUM_THREADS,p4_index,p4_key);	
			#ifdef DEBUG_INSERT		
			if(tx==0){
				printf("Merging P4........\n");
				for(i=0;i<(NUM_THREADS+num);i++){
					printf("[%d,%d] ",p4_index[i],p4_key[i]);
				}
				printf("\n");
			}
			#endif	
			if(tx<num){
				ppqs[bx].parHeap_index[(current-1)*NUM_THREADS+tx]=p5_index[tx];
				ppqs[bx].parHeap_key[(current-1)*NUM_THREADS+tx]=p5_key[tx];
			}
			current=parent;
			parent=current/2;
			p1_index[tx]=p4_index[tx];	
			p1_key[tx]=p4_key[tx];	
		}
		while(parent>0){
			p2_index[tx]=ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx];
			p2_key[tx]=ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx];
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			#ifdef DEBUG_INSERT
			if(tx==0){
				printf("P1........\n");
				for(i=0;i<NUM_THREADS;i++){
					printf("[%d,%d] ",p1_index[i],p1_key[i]);
				}
				printf("\n");
				printf("P2........\n");
				for(i=0;i<NUM_THREADS;i++){
					printf("[%d,%d] ",p2_index[i],p2_key[i]);
				}
				printf("\n");
			}
			#endif	
			merge(p1_index,p1_key,NUM_THREADS,p2_index,p2_key,NUM_THREADS,p4_index,p4_key);
			#ifdef DEBUG_INSERT		
			if(tx==0){
				printf("Merging P4........\n");
				for(i=0;i<(NUM_THREADS+NUM_THREADS);i++){
					printf("[%d,%d] ",p4_index[i],p4_key[i]);
				}
				printf("\n");
			}
			#endif	
			ppqs[bx].parHeap_index[(current-1)*NUM_THREADS+tx]=p5_index[tx];
			ppqs[bx].parHeap_key[(current-1)*NUM_THREADS+tx]=p5_key[tx];
			current=parent;
			parent=current/2;
			p1_index[tx]=p4_index[tx];	
			p1_key[tx]=p4_key[tx];	
		}
		ppqs[bx].parHeap_index[tx]=p1_index[tx];
		ppqs[bx].parHeap_key[tx]=p1_key[tx];
		next++;
		leftover-=num;	
		num=leftover>NUM_THREADS?NUM_THREADS:leftover;
	}
	#ifdef WITH_SYNC
	__syncthreads();
	#endif
}

// 更新节点 线程的数目等于节点元素的数目
// Assuming r elements are stored in *p3
__device__ void deleteUpdate(PPQ_TYPE * ppqs,int numOfItems,int *p1_index,int *p2_index,int *p3_index,int *p4_index,int *p5_index,int *p6_index,unsigned char *p1_key,unsigned char *p2_key,unsigned char *p3_key,unsigned char *p4_key,unsigned char *p5_key,unsigned char *p6_key){
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int numOfNodes=numOfItems / NUM_THREADS;
	int offset=numOfItems % NUM_THREADS;
	int parent=1;
	int left=2;
	int right=3;
	int next;
	int keep;
	#ifdef DEBUG_DELETE
	if(tx==0){
		printf("************Delete Update******************\n");
	}
	#endif
	#ifdef DEBUG_DELETE
	int i;
	if(tx==0){
		for(i=0;i<NUM_THREADS;i++){
			printf("[%d,%d] ",p3_index[i],p3_key[i]);
		}
		printf("\n");
		printf("Start........\n");
	}
	#endif
	// merge with full node
	if((numOfNodes==0)){
		ppqs[bx].parHeap_index[tx]=p3_index[tx];
		ppqs[bx].parHeap_key[tx]=p3_key[tx];
		return;
	}
	if((numOfNodes==1)&&(offset==0)){
		ppqs[bx].parHeap_index[tx]=p3_index[tx];
		ppqs[bx].parHeap_key[tx]=p3_key[tx];
		return;
	}
	while(right<=numOfNodes){
		if(ppqs[bx].parHeap_key[left*NUM_THREADS-1]>ppqs[bx].parHeap_key[right*NUM_THREADS-1]){
			next=left;
			keep=right;
		}
		else{
			next=right;
			keep=left;
		}
		p2_index[tx]=ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+tx];
		p2_key[tx]=ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+tx];
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("P3........\n");
			for(i=0;i<NUM_THREADS;i++){
				printf("[%d,%d] ",p3_index[i],p3_key[i]);
			}
			printf("\n");
			printf("P2........\n");
			for(i=0;i<NUM_THREADS;i++){
				printf("[%d,%d] ",p2_index[i],p2_key[i]);
			}
			printf("\n");
		}
		#endif
		merge(p2_index,p2_key, NUM_THREADS,p3_index,p3_key, NUM_THREADS,p5_index,p5_key);
		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("Merging P5........\n");
			for(i=0;i<NUM_THREADS+NUM_THREADS;i++){
				printf("[%d,%d] ",p5_index[i],p5_key[i]);
			}
			printf("\n");
		}
		#endif
		p4_index[tx]=ppqs[bx].parHeap_index[(right-1)*NUM_THREADS+tx];
		p4_key[tx]=ppqs[bx].parHeap_key[(right-1)*NUM_THREADS+tx];
		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("P4........\n");
			for(i=0;i<NUM_THREADS;i++){
				printf("[%d,%d] ",p4_index[i],p4_key[i]);
			}
			printf("\n");
		}
		#endif
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		merge(p4_index,p4_key, NUM_THREADS,p5_index,p5_key, NUM_THREADS_2,p1_index,p1_key);

		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("Merging P1........\n");
			for(i=0;i<NUM_THREADS+NUM_THREADS+NUM_THREADS;i++){
				printf("[%d,%d] ",p1_index[i],p1_key[i]);
			}
			printf("\n");
			printf("One Round parent=%d,left=%d,right=%d\n",parent,left,right);
		}
		#endif
		ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx]=p1_index[tx];
		ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx]=p1_key[tx];
		ppqs[bx].parHeap_index[(keep-1)*NUM_THREADS+tx]=p2_index[tx];
		ppqs[bx].parHeap_key[(keep-1)*NUM_THREADS+tx]=p2_key[tx];
		parent=next;
		left=parent*2;
		right=parent*2+1;
	}
	//might have to merge with half-full node
	if(((numOfNodes+1)==left)&&(offset!=0)){
		if(tx<offset){
			p2_index[tx]=ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+tx];
			p2_key[tx]=ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+tx];
		}
		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("Left Node........\n");
			for(i=0;i<offset;i++){
				printf("[%d,%d] ",ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+i],ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+i]);
			}
			printf("\n");
			printf("P3........\n");
			for(i=0;i<NUM_THREADS;i++){
				printf("[%d,%d] ",p3_index[i],p3_key[i]);
			}
			printf("\n");
		}
		#endif	
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		merge(p2_index,p2_key,offset,p3_index,p3_key, NUM_THREADS,p5_index,p5_key);
		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("Merging P5........\n");
			for(i=0;i<(offset+NUM_THREADS);i++){
				printf("[%d,%d] ",p5_index[i],p5_key[i]);
			}
			printf("\n");
		}
		#endif	
		ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx]=p5_index[tx];
		ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx]=p5_key[tx];
		if(tx<offset){
			ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+tx]=p6_index[tx];
			ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+tx]=p6_key[tx];
		}
	}
	else if((numOfNodes+1)==right){
		p2_index[tx]=ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+tx];
		p2_key[tx]=ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+tx];
		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("Left Node........\n");
			for(i=0;i<NUM_THREADS;i++){
				printf("[%d,%d] ",ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+i],ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+i]);
			}
			printf("\n");
			printf("P3........\n");
			for(i=0;i<NUM_THREADS;i++){
				printf("[%d,%d] ",p3_index[i],p3_key[i]);
			}
			printf("\n");
		}
		#endif	
		#ifdef WITH_SYNC
		__syncthreads();
		#endif
		merge(p2_index,p2_key,NUM_THREADS,p3_index,p3_key, NUM_THREADS,p5_index,p5_key);
		#ifdef DEBUG_DELETE
		if(tx==0){
			printf("Merging P5........\n");
			for(i=0;i<(NUM_THREADS+NUM_THREADS);i++){
				printf("[%d,%d] ",p5_index[i],p5_key[i]);
			}
			printf("\n");
		}
		#endif	
		if(offset!=0){
			if(tx<offset){
				p4_index[tx]=ppqs[bx].parHeap_index[(right-1)*NUM_THREADS+tx];
				p4_key[tx]=ppqs[bx].parHeap_key[(right-1)*NUM_THREADS+tx];
			}
			#ifdef DEBUG_DELETE
			if(tx==0){
				printf("Right Node........\n");
				for(i=0;i<offset;i++){
					printf("[%d,%d] ",ppqs[bx].parHeap_index[(right-1)*NUM_THREADS+i],ppqs[bx].parHeap_key[(right-1)*NUM_THREADS+i]);
				}
				printf("\n");
				printf("P5........\n");
				for(i=0;i<(NUM_THREADS+NUM_THREADS);i++){
					printf("[%d,%d] ",p5_index[i],p5_key[i]);
				}
			}
			#endif	
			#ifdef WITH_SYNC
			__syncthreads();
			#endif
			merge(p4_index,p4_key, offset,p5_index,p5_key,NUM_THREADS_2,p1_index,p1_key);
			#ifdef DEBUG_DELETE
			if(tx==0){
				printf("Merging P1........\n");
				for(i=0;i<(NUM_THREADS+NUM_THREADS+offset);i++){
					printf("[%d,%d] ",p1_index[i],p1_key[i]);
				}
				printf("\n");
			}
			#endif	
			if(tx<=offset){
				ppqs[bx].parHeap_index[(right-1)*NUM_THREADS+tx]=p3_index[tx];
				ppqs[bx].parHeap_key[(right-1)*NUM_THREADS+tx]=p3_key[tx];
			}
			ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx]=p1_index[tx];
			ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx]=p1_key[tx];
			ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+tx]=p2_index[tx];
			ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+tx]=p2_key[tx];
		}
		else{
			ppqs[bx].parHeap_index[(parent-1)*NUM_THREADS+tx]=p5_index[tx];
			ppqs[bx].parHeap_key[(parent-1)*NUM_THREADS+tx]=p5_key[tx];
			ppqs[bx].parHeap_index[(left-1)*NUM_THREADS+tx]=p6_index[tx];
			ppqs[bx].parHeap_key[(left-1)*NUM_THREADS+tx]=p6_key[tx];
		}
	}
	else{
		ppqs[bx].parHeap_index[(next-1)*NUM_THREADS+tx]=p3_index[tx];
		ppqs[bx].parHeap_key[(next-1)*NUM_THREADS+tx]=p3_key[tx];
	}
	#ifdef WITH_SYNC
	__syncthreads();
	#endif
}
