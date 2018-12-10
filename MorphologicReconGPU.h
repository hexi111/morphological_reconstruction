#ifndef MORPHOLOGICRECONGPU_H
#define MORPHOLOGICRECONGPU_H

#include <time.h>
#include <sys/time.h>

#define MAX_NUM_BLOCKS	70
#define WARP_SIZE 	32
#define NUM_THREADS_SCAN	512
#define NUM_WARPS (NUM_THREADS_SCAN / WARP_SIZE)
#define LOG_NUM_THREADS 9
#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)

#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

#define XX_THREADS 4
#define XY_THREADS 64
#define MAX_THREADS 256
#define BITS 8

// 自定义
#define NUM_OF_BLOCKS 256
#define ITEM_TYPE PPQ_ITEM

/*
#define NUM_THREADS 128 //Assuming thread size is equal to r. r is power of 2
#define NUM_THREADS_2  256
#define NUM_THREADS_3 384 
#define NUM_THREADS_4 512 
#define NUM_ITEMS 640
#define NUM_ITEMS_2 1280
*/

/*
#define NUM_THREADS 8 //Assuming thread size is equal to r. r is a power of 2
#define NUM_THREADS_2 16 
#define NUM_THREADS_3 24 
#define NUM_THREADS_4 32 
#define NUM_ITEMS 40
#define NUM_ITEMS_2  80
*/

/*
#define NUM_THREADS 16 //Assuming thread size is equal to r. r is a power of 2
#define NUM_THREADS_2 32 
#define NUM_THREADS_3 48 
#define NUM_THREADS_4 64 
#define NUM_ITEMS 80
#define NUM_ITEMS_2  160
*/

/*
#define NUM_THREADS 32 //Assuming thread size is equal to r. r is a power of 2
#define NUM_THREADS_2 64 
#define NUM_THREADS_3 96 
#define NUM_THREADS_4 128 
#define NUM_ITEMS 160
#define NUM_ITEMS_2  320
*/

/*
#define NUM_THREADS 64 //Assuming thread size is equal to r. r is a power of 2
#define NUM_THREADS_2 128 
#define NUM_THREADS_3 192 
#define NUM_THREADS_4 256 
#define NUM_ITEMS 320
#define NUM_ITEMS_2  640
*/

/*
#define NUM_THREADS 256 //Assuming thread size is equal to r. r is power of 2
#define NUM_THREADS_2  512
#define NUM_THREADS_3 768 
#define NUM_THREADS_4 1024 
#define NUM_ITEMS 1280
#define NUM_ITEMS_2 2560
*/

#define NUM_THREADS 512 //Assuming thread size is equal to r. r is power of 2
#define NUM_THREADS_2  1024
#define NUM_THREADS_3 1536 
#define NUM_THREADS_4 2048 
#define NUM_ITEMS 2560
#define NUM_ITEMS_2 5120

#define WITH_SYNC
#define DEBUG_CONSTR

// 一个并发堆节点
typedef struct{
	int index;
	unsigned char key;
}
PPQ_ITEM;

// 一个并发堆（含有若干个并发堆节点）
typedef struct{
	//heap
	int * parHeap_index; // 堆节点	
	int * init_index; // 初始化区
	int * new_items_index; // 缓冲区
	int numOfItems; // 个数
	unsigned char * parHeap_key;
	unsigned char * init_key;
	unsigned char * new_items_key;
} 
PPQ_TYPE;

// 计时函数
namespace cci {
	namespace common {
		class event {
			public:
				static inline long long timestampInUS(){
					struct timeval ts;
					gettimeofday(&ts, NULL);
					return (ts.tv_sec*1000000LL + (ts.tv_usec));
				};
		};
	}
}

#endif