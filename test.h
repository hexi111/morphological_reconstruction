#ifndef _TEST_H_
#define _TEST_H_

#include <time.h>
#include <sys/time.h>

#define NUM_THREADS 8 
#define NUM_THREADS_2 16
#define NUM_THREADS_3 24 
#define NUM_THREADS_4 32 

#define NUM_OF_BLOCKS 8

typedef struct{
	int index;
	unsigned char key;
}
PPQ_ITEM;

typedef struct{
	//heap
	PPQ_ITEM *parHeap;	
	PPQ_ITEM *init;
	int numOfItems;
	PPQ_ITEM * new_items;
} 
PPQ_TYPE;

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
