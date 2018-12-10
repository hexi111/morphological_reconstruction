#ifndef MORPHOLOGICRECONSEQ_H
#define MORPHOLOGICRECONSEQ_H

#include <time.h>
#include <sys/time.h>

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