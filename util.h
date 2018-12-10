/*
这个文件主要是放一些例如排序，合并的公用函数
*/

#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdio.h> 
#include "MorphologicReconGPU.h"  

__device__ void swap(PPQ_ITEM * buffer, int source, int target, int ddd);
__device__ void bitonic_sort(PPQ_ITEM * buffer,int arrayLength,int dir);
__device__ void sort_one(PPQ_ITEM * buffer,int num,int * count);
//__device__ void merge(PPQ_ITEM * tempA, int numA, PPQ_ITEM * tempB, int numB,PPQ_ITEM * C);
__device__ void scan_backup(const int* values, int* exclusive);

#endif 
