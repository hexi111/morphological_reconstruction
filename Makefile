# for linking problem https://stackoverflow.com/questions/27590166/how-to-compile-multiple-files-in-cuda
compile:
	nvcc -o imreconGPU util.cu scan4.cu radixSort.cu MorphologicReconGPU.cu -rdc=true -L"/home/ytao3/opencv/lib64" -I"/home/ytao3/opencv/include"  -lopencv_core -lopencv_imgcodecs -std=c++11
run:
	./imreconGPU img/100-percent-marker.jpg img/100-percent-mask.jpg
	#./imreconGPU img/new-marker.jpg img/new-mask.jpg
compile1:
	nvcc -o imreconSeq MorphologicReconSeq.cpp -L"/home/ytao3/opencv/lib64" -I"/home/ytao3/opencv/include"  -lopencv_core -lopencv_imgcodecs -std=c++11
run1:
	./imreconSeq img/100-percent-marker.jpg img/100-percent-mask.jpg
	#./imreconSeq img/new-marker.jpg img/new-mask.jpg
compile2:
	nvcc -o imreconTest test.cu -L"/home/ytao3/opencv/lib64" -I"/home/ytao3/opencv/include"  -lopencv_core -lopencv_imgcodecs -std=c++11
run2:
	./imreconTest 

