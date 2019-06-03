这个项目主要是为发表于计算机应用中的一篇论文服务的。论文里，我们利用平行堆集群来构建一个并行的形态学重建系统，并与单线程的形态学重建系统进行比较。

运行环境：

1. 安装cmake. 现在我装到是3.11.4. 版本太低无法与别的软件工作。
	
	./bootstrap
	./make
	不需要make install, 因为可以通过路径直接引用cmake
   
2. 因为系统装了CUDA 9,所以只能安装opencv 3 或者opencv 4. 以前的opencv 2.4.5无法兼容CUDA 9。
	
	在opencv源文件的目录里
	mkdir build
	cd build
	/home/ytao3/cmake-3.11.4/bin/cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/ytao3/opencv  -DWITH_CUDA=ON 
	make
	make install

3. 编译程序

	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ytao3/opencv/lib64
	nvcc -o imreconTest MorphologicReconGPU.cu -L"/home/ytao3/opencv/lib64" -I"/home/ytao3/opencv/include"  -lopencv_core -lopencv_imgcodecs -std=c++11
  
  
实验步骤:

1. make compile 编译并行的形态学重建系统

2. make run 运行并行的形态学重建系统

3. make compile1 编译单线程的形态学重建系统

4. make run 运行单线程的形态学重建系统
