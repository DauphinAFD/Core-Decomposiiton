
K_cpp := KcoreSer.cpp ./source/Kcore.cpp
K_cu := KcorePrllel.cu ./source/Helper.cu ./source/Decomposition.cu

KcoreCPP : $(K_cpp)
	g++ $(K_cpp) -o run
	./run
KcoreCU : $(K_cu)
	nvcc $(K_cu) -o run
	./run

clear :
	rm run
