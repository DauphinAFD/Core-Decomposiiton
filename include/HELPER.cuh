#ifndef HELPER_H
#define HELPER_H

# define CHECK( call )\
{\
    const cudaError_t error = call;\
    if( error != cudaSuccess) \
    {\
        cout << " Error " << __FILE__ << " : " << __LINE__ << endl;\
        cout << " Code : " << error << ", reason : " << cudaGetErrorString(error);\
        exit(1);\
    }\
}


#include <cuda.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include<algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <sstream>
#include <fstream>

__constant__ __device__ float M;


struct Graph{
    int V, E;
    int *EdgeIdx, *Edges, *K, *Deg; 
};

struct D_Graph{
    int *V, *E;
    // bool *update;
    int *EdgeIdx, *Edges, *K, *Max_K, *Deg; 
};

Graph Read(std::string );
void find_Deg_and_Core(Graph & );
// void K_CoreAlgo(Graph & );
// void Print_K_Core(Graph &, int );

void prepareDevice(Graph &, D_Graph &);
void deleteVars(Graph &, D_Graph &);

#endif /* HELPER_H */