#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

#include "../include/HELPER.cuh"
#include <cuda.h>
#include<algorithm>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/partition.h>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <sstream>
#include <fstream>

// const int THREADS = 128;
const int numInterval = 8;

const int Intervels[8] = {0, 2, 8, 16, 32, 128, 256, INT_MAX};

const dim3 threads[7]{ 
    {32}, {32}, {32}, {32}, {128}, {256}, {512}
};


void K_CoreAlgo(Graph &, D_Graph & );
void Print_K_Core(Graph & );

#endif /* DECOMPOSITION_H */