#ifndef KCORE_H
#define KCORE_H

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>
#include <sstream>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <cmath>
#include <array> 

struct Graph{
    int V, E;
    int *EdgeIdx, *Edges, *K, *Deg; 
};

Graph Read(std::string );
void K_CoreAlgo(Graph & );
void Print_K_Core(Graph &, int );

#endif /*KCORE_H*/