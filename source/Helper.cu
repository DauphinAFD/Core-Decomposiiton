#include "../include/HELPER.cuh"

using namespace std;


Graph Read(string filename){

    Graph G;
    int V, E;
    string line;
    stringstream ss;
    int v, u, idx = 0;
    
    fstream k_decom(filename);
    if(k_decom.is_open()){
        
        getline(k_decom,line);
        ss.clear();
        ss.str("");
        ss.str(line);
        ss >> V >> E;
        
        G.V = V;
        G.E = E;

        vector< vector<int> > neigh(V);
        while(getline(k_decom,line)) {
            ss.clear();
            ss.str("");
            ss.str(line);
        
            ss >> v >> u ;
            
            neigh[v].push_back(u);
            if(v != u){
                E++;
                neigh[u].push_back(v);
            }
        }
        
        G.E = E;
        
        size_t size_edge = E * sizeof(int);
        size_t size_vtx = V * sizeof(int);
        
        G.EdgeIdx = (int  *)malloc((V+1) * sizeof(int));
        G.Edges = (int  *)malloc(size_edge);
        G.Deg = (int  *)malloc(size_vtx);
        G.K = (int  *)malloc(size_vtx);
        for(int i = 0; i < V; i++){
            G.EdgeIdx[i] = idx;
            for(auto x : neigh[i]){
                G.Edges[idx] = x;
                idx++;
            }
        }
        G.EdgeIdx[V] = G.E;
        k_decom.close();
    }
    return G;
}

void find_Deg_and_Core(Graph &G){
    int start,end;
    for(int i = 0; i < G.V; i++){
        start = G.EdgeIdx[i];
        end = G.EdgeIdx[i + 1];
        G.Deg[i] = end - start;
        G.K[i] = G.Deg[i];
    }
}


void prepareDevice(Graph &G, D_Graph & DevG){
    int V = G.V;
    int E = G.E;
    printf("%d : %d",V,E);
    //cout << "Entering Function"<<endl;

    size_t size_edge = E * sizeof(int);
    size_t size_vtx = (V) * sizeof(int);

    // CHECK(cudaMalloc((void**)&DevG.update, sizeof(bool)));
    CHECK(cudaMalloc((void**)&DevG.EdgeIdx, (V+1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&DevG.Edges, size_edge));
    CHECK(cudaMalloc((void**)&DevG.Max_K, size_vtx));
    CHECK(cudaMalloc((void**)&DevG.V, sizeof(int)));
    CHECK(cudaMalloc((void**)&DevG.E, sizeof(int)));
    CHECK(cudaMalloc((void**)&DevG.Deg, size_vtx));
    CHECK(cudaMalloc((void**)&DevG.K, size_vtx));

    thrust::fill(thrust::device, DevG.Max_K, DevG.Max_K + V, 0);
    CHECK(cudaMemcpy(DevG.EdgeIdx, G.EdgeIdx, (V+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(DevG.Edges, G.Edges, size_edge, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(DevG.V, &G.V, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(DevG.E, &G.E, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(DevG.Deg, G.Deg, size_vtx, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(DevG.K, G.K, size_vtx, cudaMemcpyHostToDevice)); // Update is not coppied here 

    //cout << "finish Preparing"<<endl;

}


void deleteVars(Graph &G, D_Graph & DevG){

    //cout << "Started deleting" << endl;

    CHECK(cudaFree(DevG.EdgeIdx));
	CHECK(cudaFree(DevG.Edges));
	CHECK(cudaFree(DevG.Deg));
	CHECK(cudaFree(DevG.K));

    free(G.EdgeIdx);
    free(G.Edges);
    free(G.Deg);
    free(G.K);

    //cout << "finish deleting" << endl;
    
}
