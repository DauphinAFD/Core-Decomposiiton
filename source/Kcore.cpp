#include "../include/KCORE.h"

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

vector<int> Make_Bucket(Graph G, int vtx){
    int adj_core;
    vector<int> bins(G.K[vtx],0);
   
    int start = G.EdgeIdx[vtx];
    int stop = G.EdgeIdx[vtx+1];
    // for(int idx = start; idx < stop; idx++){
    //     printf( "%d : %d\n",G.Edges[idx],G.K[G.Edges[idx]]);
    // }

    for(int idx = start; idx < stop; idx++){
        adj_core = G.K[G.Edges[idx]];
        if(adj_core > G.K[vtx])
            adj_core = G.K[vtx];
        bins[G.K[vtx] - adj_core] += 1;
        // if(vtx == 2 && G.Edges[idx] == 1){
                      
        //     printf("%d : %d : %d : %d\n",idx,G.Edges[idx],G.K[G.Edges[idx]],adj_core);
            
        // }
    
        
    }
    
    return bins;
}

int Next_Kcore(vector<int> bucket){
    int k_core = bucket.size() - 1;
    // cout << bucket.size() << " : bucket size \n";
    for(auto v : bucket){
        if(v >= k_core){
            return v;
        }
        k_core -= 1;
    }
    return -1;
}

void K_CoreAlgo(Graph &G){

    vector< int > bucket;
    bool update = false;
    int prev_K;
    do{
       
        for(int v = 0; v < G.V; v++){
            prev_K = G.K[v];
            bucket = Make_Bucket(G,v);
            inclusive_scan(bucket.begin(), bucket.end(), bucket.begin());
            G.K[v] = Next_Kcore(bucket);
           
            if(prev_K > G.K[v] && update == false){
                update = true;
            }
            else{
                update = false;
            }
        }

    }while(update);
    
    
}

void Print_K_Core(Graph &G, int k){
    int start, end;
    for(int i = 0; i < G.V; i++){
        if(G.K[i] >= k){
            start = G.EdgeIdx[i];
            end = G.EdgeIdx[i + 1];
            printf("[ %d ] ",i);
            for(int u = start; u < end; u++){
                if(G.K[G.Edges[u]] >= k){
                    printf("-> %d ",G.Edges[u]);
                }
                
            }
            cout << endl;
        }
    }

}