#include "./include/KCORE.h"

using namespace std;

int main(){

    string filename = "./text/kcore.txt";
    int k = 3;
    Graph G = Read(filename);
    int start,end;
    cout << G.V << G.E << endl;
    cout << "Completed Reading the file\n "<<endl ;
    
    for(int i = 0; i < G.V; i++){
        start = G.EdgeIdx[i];
        end = G.EdgeIdx[i + 1];
        G.Deg[i] = end - start;
        G.K[i] = G.Deg[i];
    }
    cout << "K core at first : \n";
    for(int i = 0; i < G.V; i++){
        cout << G.K[i] << " ";
    }
    cout << endl;
    
    K_CoreAlgo(G);
    cout << "After that \n";
    Print_K_Core(G,k);
    
    
    return 0;
}

   

