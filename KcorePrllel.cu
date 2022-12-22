#include "./include/HELPER.cuh"
#include "./include/DECOMPOSITION.cuh"

using namespace std;

int main(){
   
    string filename = "./text/kcore.txt";

    Graph G = Read(filename);
    cout << G.V << " " << G.E << endl;
    cout << "Completed Reading the file\n "<<endl ;
    
    find_Deg_and_Core(G);

    cout << "Coreness of each node at first : \n";
    Print_K_Core(G);

    D_Graph DevG;

    // Preparing device Variables
    prepareDevice(G, DevG);
    cout << endl << "Prepare Device Successful" << endl;
    
    // Running K core
    K_CoreAlgo(G, DevG);

    cout << "\nCoreness of each node at end : " << endl;
    Print_K_Core(G);

    deleteVars(G, DevG);

    
    cout << "Finished successfully";
    return 0;

}