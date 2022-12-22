#include "../include/DECOMPOSITION.cuh"
#include <cmath>


using namespace std;

struct CheckBucket{
    int minDeg, maxDeg;
	int *edgeIdx;

	CheckBucket(int minimum, int maximum, int *edgesIndex) {
		minDeg = minimum;
		maxDeg = maximum;
		edgeIdx = edgesIndex;
	}

	__host__ __device__ bool operator()(const int &vtx) const {
		int Deg = edgeIdx[vtx + 1] - edgeIdx[vtx];
		return (Deg > minDeg && Deg <= maxDeg);
	}
};

__global__ void Update_K_CORE(D_Graph DevG) {
    // int thread_id = threadIdx.x;
    int vtx = blockIdx.x * blockDim.x + threadIdx.x;
    int V = *DevG.V;
    int prev_k = DevG.K[vtx];
    int max_supported_k = DevG.Max_K[vtx];

    if(vtx < V){

        if(max_supported_k < prev_k){
            DevG.K[vtx] = max_supported_k;
        }
        // printf("%d : %d : %d \n",vtx,prev_k,max_supported_k);
    }
}

__global__ void Next_K_CORE_ESTIMATE(D_Graph DevG, int* vSet, bool* d_update, int No_of_vtx, int Intvelno) {
	extern __shared__ int s[];
    int *buckets = s;                        // nI ints
    int *next_k = (int*)&buckets[blockDim.x]; // nF ints

    // __shared__ int buckets[THREADS];
    // __shared__ int next_k[THREADS];
    int thread_id = threadIdx.x;
    buckets[thread_id] = 0;
    next_k[thread_id] = 0;
    __syncthreads();
    // int V = *DevG.V;
    int vtxIdx = blockIdx.x;
    int block_size = blockDim.x; 
    int *K = DevG.K;
    int *Edges = DevG.Edges;
    int *EdgeIdx = DevG.EdgeIdx;
    int suppported_k = 0;
   
    

    if(vtxIdx < No_of_vtx){
        int vtx = vSet[vtxIdx];
        int prev_k = DevG.K[vtx];
        int limit = floorf(prev_k/128.0)*128 - 1;
        int start = EdgeIdx[vtx];
        int stop = EdgeIdx[vtx+1];
        int adj_core;
        
            // Creating a Bucket if K core is less than bucket size
        if(prev_k < block_size){

            for(int idx = start + thread_id; idx < stop; idx += block_size){
    
                adj_core = K[Edges[idx]];
                if(adj_core > K[vtx]){
                    adj_core = K[vtx];
                }
                atomicAdd(&buckets[K[vtx] - adj_core], 1); //change it later
    
            }

        }
        else{

            for(int idx = start + thread_id; idx < stop; idx += block_size){
    
                adj_core = K[Edges[idx]];
                if(adj_core >= limit + (block_size-1)){
                    atomicAdd(&buckets[0], 1);  
                }
                else if(adj_core <= limit){
                    atomicAdd(&buckets[block_size-1], 1);
                }
                else{
                    atomicAdd(&buckets[adj_core-limit], 1); //change it later
                }
    
            }

        }
            // Inclusive Sum
        for(int stride = 1; stride <= blockDim.x/2; stride *= 2){ 
            int v;
            if(thread_id >= stride) {
                v = buckets[thread_id - stride];
            }
            __syncthreads();

            if(thread_id >= stride) {
                buckets[thread_id] += v;
            }
            __syncthreads();

        }

        if(prev_k < block_size){
            suppported_k =  K[vtx] - thread_id;
        }
        else{
            suppported_k = block_size - thread_id + limit;
        }
        

            // Finding Next K
        if(buckets[thread_id] >= suppported_k){
            next_k[thread_id] = suppported_k;
        }
        else{
            next_k[thread_id] = 0;
        }
        __syncthreads();

        for(int stride = blockDim.x/2; stride > 0; stride /= 2){ //Reduction
            int v;
            if(thread_id < stride) {
                v = next_k[thread_id + stride];
            }
            __syncthreads();

            if(thread_id < stride) {
                next_k[thread_id] =  max(next_k[thread_id],v);
            }
            __syncthreads();
        }
        if(thread_id == 0 ){
            DevG.Max_K[vtx] = next_k[thread_id];
            // *DevG.update = true;
        }

        if(thread_id == 0 && DevG.Max_K[vtx] < prev_k){

            d_update[Intvelno] = true;
        }

    }

}

void K_CoreAlgo(Graph &G, D_Graph &DevG ){
    int N = G.V;
    int *vtxEnd;
    int VtxPerInter;
    int LastIntvelNo = numInterval - 1;
    bool *d_update;
    int *d_partition;
    bool change;
    int *partition = (int *)malloc(N*sizeof(int));
    bool *update = (bool *)malloc(LastIntvelNo*sizeof(bool));
    int BLOCKS;
    dim3 DIMENSIONS;
    CHECK(cudaMalloc((void**)&d_update, LastIntvelNo * sizeof(bool)));
    CHECK(cudaMalloc((void**)&d_partition, N * sizeof(int)));
    
    thrust::sequence(thrust::host, partition, partition + N, 0);
    

    do{
   
        change = false;
        thrust::fill(thrust::host, update, update + LastIntvelNo, false);
        CHECK(cudaMemcpy(d_update, update, LastIntvelNo*sizeof(bool), cudaMemcpyHostToDevice));

        for(int Intvelno = 0; Intvelno < LastIntvelNo; Intvelno++){
            
            vtxEnd = thrust::partition(thrust::host, partition, partition + N, CheckBucket(Intervels[Intvelno], Intervels[Intvelno + 1], G.EdgeIdx));
            VtxPerInter = thrust::distance(partition, vtxEnd);
            cudaMemcpy( d_partition, partition, N*(sizeof(int)), cudaMemcpyHostToDevice);

            if(VtxPerInter > 0){
                DIMENSIONS = threads[Intvelno]; 
                // cout << "Vtx in the Interval" << VtxPerInter << "Interval no :" <<Intvelno << endl;
                Next_K_CORE_ESTIMATE <<< VtxPerInter, threads[Intvelno], 2*DIMENSIONS.x*sizeof(int) >>>(DevG,d_partition,d_update,VtxPerInter,Intvelno);
                // cudaDeviceSynchronize();
            }  

        }

        cudaDeviceSynchronize();
        CHECK(cudaMemcpy(update, d_update, LastIntvelNo*sizeof(bool), cudaMemcpyDeviceToHost));
        BLOCKS = (N + 512 - 1) / 512;
        Update_K_CORE <<< BLOCKS,512 >>>(DevG);
        cudaDeviceSynchronize();

        for(int i = 0; i < LastIntvelNo; i++){
            if(update[i] == true){
                change = true;
                break;
            }
        }

        // cout << "\nUpdate value is : "; //<< update << endl;
        // for(int i = 0; i< LastIntvelNo; i++)
        //     cout << update[i] << " ";
        // cout << endl;
        
    }while(change);

    CHECK(cudaMemcpy( G.K, DevG.K,  N * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_update);
    free(update);


}

void Print_K_Core(Graph &G){
    for(int i = 0; i < G.V; i++){
        cout << G.K[i] << " ";
    }
    cout << endl;

}
