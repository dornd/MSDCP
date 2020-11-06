#include <bits/stdc++.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define INF INT_MAX

using namespace std;

int N;

template <typename T>
struct Edge {
    int u, v;
    T dissimilarity;
};

template <typename T>
struct edgeCmp {
    __host__ __device__
    bool operator()(const Edge<T>& e1, const Edge<T>& e2) const {
        return e1.dissimilarity < e2.dissimilarity;
    }
};

struct Threads {
    dim3 dimBlocks;
    dim3 dimGrids;

    Threads(int x, int y) {

        if (x > 32) {
            dimBlocks.x = (int)ceil(x/32.0);
            dimGrids.x = 32;
        } else {
            dimGrids.x = x;
        }

        if (y > 32) {
            dimBlocks.y = (int)ceil(y/32.0);
            dimGrids.y = 32;
        } else {
            dimGrids.y = y;
        }

    }
};

int findSet(int* parent, int i) {
    return (parent[i] == i) ? i : (parent[i] = findSet(parent, parent[i]));
}

template <typename T>
thrust::host_vector<int> getCandidateDiameter(thrust::device_vector<Edge<T>> edges) {
    
    thrust::host_vector<int> eT, eC;
    thrust::host_vector<thrust::host_vector<int>> MST(N);
    int* parent = new int[N];
    int* rank = new int[N];

    for (int i = 0; i < N; ++i)
        parent[i] = i, rank[i] = 0;
    
    for (int i = edges.size()-1; i >= 0; i--) {
        Edge<T> e = edges[i];

        int u = e.u;
        int v = e.v;
        
        int x = findSet(parent, u);
        int y = findSet(parent, v);

        if (x != y) {

            eT.push_back(i);
            if (rank[x] > rank[y]) {
                parent[y] = x;
            } else {
                parent[x] = y;
                if (rank[x] == rank[y])
                    rank[y]++;
            }

            MST[u].push_back(v);
            MST[v].push_back(u);

        } else {
            eC.push_back(i);
        }
    }

    thrust::host_vector<int> dist(N, INF);
    dist[0] = 0;

    queue<int> q;
    q.push(0);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int j = 0; j < (int)MST[u].size(); ++j) {
            int v = MST[u][j];
            if (dist[v] == INT_MAX) {
                dist[v] = dist[u]+1;
                q.push(v);
            }
        }
    }

    int b, r;
    for (int i = 0; i < (int)eC.size(); ++i) {
        Edge<T> e = edges[eC[i]];

        int u = e.u;
        int v = e.v;
        b = e.dissimilarity;

        if (dist[u] % 2 == dist[v] % 2) {
            r = eC[i]; 
            break;
        }
    }

    thrust::host_vector<int> eB;
    for (int i = 0; i < (int)eT.size(); ++i) {

        Edge<T> e = edges[eT[i]];
        int w = e.dissimilarity;

        if (w > b)
            eB.push_back(eT[i]);
        else
            break;
    }
    
    eB.push_back(r);

    return eB;
}

template<typename T>
__global__ void construct_boolean_expression(
            bool* d_G, Edge<T>* E, T d1, T d2, int V, int N, int len_E) {

    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = i*N+j;

    if (idx < len_E) {

        int u_i = E[idx].u;
        int u_j = E[idx].v;
        int d_ij = E[idx].dissimilarity;

        if (d1 < d_ij) {

            d_G[u_i*V+(u_j+N)] = true;  // (u_i' -> u_j)
            d_G[u_j*V+(u_i+N)] = true;  // (u_j' -> u_i)
            d_G[(u_i+N)*V+u_j] = true;  // (u_i  -> u_j')
            d_G[(u_j+N)*V+u_i] = true;  // (u_j  -> u_i')

        } else if (d2 < d_ij) {

            d_G[(u_i+N)*V+u_j] = true;  // (u_i -> u_j')
            d_G[(u_j+N)*V+u_i] = true;  // (u_j -> u_i')

        }
    }
}

__global__ 
void check_satisfiability(bool* d_G, bool* cycle, int V, int N, int k) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < V && j < V) {

        d_G[i*V+j] = d_G[i*V+j] || (d_G[i*V+k] && d_G[k*V+j]);
        cycle[i] = true; // no cycle = 1, cycle = 0

        __syncthreads();

        if (k == V-1)
            cycle[i] = !(d_G[i*V+(i+N)] && d_G[(i+N)*V+i]);
    }
}

template <typename T>
T solve(thrust::device_vector<Edge<T>> edges) {

    int V = 2*N;
    thrust::sort(edges.begin(), edges.end(), edgeCmp<T>());
    thrust::host_vector<int> eB = getCandidateDiameter<int>(edges);

    Edge<T> e_ans = edges[edges.size()-1];
    T ans = e_ans.dissimilarity;

    struct Threads t_n2 = Threads(N, N);
    struct Threads t_v2 = Threads(V, V);
    Edge<T>* e_ptr = thrust::raw_pointer_cast(&edges[0]);

    for (int i = 0; i < (int)eB.size(); ++i) {
        int l = 0;
        int r = eB[i]-1;

        Edge<T> e1 = edges[eB[i]];
        T d1 = e1.dissimilarity;

        while (l <= r) {
            int m = (l+r)/2;
            Edge<T> e2 = edges[m];
            T d2 = e2.dissimilarity;
            
            bool* d_G, *cycle;
            cudaMallocManaged(&d_G, V*V*sizeof(bool));
            cudaMallocManaged(&cycle, V*sizeof(bool));

            construct_boolean_expression<T><<<t_n2.dimBlocks, t_n2.dimGrids>>>(
                                        d_G, e_ptr, d1, d2, V, N, (int)edges.size());
            cudaDeviceSynchronize();

            for (int k = 0; k < V; ++k) {
                check_satisfiability<<<t_v2.dimBlocks, t_v2.dimGrids>>>(d_G, cycle, V, N, k);
                cudaDeviceSynchronize();
            }

            bool satisfy = thrust::reduce(cycle, cycle+N, true, thrust::bit_and<bool>());

            vector<vector<int>> G(V, vector<int>(V, 0));

            if (satisfy)
                r = m-1, ans = min(ans, d1+d2);
            else 
                l = m+1;

            cudaFree(d_G);
            cudaFree(cycle);

        }
    }

    return ans;
}

int main() {

    while (cin >> N) {
        
        thrust::device_vector<Edge<int>> edges;
        for (int i = 0; i < N; i++) {
            for (int j = i+1; j < N; j++) {
                int d;
                cin >> d;
                Edge<int> e = {i, j, d};
                edges.push_back(e);
            }

            Edge<int> e = {i, i, 0};
            edges.push_back(e);
        }
        
        int ans = solve<int>(edges);
        printf("%d\n", ans);
    }
    
    return 0;
}
