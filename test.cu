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
thrust::device_vector<int> getCandidateDiameter(thrust::device_vector<Edge<T>> edges) {
    
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

    thrust::device_vector<int> eB;
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

struct P {
    bool *G;
    int sum = INT_MAX;
};

template <typename T>
__device__ bool check_satisfy(bool* G, Edge<T>* E, T d1, T d2, int len_E, int N, int V) {

    for (int e = len_E-1; e >= 0; e--) {

        int i = E[e].u;
        int j = E[e].v;
        T d_ij = E[e].dissimilarity; 

        if (d1 < d_ij) {
            G[i*V+(j+N)] = true; // (u_i' -> u_j)
            G[j*V+(i+N)] = true; // (u_j' -> u_i)
            G[(i+N)*V+j] = true; // (u_i -> u_j')
            G[(j+N)*V+i] = true; // (u_j -> u_i');

        } else if (d2 < d_ij) {

            G[(i+N)*V+j] = true;
            G[(j+N)*V+i] = true;
        }
    }

    for (int k = 0; k < V; k++) 
        for (int i = 0; i < V; i++) 
            for (int j = 0; j < V; j++) 
                G[i*V+j] = G[i*V+j] || (G[i*V+k] && G[k*V+j]);

    for (int i = 0; i < N; ++i)
        if (G[i*V+(i+N)] && G[(i+N)*V+i])
            return false;

    return true; 
}

template <typename T>
__global__ void solve(P* p, int* eB, Edge<T>* E, int len_eB, int len_E, int N, int V, int n_pair) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < len_eB) {
        T d1 = E[eB[idx]].dissimilarity;
        
        int l = 0;
        int r = eB[idx]-1;
        while (l <= r) {
            int m = (l+r)/2;
            T d2 = E[m].dissimilarity;

            int p_i = idx*len_eB+m;
            bool satisfy = check_satisfy(p[p_i].G, E, d1, d2, len_E, N, V);
            if (satisfy) {
                r = m-1;
                p[p_i].sum = min(p[p_i].sum, d1+d2);
            } else {
                l = m+1;
            }
        }
    }
}

struct pCmp {
    __host__ __device__
    bool operator()(const P& p1, const P& p2) const {
        return p1.sum < p2.sum;
    }
};

template <typename T>
T solve(thrust::device_vector<Edge<T>> edges) {

    int V = 2*N;
    thrust::sort(edges.begin(), edges.end(), edgeCmp<T>());
    thrust::device_vector<int> eB = getCandidateDiameter<int>(edges);

    Edge<T> e_ans = edges[edges.size()-1];
    T ans = e_ans.dissimilarity;

    int len_eB = eB.size();
    int len_E = edges.size();
    int n_pair = len_eB * ceil(log2(len_E));

    thrust::device_vector<P> v_p;
    for (int i = 0; i < len_eB*len_E; ++i) {
        P d_p;
        cudaMallocManaged(&d_p.G, V*V*sizeof(bool));
        v_p.push_back(d_p);
    }

    int* eB_ptr = thrust::raw_pointer_cast(&eB[0]);
    Edge<T>* e_ptr = thrust::raw_pointer_cast(&edges[0]);
    P* p_ptr = thrust::raw_pointer_cast(&v_p[0]);

    solve<int><<<(int)ceil(len_eB/32.0), 32>>>(p_ptr, eB_ptr, e_ptr, len_eB, len_E, N, V, n_pair);
    cudaDeviceSynchronize();

    P minimum = *thrust::min_element(thrust::device, v_p.begin(), v_p.end(), pCmp());
    ans = minimum.sum;

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
        
        clock_t begin = clock();

        int ans = solve<int>(edges);

        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        printf("GPU time(secs) = %.10lf\n", elapsed_secs);

        printf("%d\n", ans);
    }
    
    return 0;
}
