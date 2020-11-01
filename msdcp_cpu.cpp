#include <bits/stdc++.h>

using namespace std;

#define INF INT_MAX
int N;

int findSet(int* parent, int i) {
    return (parent[i] == i) ? i : (parent[i] = findSet(parent, parent[i]));
}

vector<int> getCandidateDiameter(
                        vector<pair<int, pair<int, int>>> edges) {

    vector<int> eT, eC;
    vector<vector<int>> MST(N);

    int parent[N], rank[N];    
    for (int i = 0; i < N; ++i)
        parent[i] = i, rank[i] = 0;

    for (int i = edges.size()-1; i >= 0; i--) {

        int u = edges[i].second.first;
        int v = edges[i].second.second;

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

    vector<int> dist(N, INF);
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
        pair<int, pair<int, int>> e = edges[eC[i]];

        int u = e.second.first;
        int v = e.second.second;
        b = e.first;

        if (dist[u] % 2 == dist[v] % 2) {
            r = eC[i]; 
            break;
        }
    }

    vector<int> eB;
    for (int i = 0; i < (int)eT.size(); ++i) {
        pair<int, pair<int, int>> e = edges[eT[i]];

        int u = e.second.first;
        int v = e.second.second;
        int w = e.first;
        if (w > b)
            eB.push_back(eT[i]);
        else
            break;
    }
    
    eB.push_back(r);

    return eB;
}

vector<vector<int>> construct_boolean_expression(
            vector<pair<int, pair<int, int>>> edges, int d1, int d2, int V) {

    vector<vector<int>> G(V);

    for (int e = edges.size()-1; e >= 0; e--) {

        double d_ij = edges[e].first;

        double i = edges[e].second.first;
        double j = edges[e].second.second;

        if (d1 < d_ij) {
            G[i].push_back(j+N); // (u_i' -> u_j)
            G[j].push_back(i+N); // (u_j' -> u_i)
            G[i+N].push_back(j); // (u_i -> u_j')
            G[j+N].push_back(i); // (u_j -> u_i');
        } else if (d2 < d_ij) {
            G[i+N].push_back(j);
            G[j+N].push_back(i);
        }
    }

    return G;
}

void tarjanSCC(vector<vector<int>>& G, int u, int dfs_num[], 
                int dfs_low[], bool visited[], stack<int>&s, int *numSCC, int SCCs[]) {
    static int time = 0;
    dfs_low[u] = dfs_num[u] = ++time;
    s.push(u);
    visited[u] = true;
    for(int j = 0; j < (int)G[u].size(); j++) {
        int v = G[u][j];
        if(dfs_num[v] == INF)
            tarjanSCC(G, v, dfs_num, dfs_low, visited, s, numSCC, SCCs);
        if(visited[v])
            dfs_low[u] = min(dfs_low[u], dfs_low[v]);
    }

    int v = 0;
    if (dfs_low[u] == dfs_num[u]) {
        ++(*numSCC);
        while (s.top() != u) {
            v = s.top();
            SCCs[v] = *numSCC;
            visited[v] = false;
            s.pop();
        }

        v = s.top();
        visited[v] = false;
        SCCs[v] = *numSCC;
        s.pop();
    }
}

bool check_satisfiability(vector<vector<int>> G, int V) {

    stack<int> s;
    bool visited[V];
    int dfs_num[V], dfs_low[V], SCCs[V];

    for (int i = 0; i < V; ++i) {
        dfs_num[i] = dfs_low[i] = INF;
        visited[i] = false;
        SCCs[i] = -1;
    }

    int numSCC = 0;
    for (int i = 0; i < V; ++i)
        if (dfs_num[i] == INF)
            tarjanSCC(G, i, dfs_num, dfs_low, visited, s, &numSCC, SCCs);

    for (int i = 0; i < N; ++i)
        if (SCCs[i] == SCCs[i+N])
            return false;

    return true;
}

int solve(vector<pair<int, pair<int, int>>> edges) {

    sort(edges.begin(), edges.end());

    int V = 2*N;
    vector<int> eB = getCandidateDiameter(edges);
    int ans = edges[edges.size()-1].first;

    for (int i = 0; i < (int)eB.size(); ++i) {

        int l = 0;
        int r = eB[i]-1;

        int d1 = edges[eB[i]].first;

        while (l <= r) {
            int m = (l+r)/2;
            int d2 = edges[m].first;

            vector<vector<int>> G = construct_boolean_expression(edges, d1, d2, V);
            bool satisfiable = check_satisfiability(G, V);

            if (satisfiable) {
                r = m-1, ans = min(ans, d1+d2);
            } else {
                l = m+1;
            }
        }
    }

    return ans;
}

int main() {

    while (cin >> N) {

        if (N < 3) {
            cout << 0 << endl;
            return 0;
        }

        vector<pair<int, pair<int, int>>> edges;

        for (int i = 0; i < N; i++) {
          for (int j = i+1; j < N; j++) {
            int d;
            cin >> d;
            edges.push_back(make_pair(d, make_pair(i, j)));
          }
          edges.push_back(make_pair(0, make_pair(i, i)));
        }
        
        int ans = solve(edges);
        printf("%d\n", ans);
    }

    return 0;
}
