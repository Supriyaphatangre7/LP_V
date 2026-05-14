#include<iostream>
#include<vector>
#include<queue>
#include<omp.h>
using namespace std;

class Graph{
    int V;
    vector<vector<int>>adj;

    public:
    Graph(int v)
    {
        V=v;
        adj.resize(V);
    }

    void addEdge(int u,int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void BFS(int start)
    {
        vector<int>visited(V,0);
        queue<int>q;

        visited[start]=1;
        q.push(start);

        cout<<"parallel bfs: ";

        while(!q.empty())
        {
            int node=q.front();
            q.pop();
            cout<<node<<" ";

            #pragma omp parallel for
            for(int i=0;i<adj[node].size();i++)
            {
                int next=adj[node][i];

                if(!visited[next])
                {
                    #pragma omp critical
                    {
                         if(!visited[next])
                         {
                            visited[next]=1;
                            q.push(next);
                         }
                    }
                }
            }
        }
    }

    void DfsUtil(int node,vector<int> & visited)
    {
        visited[node]=1;
        #pragma omp critical
        cout<<node<<" ";

        #pragma omp parallel for
        for(int i=0;i<adj[node].size();i++)
        {
            int next=adj[node][i];
            if(!visited[next])
            {
                DfsUtil(next,visited);
            }
        }
    }

    void DFS(int start)
    {
        vector<int>visited(V,0);
        cout<<"parallel dfs";
         DfsUtil(start,visited);

    }
};

int main()
{
    int V,E;
    cout<<"enter the no of vertices:";
    cin>>V;

    cout<<"enetr no of edges:";
    cin>>E;

    Graph g(V);

    cout<<"enter edges:";
    for(int i=0;i<E;i++)
    {
        int u,v;
        cin>>u>>v;
        g.addEdge(u,v);
    }
    int start;

    cout<<"enter starting vertex:";
    cin>>start;

   g. BFS(start);
   g. DFS(start);


}