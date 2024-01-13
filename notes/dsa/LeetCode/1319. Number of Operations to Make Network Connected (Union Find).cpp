class UnionFind
{
public:
    explicit UnionFind(int sz) : root(sz), rank(sz, 1)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    int find(int x)
    {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

    void merge(int x, int y)
    {
        if (int rx = find(x), ry = find(y); rx != ry)
        {
            if (rank[rx] < rank[ry])
            {
                root[rx] = ry;
            }
            else if (rank[ry] < rank[rx])
            {
                root[ry] = rx;
            }
            else
            {
                root[ry] = rx;
                ++rank[rx];
            }
        }
    }

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

private:
    std::vector<int> root;
    std::vector<int> rank;
};


class Solution 
{
public:
    int makeConnected(int n, vector<vector<int>> & connections) 
    {
        if (connections.size() < n - 1) 
        {
            return -1;
        }

        UnionFind uf(n);

        int numConnectedComponents = n;

        for (const auto & connection : connections)
        {
            if (not uf.connected(connection[0], connection[1]))
            {
                --numConnectedComponents;
                uf.merge(connection[0], connection[1]);
            }
        }

        return numConnectedComponents - 1;
    }
};