class Solution
{
public:
    int findCircleNum(std::vector<std::vector<int>> & isConnected)
    {
        auto n = static_cast<const int>(isConnected.size());
        
        root.resize(n);
        std::iota(root.begin(), root.end(), 0);
        rank.resize(n, 1);
        
        int ans = n;

        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                if (isConnected[i][j] && find(i) != find(j))
                {
                    --ans;
                    merge(i, j);
                }
            }
        }

        return ans;
    }

private:
    int find(int x)
    {
        if (root[x] == x) return x;
        return root[x] = find(root[x]);
    }

    void merge(int x, int y)
    {
        int rx = find(x);
        int ry = find(y);
        if (rx == ry) return;

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

    std::vector<int> root;
    std::vector<int> rank;
};