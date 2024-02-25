class UnionFind
{
public:
    explicit UnionFind(int size) : size(size), root(size), rank(size, 0)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    int find(int x)
    {
        if (root[x] == x) return x;
        return root[x] = find(root[x]);
    }

    void unite(int x, int y)
    {
        int rx = find(x);
        int ry = find(y);
        if (rx == ry) return;

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

    void reset(int x)
    {
        root[x] = x;
        rank[x] = 0;
    }

    void reset()
    {
        std::iota(root.begin(), root.end(), 0);
        rank.assign(size, 0);
    }

private:
    int size = 0;
    std::vector<int> root;
    std::vector<int> rank;
};

class Solution
{
public:
    std::vector<int> findAllPeople(int n, std::vector<std::vector<int>> & meetings, int firstPerson)
    {
        std::sort(meetings.begin(), meetings.end(), [](const auto & a, const auto & b)
        {
            return a.back() < b.back();
        });

        UnionFind uf(n);
        uf.unite(0, firstPerson);
        auto m = static_cast<const int>(meetings.size());

        for (int i = 0, j = 0; i < m && j < m; i = j)
        {
            while (j < m && meetings[j][2] == meetings[i][2]) ++j;

            for (int k = i; k < j; ++k)
            {
                uf.unite(meetings[k][0], meetings[k][1]);
            }
            
            for (int k = i; k < j; ++k)
            {
                if (!uf.connected(meetings[k][0], 0))
                {
                    uf.reset(meetings[k][0]);
                    uf.reset(meetings[k][1]);
                }
            }
        }

        std::vector<int> ans {0};
        ans.reserve(n);

        for (int i = 1; i < n; ++i)
        {
            if (uf.connected(i, 0))
            {
                ans.emplace_back(i);
            }
        }

        return ans;
    }
};