class Solution
{
public:
    int minCostToSupplyWater(int n, std::vector<int> & wells, std::vector<std::vector<int>> & pipes)
    {
        root.resize(n + 10, 0);
        std::iota(root.begin(), root.end(), 0);
        e.reserve(pipes.size() + n + 10);

        // Assume there's a dummy house #0, 
        // #0 has an edge to each house #i (1 <= i <= n), 
        // with weight wells[i - 1]. 
        // We simply solve the MST of this new graph. 
        for (int i = 1; i <= n; ++i)
        {
            e.emplace_back(0, i, wells[i - 1]);
        }

        for (const auto & ee : pipes)
        {
            e.emplace_back(ee[0], ee[1], ee[2]);
        }

        std::sort(e.begin(), e.end(), [](const auto & a, const auto & b)
        {
            return std::get<2>(a) < std::get<2>(b);
        });

        int ans = 0;

        for (auto [s, t, w] : e)
        {
            if (!connected(s, t))
            {
                unite(s, t);
                ans += w;
            }
        }

        return ans;
    }

private:
    int find(int x)
    {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

    void unite(int x, int y)
    {
        int rx = find(x), ry = find(y);
        if (rx == ry) return;
        root[rx] = ry;
    }

    bool connected(int x, int y)
    {
        return find(x) == find(y);
    }

    std::vector<int> root;

    using Source = int;
    using Target = int;
    using Weight = int;
    std::vector<std::tuple<Source, Target, Weight>> e;
};