class Solution
{
public:
    int numSimilarGroups(std::vector<std::string> & strs)
    {
        std::iota(root.begin(), root.end(), 0);
        std::fill(rank.begin(), rank.end(), 0);

        auto n = static_cast<const int>(strs.size());
        segments = n;

        auto diff = [](const std::string & a, const std::string & b) -> int
        {
            auto n = static_cast<const int>(a.size());
            int ans = 0;

            for (int i = 0; i < n; ++i)
            {
                ans += a[i] != b[i];
            }

            return ans;
        };

        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                int dif = diff(strs[i], strs[j]);
                
                if (dif == 0 || dif == 2)
                {
                    unite(i, j);
                }
            }
        }

        return segments;
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

        --segments;

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[ry] < rank[rx]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

private:
    static constexpr int kMaxSize = 310;

    static std::array<int, kMaxSize> root;
    static std::array<int, kMaxSize> rank;

private:
    int segments = 0;
};

std::array<int, Solution::kMaxSize> Solution::root = {0};
std::array<int, Solution::kMaxSize> Solution::rank = {0};
