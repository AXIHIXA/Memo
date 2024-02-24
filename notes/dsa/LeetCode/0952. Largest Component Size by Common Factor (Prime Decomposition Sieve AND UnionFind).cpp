class Solution
{
public:
    int largestComponentSize(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());

        // UnionFind containing [0, max {nums}]
        int maximum = *std::max_element(nums.cbegin(), nums.cend());
        root.resize(maximum + 1);
        std::iota(root.begin(), root.end(), 0);
        rank.assign(maximum, 0);

        // The UnionFind handle of a number is its first prime factor. 
        // We further union all prime factors for each number in nums. 
        std::unordered_map<int, int> firstPrimeFactor;

        for (int x : nums)
        {
            std::vector<int> pfs = primeDecomposition(x);
            firstPrimeFactor.emplace(x, pfs.front());
            for (int i = 0; i < pfs.size() - 1; ++i) merge(pfs[i], pfs[i + 1]);
        }
        
        int ans = 1;
        std::unordered_map<int, int> groupCount;

        for (int x : nums)
        {
            int pf = firstPrimeFactor.at(x);
            int group = find(pf);
            ++groupCount[group];
            ans = std::max(ans, groupCount.at(group));
        }

        return ans;
    }

private:
    static std::vector<int> primeDecomposition(int x)
    {
        std::vector<int> ans;

        for (int f = 2; f * f <= x; )
        {
            if (x % f == 0)
            {
                ans.emplace_back(f);
                x /= f;
            }
            else
            {
                ++f;
            }
        }

        ans.emplace_back(x);

        return ans;
    }

    int find(int x)
    {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

    void merge(int x, int y)
    {
        int rx = find(x);
        int ry = find(y);
        if (rx == ry) return;

        if (rank[rx] < rank[ry]) root[rx] = ry;
        else if (rank[rx] < rank[ry]) root[ry] = rx;
        else root[ry] = rx, ++rank[rx];
    }

    std::vector<int> root;
    std::vector<int> rank;  // actual sizes! 
};