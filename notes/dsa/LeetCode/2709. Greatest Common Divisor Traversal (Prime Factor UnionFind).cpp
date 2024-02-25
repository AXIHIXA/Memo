class UnionFind
{
public:
    explicit UnionFind(int size) : root(size), rank(size, 1)
    {
        std::iota(root.begin(), root.end(), 0);
    }

    int find(int x)
    {
        if (x == root[x]) return x;
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

private:
    std::vector<int> root;
    std::vector<int> rank;
};

class Solution
{
public:
    bool canTraverseAllPairs(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        if (n == 1) return true;
        for (int x : nums) if (x == 1) return false;

        int maxi = *std::max_element(nums.cbegin(), nums.cend());
        UnionFind uf(1 + maxi);

        // fpf.at(i): First prime factor of nums[i]. 
        std::unordered_map<int, int> fpf;
        std::vector<int> tmp;
        
        for (int i = 0; i < n; ++i)
        {
            primeDecomposition(nums[i], tmp);
            fpf.emplace(i, tmp.front());
            for (int f : tmp) uf.unite(tmp.front(), f);
        }

        for (int i = 0; i < n; ++i)
        {
            if (!uf.connected(fpf.at(i), fpf.at(0)))
            {
                return false;
            }
        }

        return true;
    }

private:
    static void primeDecomposition(int x, std::vector<int> & ans)
    {
        ans.clear();

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
    }
};