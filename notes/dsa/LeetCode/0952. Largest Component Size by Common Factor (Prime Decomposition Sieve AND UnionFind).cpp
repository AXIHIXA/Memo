static const int init = []
{ 
    std::ios::sync_with_stdio(false); 
    std::cin.tie(nullptr); 
    std::cout.tie(nullptr); 
    return 0; 
}();

class UnionFind
{
public:
    UnionFind(int n)
    {
        root.assign(n, 0);
        std::iota(root.begin(), root.end(), 0);
        rank.assign(n, 1);
    }

    int find(int x)
    {
        if (x == root[x]) return x;
        else return root[x] = find(root[x]);
    }

    void merge(int x, int y)
    {
        if (int rx = find(x), ry = find(y); rx != ry)
        {
            if (rank[rx] < rank[ry]) root[rx] = ry, rank[ry] += rank[rx];
            else root[ry] = rx, rank[rx] += rank[ry];
        }
    }

    std::vector<int> root;
    std::vector<int> rank;  // Actual size of group, differs from vanilla version. 
};

class Solution
{
public:
    int largestComponentSize(std::vector<int> & nums)
    {
        int n = nums.size();
        if (n == 1) return 1;

        int maximum = *std::max_element(nums.cbegin(), nums.cend());
        UnionFind uf(maximum + 1);  // PLUS ONE or heap overflow in UnionFind!
        
        // Maps a number to its smallest prime factor. 
        std::unordered_map<int, int> firstPrimeFactor;

        for (int x : nums)
        {
            std::vector<int> pfs = primeDecompose(x);
            pfs.erase(std::unique(pfs.begin(), pfs.end()), pfs.end());
            
            firstPrimeFactor.emplace(x, pfs.front());

            for (int i = 0; i < pfs.size() - 1; ++i)
            {
                uf.merge(pfs[i], pfs[i + 1]);
            }
        }

        int ans = 1;

        std::unordered_map<int, int> groupCount;

        for (int x : nums)
        {
            int groupId = uf.find(firstPrimeFactor.at(x));
            auto it = groupCount.find(groupId);
            if (it == groupCount.end()) it = groupCount.emplace(groupId, 1).first;
            else ++it->second;
            ans = std::max(ans, it->second);
        }

        return ans;
    }

private:
    static std::vector<int> primeDecompose(int x)
    {
        std::vector<int> ans;

        for (int factor = 2; factor * factor <= x; )
        {
            if (x % factor == 0)
            {
                x /= factor;
                ans.emplace_back(factor);
            }
            else
            {
                ++factor;
            }
        }

        ans.emplace_back(x);

        return ans;
    }

    static int gcd(int a, int b)
    {
        if (a < b) std::swap(a, b);

        while (b)
        {
            int t = b;
            b = a % b;
            a = t;
        }

        return a;
    }
};