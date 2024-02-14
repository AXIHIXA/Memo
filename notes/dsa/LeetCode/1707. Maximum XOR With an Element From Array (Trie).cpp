class Solution
{
public:
    std::vector<int> maximizeXor(std::vector<int> & nums, std::vector<std::vector<int>> & queries)
    {
        std::sort(nums.begin(), nums.end());

        std::vector<std::tuple<int, int, int>> q;
        q.reserve(queries.size());
        for (int i = 0; i < queries.size(); ++i) q.emplace_back(queries[i][1], queries[i][0], i);
        std::sort(q.begin(), q.end());

        std::vector<int> ans(queries.size(), -1);

        int i = 0;
        clear();

        for (const auto [m, x, offset] : q)
        {
            if (m < nums.front()) continue;
            while (i < nums.size() && nums[i] <= m) insert(nums[i++]);
            ans[offset] = getMaxXor(x);
        }

        return ans;
    }

private:
    int getMaxXor(int x)
    {
        int cur = 1;
        int ans = 0;

        for (int i = 31, bit, reversedBit; 0 <= i; --i)
        {
            bit = (x >> i) & 1;
            reversedBit = bit ^ 1;

            // We go as deep as 32 layers, 
            // we have inserted multiple int32s into the trie
            // so at least one branch must be non-null. 
            if (tree[cur][reversedBit])
            {
                ans |= (1 << i);
                cur = tree[cur][reversedBit];
            }
            else
            {
                cur = tree[cur][bit];
            }
        }

        return ans;
    }

    void insert(int x)
    {
        int cur = 1;

        for (int i = 31, path; 0 <= i; --i)
        {
            path = (x >> i) & 1;
            if (!tree[cur][path]) tree[cur][path] = ++cnt;
            cur = tree[cur][path];
        }
    }

    void clear()
    {
        for (auto & node : tree) node[0] = node[1] = 0;
        cnt = 1;
    }

private:
    static constexpr int kMaxSize = 2'000'000;
    std::array<std::array<int, 2>, kMaxSize> tree = {{0}};
    int cnt = 1;
};