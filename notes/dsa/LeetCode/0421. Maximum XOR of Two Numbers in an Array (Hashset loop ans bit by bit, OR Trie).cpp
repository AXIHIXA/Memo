class Solution
{
public:
    int findMaximumXOR(std::vector<int> & nums)
    {
        auto n = static_cast<int>(nums.size());
        if (n == 1) return 0;
        if (n == 2) return nums[0] ^ nums[1];

        int ans = 0;
        std::unordered_set<int> seen;
        
        for (int i = 31, mask = 0, xorCandidate = 0; 0 <= i; --i)
        {
            mask |= (1 << i);
            xorCandidate = ans | (1 << i);
            seen.clear();

            for (int x : nums)
            {
                x &= mask;

                if (seen.contains(xorCandidate ^ x))
                {
                    ans = xorCandidate;
                    break;
                }

                seen.insert(x);
            }
        }

        return ans;
    }

private:
    int findMaximumXORTrie(std::vector<int> & nums)
    {
        auto n = static_cast<int>(nums.size());
        if (n == 1) return 0;
        if (n == 2) return nums[0] ^ nums[1];
        
        clear();
        for (int x : nums) insert(x);

        int ans = 0;

        for (int x : nums)
        {
            int cur = 1;
            int xorCandidate = 0;

            for (int i = 31, bit, revBit; 0 <= i; --i)
            {
                bit = (x >> i) & 1;
                revBit = bit ^ 1;

                if (tree[cur][revBit])
                {
                    xorCandidate |= (1 << i);
                    cur = tree[cur][revBit];
                }
                else
                {
                    cur = tree[cur][bit];
                }
            }

            ans = std::max(ans, xorCandidate);
        }

        return ans;
    }

    void insert(int x)
    {
        int cur = 1;

        for (int i = 31, bit; 0 <= i; --i)
        {
            bit = (x >> i) & 1;
            if (!tree[cur][bit]) tree[cur][bit] = ++cnt;
            cur = tree[cur][bit];
        }
    }

    void clear()
    {
        for (auto & node : tree) node[0] = node[1] = 0;
        cnt = 1;
    }

private:
    static constexpr int kMaxSize = 3'200'000;
    std::array<std::array<int, 2>, kMaxSize> tree = {{0}};
    int cnt = 1;
};