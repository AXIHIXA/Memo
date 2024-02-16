static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
    std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
    return 0;
}();

class Solution
{
public:
    int maximumStrongPairXor(std::vector<int> & nums)
    {
        auto n = static_cast<int>(nums.size());
        if (n == 1) return 0;
        std::sort(nums.begin(), nums.end());

        int ans = 0;

        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            insert(nums[rr]);
            // std::printf("insert(%d)\n", nums[rr]);

            while (ll < rr && 2 * nums[ll] < nums[rr])
            {
                erase(nums[ll++]);
                // std::printf("erase(%d)\n", nums[ll - 1]);
            }

            if (ll == rr) continue;

            int cur = 1;
            int res = 0;

            for (int i = kMsb, bit, revBit; 0 <= i; --i)
            {
                bit = (nums[rr] >> i) & 1;
                revBit = bit ^ 1;

                if (tree[cur][revBit])
                {
                    res |= (1 << i);
                    cur = tree[cur][revBit];
                }
                else
                {
                    cur = tree[cur][bit];
                }
            }

            ans = std::max(ans, res);
        }

        return ans;
    }

private:
    void insert(int x)
    {
        if (find(x)) return;
        
        int cur = 1;
        ++pass[cur];
        
        for (int i = kMsb, bit; 0 <= i; --i)
        {
            bit = (x >> i) & 1;
            if (!tree[cur][bit]) tree[cur][bit] = ++cnt;
            cur = tree[cur][bit];
            ++pass[cur];
        }
    }

    int find(int x)
    {
        int cur = 1;

        for (int i = kMsb, bit; 0 <= i; --i)
        {
            bit = (x >> i) & 1;
            if (!tree[cur][bit]) return 0;
            cur = tree[cur][bit];
        }

        return cur;
    }

    void erase(int x)
    {
        if (!find(x)) return;

        int cur = 1;
        --pass[cur];

        for (int i = kMsb, bit; 0 <= i; --i)
        {
            bit = (x >> i) & 1;

            if (!--pass[tree[cur][bit]])
            {
                tree[cur][bit] = 0;
                return;
            }

            cur = tree[cur][bit];
        }
    }

private:
    static constexpr int kMaxSize = 1'500'000;  // 32 * 5 * 1e4
    static constexpr int kMsb = 20;             // 1 <= nums[i] <= 2**20 -1
    std::array<std::array<int, 2>, kMaxSize> tree = {{0}};
    std::array<int, kMaxSize> pass = {0};
    int cnt = 1;
};