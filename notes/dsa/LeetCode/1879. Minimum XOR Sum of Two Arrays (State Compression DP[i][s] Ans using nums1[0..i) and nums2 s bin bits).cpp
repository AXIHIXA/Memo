class Solution
{
public:
    int minimumXORSum(std::vector<int> & nums1, std::vector<int> & nums2)
    {
        auto n = static_cast<const int>(nums1.size());

        // State-compression DP. 
        // dp[i][s]: Min XOR sum using nums1[0...i) and s bin bits' indices from nums2. 
        const int kMaxState = 1 << n;
        std::vector dp(n + 1, std::vector<int>(kMaxState, std::numeric_limits<int>::max()));
        dp[0][0] = 0;

        for (int i = 1; i <= n; ++i)
        {
            for (int s = 0; s < kMaxState; ++s)
            {
                if (__builtin_popcount(s) != i)
                {
                    continue;
                }

                for (int j = 0; j < n; ++j)
                {
                    if (((s >> j) & 1) == 0)
                    {
                        continue;
                    }

                    dp[i][s] = std::min(dp[i][s], dp[i - 1][s ^ (1 << j)] + (nums1[i - 1] ^ nums2[j]));
                }
            }
        }

        return dp[n][kMaxState - 1];
    }
};