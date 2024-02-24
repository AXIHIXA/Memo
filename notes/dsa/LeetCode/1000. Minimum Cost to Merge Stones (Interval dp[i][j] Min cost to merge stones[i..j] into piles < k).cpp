class Solution
{
public:
    int mergeStones(std::vector<int> & stones, int k)
    {
        auto n = static_cast<const int>(stones.size());
        if ((n - 1) % (k - 1)) return -1;

        // Shifted for interval sum across the whole array without edge cases. 
        // sum stones[i..j] = cm[j + 1] - cm[i], 0 <= i <= j <= n - 1. 
        std::vector<int> cm(n + 1, 0);
        for (int i = 0; i < n; ++i) cm[i + 1] = cm[i] + stones[i];

        // dp[i][j]: 
        //     Min cost to merge stones[i..j] into #piles < k. 
        //     NOTE that #piles is NOT necessarily 1. 
        // dp[i][j] = min { dp[i][mi] + dp[mi + 1][j] for 0 <= mi < j } + 
        //            sum[i..j] if (j - i) % (k - 1) == 0.  
        //            Note: mi += k - 1, either left or right interval
        //            must be one mergeable-to-one-pile!
        std::vector dp(n, std::vector<int>(n, 100'000'000));

        // NOTE: Single piles have ZERO cost!!!
        for (int i = 0; i < n; ++i) dp[i][i] = 0;

        // DP. 
        for (int len = 2; len <= n; ++len)
        {
            for (int i = 0; i + len - 1 < n; ++i)
            {
                int j = i + len - 1;
                
                // NOTE: 
                //     p += k - 1!!!
                //     This is because the dp transfer happens
                //     when there is actually one merge. 
                //     suppose [i, j] == [0, 3] and k == 3. 
                //     ++p gives p == 1 => no transfer could happen for [0, 1] [2, 3]. 
                for (int p = i; p < j; p += k - 1)
                {
                    dp[i][j] = std::min(
                            dp[i][j], 
                            dp[i][p] + dp[p + 1][j]
                    );
                }

                if ((j - i) % (k - 1) == 0) dp[i][j] += cm[j + 1] - cm[i];
            }
        }

        return dp[0][n - 1];
    }
};