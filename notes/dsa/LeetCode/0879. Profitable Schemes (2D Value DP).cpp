class Solution
{
public:
    int profitableSchemes(
            int n, 
            int minProfit, 
            std::vector<int> & group, 
            std::vector<int> & profit)
    {
        auto groupLength = static_cast<const int>(group.size());
        std::vector dp(n + 10, std::vector<int>(minProfit + 10, 0));

        // dp[maxGroup][members][profit]. 
        // dp[k][i][j] = dp[k - 1][i][j] + dp[k - 1][i - group[k]][j - profit[k]]. 

        for (int i = 0; i <= n; ++i)
        {
            dp[i][0] = 1;
        }

        for (int k = 0; k < groupLength; ++k)
        {
            for (int i = n; group[k] <= i; --i)
            {
                for (int j = minProfit; 0 <= j; --j)
                {
                    dp[i][j] = (dp[i][j] + dp[i - group[k]][std::max(0, j - profit[k])]) % p;
                }
            }
        }

        return dp[n][minProfit];
    }

private:
    static constexpr int p = 1'000'000'007;
};
