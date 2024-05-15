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

        for (int r = 0; r <= n; ++r)
        {
            dp[r][0] = 1;
        }

        for (int i = 0; i < groupLength; ++i)
        {
            for (int r = n; group[i] <= r; --r)
            {
                for (int s = minProfit; 0 <= s; --s)
                {
                    int p1 = dp[r][s];
					int p2 = group[i] <= r ? dp[r - group[i]][std::max(0, s - profit[i])] : 0;
					dp[r][s] = (p1 + p2) % p;
                }
            }
        }

        return dp[n][minProfit];
    }

private:
    static constexpr int p = 1'000'000'007;
};