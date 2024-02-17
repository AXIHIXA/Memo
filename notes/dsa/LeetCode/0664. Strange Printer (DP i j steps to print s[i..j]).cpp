class Solution
{
public:
    int strangePrinter(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        // dp[i][j]: 
        // #Operations needed to print s[i...j]. 
        // O(n**3) DP: Traverse length, left node, and mid node. 
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1, 0));

        for (int len = 1; len <= n; ++len)
        {
            for (int i = 0, j; i + len - 1 < n; ++i)
            {
                j = i + len - 1;
                dp[i][j] = 1 + dp[i + 1][j];

                for (int k = i + 1; k <= j; ++k)
                {
                    if (s[i] == s[k])
                    {
                        dp[i][j] = std::min(dp[i][j], dp[i][k - 1] + dp[k + 1][j]);
                    }
                }
            }
        }

        return dp[0][n - 1];
    }
};
