class Solution
{
public:
    int strangePrinter(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        // dp[i][j]: 
        // #Operations needed to print eee...ee (j-i chars) 
        // into target string s[i...j), where e == s[j-1]. 
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(n + 1));

        for (int len = 1; len <= n; ++len)
        {
            for (int i = 0, j; i <= n - len; ++i)
            {
                j = i + len;
                dp[i][j] = len;

                for (int k = i + 1; k < j; ++k)
                {
                    dp[i][j] = std::min(dp[i][j], dp[i][k] + dp[k][j]);
                }

                if (1 < len && s[i] == s[j - 1]) --dp[i][j];
            }
        }

        return dp[0][n];
    }
};
