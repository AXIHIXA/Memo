class Solution
{
public:
    int longestPalindromeSubseq(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        // dp[i][j]: Length of LPS of s[i...j]. 
        std::vector dp(n, std::vector<int>(n, 0));

        for (int i = 0; i < n; ++i)
        {
            dp[i][i] = 1;
        }

        for (int i = 0; i < n - 1; ++i)
        {
            dp[i][i + 1] = (s[i] == s[i + 1] ? 2 : 1);
        }

        for (int len = 3; len <= n; ++len)
        {
            for (int i = 0; i + len - 1 < n; ++i)
            {
                if (s[i] == s[i + len - 1])
                {
                    dp[i][i + len - 1] = dp[i + 1][i + len - 2] + 2;
                }
                else
                {
                    dp[i][i + len - 1] = std::max(dp[i][i + len - 2], dp[i + 1][i + len - 1]);
                }
            }
        }

        return dp[0][n - 1];
    }
};