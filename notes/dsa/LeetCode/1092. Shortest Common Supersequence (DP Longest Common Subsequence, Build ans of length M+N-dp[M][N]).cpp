class Solution
{
public:
    std::string shortestCommonSupersequence(std::string s1, std::string s2)
    {
        auto m = static_cast<const int>(s1.size());
        auto n = static_cast<const int>(s2.size());

        // dp[i][j]: Length of longest common subsequence of s1[0..i) and s2[0..j). 
        std::vector dp(m + 1, std::vector<int>(n + 1, 0));

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (s1[i - 1] == s2[j - 1])
                {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                }
                else
                {
                    dp[i][j] = std::max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }

        int i = m;
        int j = n;
        int k = m + n - dp[m][n] - 1;
        std::string ans(m + n - dp[m][n], '\0');

        while (0 < i && 0 < j)
        {
            if (s1[i - 1] == s2[j - 1])
            {
                ans[k--] = s1[i - 1];
                --i;
                --j;
            }
            else if (dp[i][j - 1] < dp[i - 1][j])
            {
                ans[k--] = s1[i - 1];
                --i;
            }
            else
            {
                ans[k--] = s2[j - 1];
                --j;
            }
        }

        for ( ; 0 < i; --i) ans[k--] = s1[i - 1];
        for ( ; 0 < j; --j) ans[k--] = s2[j - 1];

        return ans;
    }
};