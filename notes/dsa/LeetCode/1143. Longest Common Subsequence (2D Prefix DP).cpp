class Solution
{
public:
    int longestCommonSubsequence(std::string text1, std::string text2)
    {
        auto m = static_cast<const int>(text1.size());
        auto n = static_cast<const int>(text2.size());

        // dp[i][j]: Length of LCS of text1[0..i) and text2[0..j). 
        std::vector dp(m + 1, std::vector<int>(n + 1, 0));

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (text1[i - 1] == text2[j - 1])
                {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                else
                {
                    dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[m][n];
    }
};