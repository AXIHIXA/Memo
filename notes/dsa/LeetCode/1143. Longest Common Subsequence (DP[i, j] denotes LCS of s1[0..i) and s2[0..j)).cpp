class Solution
{
public:
    int longestCommonSubsequence(std::string text1, std::string text2)
    {
        auto m = static_cast<int>(text1.size());
        auto n = static_cast<int>(text2.size());

        // dp[i][j]: Length of LCS of text1[:i] and text2[:j]. 
        std::array<std::array<int, 1001>, 1001> dp {{0}};
        
        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (text1[i - 1] == text2[j - 1])
                {
                    dp[i][j] = 1 + dp[i - 1][j - 1];
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