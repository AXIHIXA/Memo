class Solution
{
public:
    int minDistance(std::string word1, std::string word2)
    {
        auto m = static_cast<int>(word1.size());
        auto n = static_cast<int>(word2.size());
        if (m == 0) return n;
        if (n == 0) return m;

        // dp[i][j]: Edit distance between word1[:i] and word2[:j]. 
        // After conversion, word1[i - 1] could still present, or get modified. 
        // So does word2[j - 1]. 
        std::vector dp(m + 1, std::vector<int>(n + 1));
        dp[0][0] = 0;
        for (int i = 1; i <= m; ++i) dp[i][0] = i;
        for (int j = 1; j <= n; ++j) dp[0][j] = j;

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                dp[i][j] = std::min({
                        dp[i - 1][j - 1] + !(word1[i - 1] == word2[j - 1]), 
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1
                });
            }
        }

        return dp.back().back();
    }
};