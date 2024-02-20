class Solution
{
public:
    int minimumTimeToInitialState(std::string word, int k)
    {
        auto m = static_cast<const int>(word.size());

        // dp[j]为s[:j+1]的最长公共前后缀的长度
        // dp[j]由前缀DP求得
        // 求解dp[j]时，t为dp[j - 1]
        std::vector<int> dp(m, 0);

        for (int t = 0, j = 1; j < m; ++j)
        {
            while (0 < t && word[t] != word[j]) t = dp[t - 1];
            t = dp[j] = t + (word[t] == word[j]);
        }

        int t = dp.back();
        while (0 < t && 0 < (m - t) % k) t = dp[t - 1];
        
        // upper rounded integer of (n - v) / k
        return (m - t + k - 1) / k;
    }
};