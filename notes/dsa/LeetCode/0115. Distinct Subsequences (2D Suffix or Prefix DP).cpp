class Solution
{
public:
    int numDistinct(std::string s, std::string t)
    {
        auto m = static_cast<const int>(s.size());
        auto n = static_cast<const int>(t.size());

        // dp[i][j]: Num distinct subseqs of s[0..i) and t[0..j). 
        std::vector dp(m + 1, std::vector<long long>(n + 1, 0LL));

        for (int i = 0; i < m; ++i)
        {
            dp[i][0] = 1LL;
        }

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                dp[i][j] = dp[i - 1][j];

                if (s[i - 1] == t[j - 1])
                {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % std::numeric_limits<int>::max();
                }
            }
        }

        return static_cast<int>(dp[m][n]);
    }
};