class Solution
{
public:
    int minDistance(std::string s, std::string t)
    {
        auto m = static_cast<const int>(s.size());
        auto n = static_cast<const int>(t.size());
        
        // dp[i][j]: Edit distance from s[0..i) to t[0..j).
        std::vector dp(m + 1, std::vector<int>(n + 1, 0x3f3f3f3f));

        for (int i = 0; i <= m; ++i)
        {
            dp[i][0] = i;
        }

        for (int j = 0; j <= n; ++j)
        {
            dp[0][j] = j;
        }

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});

                if (s[i - 1] == t[j - 1])
                {
                    dp[i][j] = std::min(dp[i][j], dp[i - 1][j - 1]);
                }
            }
        }

        return dp[m][n];
    }
};