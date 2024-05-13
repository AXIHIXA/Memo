class Solution
{
public:
    bool isInterleave(std::string s1, std::string s2, std::string s3)
    {
        auto m = static_cast<const int>(s1.size());
        auto n = static_cast<const int>(s2.size());
        auto mn = static_cast<const int>(s3.size());

        if (m == 0)
        {
            return s2 == s3;
        }

        if (n == 0)
        {
            return s1 == s3;
        }

        if (m + n != mn)
        {
            return false;
        }

        // dp[i][j]: 
        // Whether it's possible to interleave s1[0..i) and s2[0..j) into s3[0..i+j). 
        std::vector dp(m + 1, std::vector<std::uint8_t>(n + 1, false));

        for (int i = 0; i < m && s1[i] == s3[i]; ++i)
        {
            dp[i + 1][0] = true;
        }

        for (int j = 0; j < n && s2[j] == s3[j]; ++j)
        {
            dp[0][j + 1] = true;
        }

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (s3[i + j - 1] == s1[i - 1])
                {
                    dp[i][j] |= dp[i - 1][j];
                }

                if (s3[i + j - 1] == s2[j - 1])
                {
                    dp[i][j] |= dp[i][j - 1];
                }
            }
        }

        return dp[m][n];
    }
};