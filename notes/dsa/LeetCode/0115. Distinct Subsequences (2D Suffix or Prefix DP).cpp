class Solution
{
public:
    int numDistinct(std::string s, std::string t)
    {
        auto m = static_cast<int>(s.size());
        auto n = static_cast<int>(t.size());
        if (m < n) return 0;

        // dp[i][j]: #subsequences in s[i:] matching t[j:]. 
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

        for (int j = 0; j <= n; ++j) dp[m][j] = 0;  // "" match t, no match. 
        for (int i = 0; i <= m; ++i) dp[i][n] = 1;  // s match "", 1 empty substring. 

        for (int i = m - 1; 0 <= i; --i)
        {
            for (int j = n - 1; 0 <= j; --j)
            {
                dp[i][j] = dp[i + 1][j];

                if (s[i] == t[j])
                {
                    dp[i][j] = (dp[i][j] % p + dp[i + 1][j + 1] % p) % p;
                }
            }
        }

        return dp[0][0];
    }

    int numDistinctPrefix(std::string s, std::string t)
    {
        auto m = static_cast<int>(s.size());
        auto n = static_cast<int>(t.size());
        if (m < n) return 0;

        // dp[i][j]: #subsequences in s[..i) matching t[..j). 
        // Special: i == 0, then s == "", could not match anything, dp[0][?] == 0. 
        std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

        // s == ??? has only one trivial subsequence matching t == "". 
        for (int i = 0; i <= m; ++i)
        {
            dp[i][0] = 1;  
        }
        
        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                dp[i][j] = dp[i - 1][j];

                // Don't forget the offset of i, j on s and t!
                // Well if we go suffix DP we don't have to worry about these offsets...
                if (s[i - 1] == t[j - 1])
                {
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - 1]) % p;
                }
            }
        }

        return dp[m][n];
    }

private:
    static constexpr int p = 1'000'000'007;
};