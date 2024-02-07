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

private:
    static constexpr int p = 1000000007;
};