class Solution
{
public:
    int longestPalindromeSubseq(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        // dp[i][j]: Longest Palindrome Subsequence in s[i..j]. 
        std::vector dp(n, std::vector<int>(n, 0));
        for (int i = 0; i < n; ++i) dp[i][i] = 1;

        for (int i = n - 1; 0 <= i; --i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                if (s[i] == s[j]) dp[i][j] = 2 + dp[i + 1][j - 1];
                else dp[i][j] = std::max(dp[i + 1][j], dp[i][j - 1]);
            }
        }

        return dp[0][n - 1];
    }
};