class Solution
{
public:
    bool isScramble(std::string s1, std::string s2)
    {
        auto n = static_cast<const int>(s1.size());

        // dp[i][j][len]: 
        // Whether s2.substr(j, len) is a scrambled version of s1.substr(i, len). 
        // 0 <= i, j < n;
        // 0 <= len <= n. 
        std::memset(dp, 0, sizeof(bool) * 30 * 30 * 31);

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                dp[i][j][0] = true;
                dp[i][j][1] = s1[i] == s2[j];
            }
        }

        for (int len = 2; len <= n; ++len)
        {
            for (int i = 0; i + len <= n; ++i)
            {
                for (int j = 0; j + len <= n; ++j)
                {
                    for (int k = 1; k < len; ++k)
                    {
                        dp[i][j][len] |= (dp[i][j][k] && dp[i + k][j + k][len - k]);
                        dp[i][j][len] |= (dp[i][j + len - k][k] && dp[i + k][j][len - k]);
                    }
                }
            }
        }

        return dp[0][0][n];
    }

private:
    static bool dp[30][30][31];
};

bool Solution::dp[30][30][31];
