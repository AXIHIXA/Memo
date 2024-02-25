class Solution
{
public:
    int findGoodStrings(int n, std::string s1, std::string s2, std::string evil)
    {
        std::vector<int> pfx = buildPrefix(evil);
        std::memset(dp, 0, sizeof(int) * 501 * 51 * 2 * 2);
        return dfs(s1, s2, evil, pfx, 0, 0, true, true);
    }

private:
    static constexpr long long p = 1'000'000'007LL;

    static long long dfs(
            const std::string & s1, 
            const std::string & s2, 
            const std::string & evil, 
            const std::vector<int> & prefix, 
            int sp,  // Current good string construction index
            int ep,  // Constructed string's suffix matches evil[0..ep)
            bool boundLeft,  // This index's char should >= s1[sp]
            bool boundRight  // This index's char should <= s2[sp]
    )
    {
        if (ep == evil.size()) return 0LL;
        if (sp == s1.size()) return 1LL;

        if (!dp[sp][ep][boundLeft][boundRight])
        {
            for (char c = (boundLeft ? s1[sp] : 'a'); c <= (boundRight ? s2[sp] : 'z'); ++c)
            {
                auto t = ep;
                while (0 < t && c != evil[t]) t = prefix[t - 1];

                dp[sp][ep][boundLeft][boundRight] = (
                        dp[sp][ep][boundLeft][boundRight] 
                        + 
                        dfs(s1, s2, evil, prefix, 
                            sp + 1, 
                            c == evil[t] ? t + 1 : 0, 
                            boundLeft && (c == s1[sp]), 
                            boundRight && (c == s2[sp]))
                ) % p;
            }
        }

        return dp[sp][ep][boundLeft][boundRight];
    }

    static std::vector<int> buildPrefix(const std::string & s)
    {
        auto m = static_cast<const int>(s.size());
        std::vector<int> dp(m, 0);

        for (int t = 0, j = 1; j < m; ++j)
        {
            while (0 < t && s[t] != s[j]) t = dp[t - 1];
            t = dp[j] = t + (s[t] == s[j]);
        }

        return dp;
    }

    static int dp[501][51][2][2];
};

int Solution::dp[501][51][2][2];
