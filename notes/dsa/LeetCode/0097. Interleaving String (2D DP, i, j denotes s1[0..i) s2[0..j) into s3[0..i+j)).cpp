class Solution
{
public:
    bool isInterleave(std::string s1, std::string s2, std::string s3)
    {
        auto l1 = static_cast<int>(s1.size());
        auto l2 = static_cast<int>(s2.size());
        auto l3 = static_cast<int>(s3.size());
        if (l1 + l2 != l3) return false;
        if (l1 == 0) return s2 == s3;
        if (l2 == 0) return s1 == s3;

        // dp[i][j]: Whether it's possible to interleave s1[:i] and s2[:j] into s3[:i+j]. 
        std::vector dp(l1 + 1, std::vector<unsigned char>(l2 + 1, false));
        dp[0][0] = true;
        for (int i = 1; i <= l1 && s1[i - 1] == s3[i - 1]; ++i) dp[i][0] = true;
        for (int j = 1; j <= l2 && s2[j - 1] == s3[j - 1]; ++j) dp[0][j] = true;

        for (int i = 1; i <= l1; ++i)
        {
            for (int j = 1; j <= l2; ++j)
            {
                dp[i][j] = 
                        (dp[i - 1][j] && (s1[i - 1] == s3[i + j - 1])) || 
                        (dp[i][j - 1] && (s2[j - 1] == s3[i + j - 1]));
            }
        }

        return dp.back().back();
    }
};