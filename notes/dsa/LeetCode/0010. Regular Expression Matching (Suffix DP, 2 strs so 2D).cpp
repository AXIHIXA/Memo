class Solution 
{
public:
    bool isMatch(std::string s, std::string p) 
    {
        // If there is no *:
        //     dp[i][j] = match(s[i], p[j]) and match(s[i + 1:], p[j + 1:]).
        // When x* pattern, 
        // we either consume one char in s and keep x*: 
        //     dp[i][j] = either match(s[i], p[j]) and match(s[i + 1:], p[j:]); 
        // or consume the whole x* pattern immediately without taking anything from s: 
        //     dp[i][j] = or match(s[i], p[j + 2:]). 

        // dp[i][j] denotes s[i:] matches p[j:]. 
        // Init: Two empty str matches. 
        auto m = static_cast<const int>(s.size());
        auto n = static_cast<const int>(p.size());
        std::vector dp(m + 1, std::vector<unsigned char>(n + 1, false));
        dp[m][n] = true;

        // NOTE: i must start from m so s{""}-p{".*"} matches could be initialized properly. 
        for (int i = m; 0 <= i; --i)
        {
            for (int j = n - 1; 0 <= j; --j)
            {
                auto frontMatch = i < m && (p[j] == '.' || s[i] == p[j]);

                if (j + 1 < n && p[j + 1] == '*')
                {
                    dp[i][j] = (frontMatch && dp[i + 1][j]) || dp[i][j + 2];
                }
                else
                {
                    dp[i][j] = frontMatch && dp[i + 1][j + 1];
                }
            }
        }

        return dp[0][0];
    }
};