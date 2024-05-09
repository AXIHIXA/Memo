class Solution
{
public:
    int numDecodings(std::string s)
    {
        auto n = static_cast<const int>(s.size());

        // dp[i]: #Ways to decode s[i:]. 
        std::vector<int> dp(n + 1, 0);
        dp[n] = 1;
        dp[n - 1] = (s[n - 1] == '0' ? 0 : 1);

        for (int i = n - 2; 0 <= i; --i)
        {
            if (s[i] == '0')
            {
                // dp[i] = 0;  // default in constructor. 
                continue;
            }
            
            dp[i] += dp[i + 1];

            if (10 * (s[i] - '0') + (s[i + 1] - '0') < 27)
            {
                dp[i] += dp[i + 2];
            }
        }

        return dp[0];
    }
};