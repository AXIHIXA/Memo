class Solution
{
public:
    int findSubstringInWraproundString(std::string s)
    {
        // Char-based DP instead of index-based DP. 
        // dp[i]: #Substrs ending with char 'a' + i. 
        // How to de-duplicate: 
        // (1) Find longest valid substr in s. 
        //     The longest will cover any shorter substr chunks!
        // (2) Any two chunks of valid substrs will NOT overlap. 
        std::vector<int> dp(26, 0);
        dp[s[0] - 'a'] = 1;

        auto n = static_cast<const int>(s.size());

        for (int i = 1, len = 1; i < n; ++i)
        {
			if ((s[i - 1] == 'z' && s[i] == 'a') || s[i - 1] + 1 == s[i])
            {
                ++len;
            }
            else
            {
                len = 1;
            }

            dp[s[i] - 'a'] = std::max(dp[s[i] - 'a'], len);
		}

        return std::reduce(dp.cbegin(), dp.cend(), 0);
    }
};