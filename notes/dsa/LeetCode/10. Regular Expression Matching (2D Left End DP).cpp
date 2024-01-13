class Solution 
{
public:
    bool isMatch(string s, string p) 
    {
        // If there is no *:
        //     dp[i][j] = match(s[i], p[j]) and match(s[i + 1:], p[j + 1:]).
        // When .* pattern, 
        // we either consume one char in string: 
        //     dp[i][j] = either match(s[i], p[j]) and match(s[i + 1:], p[j:]); 
        // or consume the whole .* pattern: 
        //     dp[i][j] = or match(s[i], p[j + 2:]). 

        // dp[i][j] denotes s[i:] matches p[j:]. 
        std::vector<std::vector<unsigned char>> dp;
        dp.assign(s.size() + 1, std::vector<unsigned char>(p.size() + 1, 0));
        dp[s.size()][p.size()] = true;

        for (int i = s.size(); 0 <= i; --i)
        {
            for (int j = p.size() - 1; 0 <= j; --j)
            {
                bool match = i < s.size() and (p[j] == s[i] or p[j] == '.');
                
                if (j + 1 < p.size() and p[j + 1] == '*')
                {
                    dp[i][j] = dp[i][j + 2] or (match and dp[i + 1][j]);
                }
                else
                {
                    dp[i][j] = match and dp[i + 1][j + 1];
                }
            }
        }

        return dp[0][0];
    }
};