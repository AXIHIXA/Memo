class Solution 
{
public:
    string longestPalindrome(string s) 
    {
        // dp[i][j] denotes whether s[i:(j+1)] is a palindrome or not. 
        vector<vector<unsigned char>> dp(s.size(), vector<unsigned char>(s.size(), 0));
        pair<int, int> ans {0, 0};

        for (int i = 0; i != s.size(); ++i)
        {
            dp[i][i] = 1;
        }

        for (int i = 0; i < s.size() - 1; ++i)
        {
            if (s[i] == s[i + 1])
            {
                dp[i][i + 1] = 1;
                ans.first = i;
                ans.second = i + 1;
            }
        }

        for (int diff = 2; diff < s.size(); ++diff)
        {
            for (int i = 0; i < s.size() - diff; ++i)
            {
                if (int j = i + diff; s[i] == s[j] and dp[i + 1][j - 1])
                {
                    dp[i][j] = 1;
                    ans.first = i;
                    ans.second = j;
                }
            }
        }

        return s.substr(ans.first, ans.second - ans.first + 1);
    }
};