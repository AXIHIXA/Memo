class Solution 
{
public:
    std::string longestPalindrome(std::string s) 
    {
        auto n = static_cast<int>(s.size());
        if (n == 1) return s;

        // dp[i][j]: Whether s[i:j+1] is palindromic. 
        std::vector<std::vector<unsigned char>> dp(n + 1, std::vector<unsigned char>(n + 1, false));
        for (int i = 0; i < n; ++i) dp[i][i] = true;

        int maxLen = 1;
        int maxLenOffset = 0;

        for (int i = 0; i < n - 1; ++i)
        {
            if (s[i] == s[i + 1])
            {
                dp[i][i + 1] = true;
                maxLen = 2;
                maxLenOffset = i;
            }
        }

        for (int di = 2; di < n; ++di)
        {
            for (int i = 0; i + di < n; ++i)
            {
                if ((s[i] == s[i + di]) && dp[i + 1][i + di - 1])
                {
                    dp[i][i + di] = true;
                    maxLen = di + 1;
                    maxLenOffset = i;
                } 
            }
        }

        return s.substr(maxLenOffset, maxLen);
    }
};