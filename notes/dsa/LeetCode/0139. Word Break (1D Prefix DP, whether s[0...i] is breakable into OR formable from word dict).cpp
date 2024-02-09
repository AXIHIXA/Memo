class Solution
{
public:
    bool wordBreak(std::string s, std::vector<std::string> & wordDict)
    {
        auto n = static_cast<int>(s.size());

        // dp[i]: Whether s[:i + 1] could break into wordDict. 
        std::vector<unsigned char> dp(n , false);

        for (int i = 0; i < n; ++i)
        {
            for (const std::string & word : wordDict)
            {
                if (i + 1 < word.size()) continue;
                
                if (i == word.size() - 1 || dp[i - word.size()])
                {
                    if (s.substr(i - word.size() + 1, word.size()) == word)
                    {
                        dp[i] = true;
                        break;
                    }
                }
            }
        }

        // Another DP approach:
        // dp[i]: Whether it's possible to form s[:i + 1] with wordDict. 

        return dp.back();
    }
};