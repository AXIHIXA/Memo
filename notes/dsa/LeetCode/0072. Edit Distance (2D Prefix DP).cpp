class Solution 
{
public:
    Solution()
    {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        std::cout.tie(nullptr);
    }

    int minDistance(string word1, string word2) 
    {   
        if (word1.empty()) return word2.size();
        if (word2.empty()) return word1.size();
        
        // dp[i][j] denotes edit distance between word1[:i] and word2[:j].
        // dp[i][j] == 
        //     if word1[i - 1] == word2[j - 1]: 
        //         dp[i - 1][j - 1]
        //     else: 
        //         min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        //             del word1[i]; del word2[j]; mod word1[i] -> word2[j]
        // std::vector<std::vector<int>> dp(word1.size() + 1, std::vector<int>(word2.size() + 1, 0));

        for (int i = 1; i <= word1.size(); ++i) dp[i][0] = i;
        for (int j = 1; j <= word2.size(); ++j) dp[0][j] = j;

        for (int i = 1; i <= word1.size(); ++i)
        {
            for (int j = 1; j <= word2.size(); ++j)
            {
                if (word1[i - 1] == word2[j - 1])
                {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else
                {
                    dp[i][j] = std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
                }
            }
        }

        return dp[word1.size()][word2.size()];
    }

private:
    static constexpr int kMaxLen {510};
    static int dp[kMaxLen][kMaxLen];
};

int Solution::dp[kMaxLen][kMaxLen] {0};
