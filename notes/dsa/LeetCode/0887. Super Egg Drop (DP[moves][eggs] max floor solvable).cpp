class Solution
{
public:
    int superEggDrop(int k, int n)
    {
        // dp[i][j]: Max floors solvable with i moves and j eggs. 
        // In an optimal strategy, we drop the egg from floor x, 
        // it either breaks and we can solve dp[i - 1][k - 1] lower floors (< x); 
        // or it doesn't break and we can solve dp[i - 1][k] higher floors (> x). 
        // => dp[i][j] = 1 + dp[i - 1][j] + dp[i - 1][j - 1]. 
        std::vector dp(n + 1, std::vector<int>(k + 1, 0));
        
        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= k; ++j)
            {
                dp[i][j] = 1 + dp[i - 1][j] + dp[i - 1][j - 1];
                if (n <= dp[i][j]) return i;
            }
        }

        return -1;
    }
};