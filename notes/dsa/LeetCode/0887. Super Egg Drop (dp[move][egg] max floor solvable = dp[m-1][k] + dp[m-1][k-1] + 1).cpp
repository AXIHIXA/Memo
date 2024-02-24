class Solution
{
public:
    int superEggDrop(int k, int n)
    {
        // dp[m][k] means that, given k eggs and m moves,
        // what is the maximum number of floor that we can check.

        // The dp equation is:
        // dp[m][k] = dp[m - 1][k - 1] + dp[m - 1][k] + 1,
        // which means we take 1 move to a floor,
        // if egg breaks, then we can check dp[m - 1][k - 1] floors.
        // if egg doesn't break, then we can check dp[m - 1][k] floors.

        // This one move will separate the floors into two non-overlapping groups, 
        // below or above (the current level we choose to drop the egg);
        // so no matter what happened to the egg, we only need to check one of those two group. 
        // If we need to check the level below the current level, then it means the egg is break, 
        // so the maximum level we are able to check is dp[m - 1][k - 1]. 
        // Otherwise if we need to check the level above or equal o the current level, 
        // it means the egg is not break, so the maximum level we can check is dp[m - 1][k], 
        // we should only return 1 + dp[m - 1][k - 1] + dp[m - 1][k] (including the current floor); 
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