class Solution
{
public:
    int coinChange(std::vector<int> & coins, int amount)
    {
        if (amount == 0) return 0;
        
        // dp[v]: Min #coins needed to make amount v. 
        std::vector<int> dp(amount + 1, amount + 1);
        dp[0] = 0;

        for (int i = 1; i <= amount; ++i)
        {
            for (int c : coins)
            {
                if (c <= i)
                {
                    dp[i] = std::min(dp[i], dp[i - c] + 1);
                }
            }
        }

        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
};