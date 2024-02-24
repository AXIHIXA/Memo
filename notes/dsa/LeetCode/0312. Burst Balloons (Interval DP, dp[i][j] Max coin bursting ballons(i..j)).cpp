class Solution
{
public:
    int maxCoins(std::vector<int> & nums)
    {
        nums.emplace(nums.begin(), 1);
        nums.emplace_back(1);
        auto n = static_cast<const int>(nums.size());

        // dp[i][j]: Max coins obtainable by bursting ballons(i..j). 
        // dp[i][j] = max { dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j] for i < k < j }
        // Enumerate interveral length and left node. 
        std::vector dp(n, std::vector<int>(n, 0));

        for (int m = 3; m <= n; ++m)
        {
            for (int i = 0; i + m - 1 < n; ++i)
            {
                int j = i + m - 1;

                for (int k = i + 1; k < j; ++k)
                {
                    dp[i][j] = std::max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]);
                }
            }
        }

        return dp[0][n - 1];
    }
};