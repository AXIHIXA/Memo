class Solution
{
public:
    int rob(std::vector<int> & nums)
    {
        auto n = static_cast<int>(nums.size());
        if (n == 1) return nums[0];
        if (n == 2) return std::max(nums[0], nums[1]);
        
        // dp[i] denotes max amount of money by robbing houses within [0, i]. 
        std::vector<int> dp(n, 0);
        dp[0] = nums[0];
        dp[1] = std::max(nums[0], nums[1]);

        for (int i = 2; i < n; ++i)
        {
            dp[i] = std::max(nums[i] + dp[i - 2], dp[i - 1]);
        }

        return dp[n - 1];
    }
};