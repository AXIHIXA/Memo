class Solution
{
public:
    int rob(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());

        if (n == 1)
        {
            return nums.front();
        }

        // Regular Robber problem plus one restrction: 
        // Could not simultaneously rob both front and back. 
        return std::max(helper(nums, 0, n - 1), helper(nums, 1, n));
    }

private:
    static int helper(const std::vector<int> & nums, int lo, int hi)
    {
        if (hi <= lo)
        {
            return 0;
        }

        if (lo + 1 == hi)
        {
            return nums[lo];
        }
        
        std::vector<int> dp(hi - lo, 0);
        dp[0] = nums[lo];
        dp[1] = std::max(nums[lo], nums[lo + 1]);
        int ans = dp[1];

        for (int i = lo + 2; i < hi; ++i)
        {
            dp[i - lo] = std::max(dp[i - lo - 2] + nums[i], dp[i - lo - 1]);
            ans = std::max(ans, dp[i - lo]);
        }

        return ans;
    }
};