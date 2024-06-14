class Solution
{
public:
    int minCapability(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());

        if (n == 1)
        {
            return nums.front();
        }

        int ll = *std::min_element(nums.cbegin(), nums.cend());
        int rr = *std::max_element(nums.cbegin(), nums.cend());
        int ans = rr;

        while (ll <= rr)
        {
            int cap = ll + ((rr - ll) >> 1);

            if (k <= robbable(nums, n, cap))
            {
                ans = cap;
                rr = cap - 1;
            }
            else
            {
                ll = cap + 1;
            }
        }

        return ans;
    }

private:
    static int robbable(const std::vector<int> & nums, int n, int cap)
    {
        std::vector<int> dp(n, 0);
        dp[0] = (nums[0] <= cap);
        dp[1] = (nums[0] <= cap) | (nums[1] <= cap);
        int ans = std::max(dp[0], dp[1]);

        for (int i = 2; i < n; ++i)
        {
            dp[i] = std::max(dp[i - 2] + (nums[i] <= cap), dp[i - 1]);
            ans = std::max(ans, dp[i]);
        }

        // std::printf("robbable(cap=%d) = [ ", cap);
        // for (int i = 0; i < n; ++i) 
        //     std::printf("%d ", dp[i]);
        // std::printf("]\n");

        return ans;
    }
};