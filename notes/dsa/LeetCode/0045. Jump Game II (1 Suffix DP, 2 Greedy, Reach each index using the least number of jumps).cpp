class Solution 
{
public:
    int jump(vector<int> & nums) 
    {
        int n = nums.size();
        int ans = 0;
        
        // Greedy. Reach each index using the least number of jumps. 
        // end is the furthest starting index of the current jump.
        // far is the furthest reachable index of the current jump.
        for (int i = 0, far = 0, end = 0; i < n - 1; ++i)
        {
            far = std::max(far, i + nums[i]);

            if (i == end)
            {
                ++ans;
                end = far;
            }
        }
        
        return ans;
    }

private:
    int jumpDp(vector<int> & nums) 
    {
        int n = nums.size();
        std::vector<int> dp(n, std::numeric_limits<int>::max() - 1);
        dp[n - 1] = 0;

        for (int i = n - 2; 0 <= i; --i)
        {
            for (int j = i + 1; j <= std::min(i + nums[i], n - 1); ++j)
            {
                dp[i] = std::min(dp[i], dp[j] + 1);
            }
        }

        return dp[0];
    }
};