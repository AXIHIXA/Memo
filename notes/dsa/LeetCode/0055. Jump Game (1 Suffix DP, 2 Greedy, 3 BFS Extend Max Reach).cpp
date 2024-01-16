class Solution 
{
public:
    bool canJump(vector<int> & nums) 
    {
        int n = nums.size();
        int maxReach = 0;

        for (int i = 0; i < n; i++) 
        {
            if (maxReach < i) return false;
            maxReach = std::max(maxReach, i + nums[i]);
        }

        return true;
    }

private:
    bool canJumpDp(vector<int> & nums) 
    {
        int n = nums.size();

        // dp[i] denotes position i could jump to the end. 
        std::vector<int> dp(n, false);
        dp[n - 1] = true;

        for (int i = n - 2; 0 <= i; --i)
        {
            for (int j = i + 1; j <= std::min(i + nums[i], n - 1); ++j)
            {
                if (dp[j])
                {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[0];
    }

    bool canJumpGreedy(vector<int> & nums) 
    {
        int n = nums.size();
        int last = n - 1;

        for (int i = n - 1; 0 <= i; --i)
        {
            if (last <= i + nums[i])
            {
                last = i;
            }
        }

        return last == 0;
    }
};
