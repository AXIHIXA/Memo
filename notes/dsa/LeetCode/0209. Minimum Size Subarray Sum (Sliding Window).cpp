class Solution
{
public:
    int minSubArrayLen(int target, std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        int ans = n + 1;

        for (int ll = 0, rr = 0, sum = 0; rr < n; ++rr)
        {
            sum += nums[rr];

            while (target <= sum)
            {
                ans = std::min(ans, rr - ll + 1);
                sum -= nums[ll];
                ++ll;
            }
        }

        return ans == n + 1 ? 0 : ans;
    }
};