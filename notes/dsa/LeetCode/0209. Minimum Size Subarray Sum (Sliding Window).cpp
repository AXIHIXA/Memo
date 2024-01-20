int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    int minSubArrayLen(int target, vector<int> & nums)
    {
        int n = nums.size(), ans = n + 1;

        for (int rr = 0, ll = 0, sum = 0; rr < n; ++rr)
        {
            sum += nums[rr];

            while (target <= sum) 
            {
                ans = std::min(ans, rr - ll + 1);
                sum -= nums[ll++];
            }
        }

        return ans == n + 1 ? 0 : ans;
    }
};
