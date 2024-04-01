class Solution
{
public:
    int maxWidthRamp(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        
        std::vector<int> stk;
        stk.reserve(n);

        int ans = 0;

        for (int i = 0; i < n; ++i)
        {
            // All possible starting points of ramps. 
            // If a[i] <= a[j], i < j, 
            // then all ramps starting at j could be extended to i, with wider width. 
            if (stk.empty() || nums[i] < nums[stk.back()])
            {
                stk.emplace_back(i);
            }
        }

        for (int j = n - 1; 0 <= j; --j)
        {
            while (!stk.empty() && nums[stk.back()] <= nums[j])
            {
                int i = stk.back();
                stk.pop_back();
                ans = std::max(ans, j - i);
            }
        }

        return ans;
    }
};