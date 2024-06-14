class Solution
{
public:
    int maxSubarraySumCircular(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());

        // Max Circ Sum == 
        // Max Reg Sum, OR
        // Sum All - Min Reg Sum
        int sumAll = std::reduce(nums.cbegin(), nums.cend(), 0);

        int maxSum = nums[0];
        int curMax = nums[0];
        
        int minSum = nums[0];
        int curMin = nums[0];
        
        for (int i = 1; i < n; ++i)
        {
            maxSum = std::max(maxSum + nums[i], nums[i]);
            curMax = std::max(curMax, maxSum);

            minSum = std::min(minSum + nums[i], nums[i]);
            curMin = std::min(curMin, minSum);
        }

        return curMin == sumAll ? curMax : std::max(curMax, sumAll - curMin);
    }
};