class Solution
{
public:
    int maxSubarraySumCircular(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());

        int curMin = 0;
        int sumMin = nums[0];

        int curMax = 0;
        int sumMax = nums[0];

        int sumAll = std::accumulate(nums.cbegin(), nums.cend(), 0);

        for (int x : nums)
        {
            // Normal Kadane's algorithm
            curMax = std::max(curMax, 0) + x;
            sumMax = std::max(sumMax, curMax);

            // Kadane's algorithm but with min to find min subarray
            curMin = std::min(curMin, 0) + x;
            sumMin = std::min(sumMin, curMin);
        }

        return sumMin == sumAll ? sumMax : std::max(sumMax, sumAll - sumMin);
    }
};