class Solution
{
public:
    int maxSubArray(std::vector<int> & nums)
    {
        // 1. Kadane's Algorithm (A trivial DP routine, O(n)).
        // return maxSubArrayKadaneDp(nums);
        
        // 2. Divide And Conquer (Makes a Segment Tree, O(n log n)).
        return std::get<0>(maxSubArrayDivideAndConquer(nums, 0, nums.size()));
    }

private:
    static constexpr int kIntMin {std::numeric_limits<int>::min()};

    static std::tuple<int, int, int, int> 
    maxSubArrayDivideAndConquer(const std::vector<int> & nums, int left, int right)
    {
        if (right < left + 1) return {kIntMin, kIntMin, kIntMin, kIntMin};
        if (right == left + 1) return {nums[left], nums[left], nums[left], nums[left]};
        
        int mi = left + ((right - left) >> 1U);
        auto [ansLeft, prefixLeft, suffixLeft, sumLeft] = maxSubArrayDivideAndConquer(nums, left, mi);
        auto [ansRight, prefixRight, suffixRight, sumRight] = maxSubArrayDivideAndConquer(nums, mi, right);

        int ans = std::max({ansLeft, ansRight, suffixLeft + prefixRight});

        int prefixSum = 0;
        int bestPrefixSum = kIntMin;

        for (int i = left; i < right; ++i)
        {
            prefixSum += nums[i];
            bestPrefixSum = std::max(bestPrefixSum, prefixSum);
        }

        int suffixSum = 0;
        int bestSuffixSum = kIntMin;

        for (int i = right - 1; left <= i; --i)
        {
            suffixSum += nums[i];
            bestSuffixSum = std::max(bestSuffixSum, suffixSum);
        }

        return {ans, bestPrefixSum, bestSuffixSum, sumLeft + sumRight};
    }

    static int maxSubArrayKadaneDp(std::vector<int> & nums)
    {
        int curSum = nums[0];
        int maxSum = nums[0];

        for (int i = 1; i < nums.size(); ++i)
        {
            curSum = std::max(nums[i], curSum + nums[i]);
            maxSum = std::max(maxSum, curSum);
        }

        return maxSum;
    }
};