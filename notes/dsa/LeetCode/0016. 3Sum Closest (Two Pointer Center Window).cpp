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
    int threeSumClosest(vector<int> & nums, int target) 
    {
        int n = nums.size();
        int res = nums[0] + nums[1] + nums[2];
        if (n == 3) return res;

        std::sort(nums.begin(), nums.end());

        for (int i = 0; i < n - 2; ++i)
        {
            int lo = i + 1, hi = n - 1;

            // These two ifs are optimizations, optional. 
            if (int lowerBound = nums[i] + nums[lo] + nums[lo + 1]; target <= lowerBound)
                return std::abs(res - target) < std::abs(lowerBound - target) ? 
                       res : lowerBound;
            if (int upperBound = nums[i] + nums[hi - 1] + nums[hi]; upperBound < target)
            {
                res = upperBound;
                continue;
            }

            while (lo < hi)
            {
                int sum = nums[i] + nums[lo] + nums[hi];
                
                if (std::abs(target - sum) < std::abs(target - res))
                    res = sum;

                if (sum < target)      ++lo;
                else if (target < sum) --hi;
                else                   return target;
            }

            // Skip succeeding duplicates after a window shrink. 
            while (0 < i && i < n - 2 && nums[i - 1] == nums[i]) ++i;
        }

        return res;
    }
};