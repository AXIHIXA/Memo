class Solution
{
public:
    std::vector<int> searchRange(std::vector<int> & nums, int target)
    {
        int lo = lowerBound(nums, target);
        int hi = upperBound(nums, target) - 1;
        return lo < nums.size() && nums[lo] == target && 0 <= hi && nums[hi] == target ? 
               std::vector<int> {lo, hi} : 
               std::vector<int> {-1, -1};
    }

private:
    static int lowerBound(const std::vector<int> & nums, int target)
    {
        int lo = 0, hi = nums.size();

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);
            if (nums[mi] < target) lo = mi + 1;
            else hi = mi; 
        }

        return lo;
    }

    static int upperBound(const std::vector<int> & nums, int target)
    {
        int lo = 0, hi = nums.size();

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);
            if (target < nums[mi]) hi = mi;
            else lo = mi + 1; 
        }

        return lo;
    }
};