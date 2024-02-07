class Solution
{
public:
    int searchInsert(std::vector<int> & nums, int target)
    {
        int lo = 0, hi = nums.size();

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);

            if (target <= nums[mi]) hi = mi;
            else lo = mi + 1;
        }

        return lo;
    }
};