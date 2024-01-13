class Solution 
{
public:
    bool search(vector<int> & nums, int target) 
    {
        int lo = 0, hi = static_cast<int>(nums.size() - 1);

        while (lo <= hi)
        {
            int mi = lo + ((hi - lo + 1) >> 1);

            if (nums[mi] == target)
            {
                return true;
            }

            if (nums[lo] == nums[mi] and nums[mi] == nums[hi])
            {
                ++lo;
                --hi;
            }
            else if (nums[lo] <= nums[mi])
            {
                if (nums[lo] <= target and target < nums[mi])
                {
                    hi = mi - 1;
                }
                else
                {
                    lo = mi + 1;
                }
            }
            else
            {
                if (nums[mi] < target and target <= nums[hi])
                {
                    lo = mi + 1;
                }
                else 
                {
                    hi = mi - 1;
                }
            }
        }

        return false;
    }
};