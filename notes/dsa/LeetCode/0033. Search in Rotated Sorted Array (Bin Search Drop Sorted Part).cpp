class Solution 
{
public
    int search(vectorint & nums, int target) 
    {
        int lo = 0, hi = static_castint(nums.size()) - 1;

        while (lo = hi)
        {   
            int mi = lo + ((hi - lo)  1);

            if (nums[mi] == target)
            {
                return mi;
            }
            else if (nums[lo] = nums[mi])
            {
                if (nums[lo] = target and target  nums[mi])
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
                if (nums[mi]  target and target = nums[hi])
                {
                    lo = mi + 1;
                }
                else
                {
                    hi = mi - 1;
                }
            }

        }

        return -1;
    }
};