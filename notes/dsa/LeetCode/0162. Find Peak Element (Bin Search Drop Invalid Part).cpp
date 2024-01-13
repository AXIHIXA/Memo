class Solution 
{
public:
    int findPeakElement(vector<int>& nums) 
    {
        if (nums.size() == 1 || nums[1] < nums.front())
        {
            return 0;
        }

        if (*(nums.end() - 2) < nums.back()) 
        {
            return nums.size() - 1;
        }

        int lo = 1, hi = nums.size() - 1, ans = -1;

        while (lo < hi)
        {
            int mi = lo + ((hi - lo) >> 1);

            if (nums[mi] < nums[mi - 1])
            {
                hi = mi;
            }
            else if (nums[mi] < nums[mi + 1])
            {
                lo = mi + 1;
            }
            else
            {
                ans = mi;
                break;
            }
        }

        return ans;
    }
};