class Solution 
{
public:
    int singleNonDuplicate(vector<int> & nums) 
    {
        int lo = 0, hi = nums.size() - 1;

        while (lo < hi)
        {
            // Bin search on even indices only
            int mi = lo + ((hi - lo) >> 1);
            if (mi & 1) --mi;

            if (nums[mi] == nums[mi + 1]) lo = mi + 2;
            else hi = mi;
        }

        return nums[lo];
    }
};