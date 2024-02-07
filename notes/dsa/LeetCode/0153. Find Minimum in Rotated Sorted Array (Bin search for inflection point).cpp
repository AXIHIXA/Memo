class Solution
{
public:
    int findMin(std::vector<int> & nums)
    {
        if (nums.size() == 1) return nums[0];
        if (nums.front() < nums.back()) return nums.front();
        
        int lo = 0, hi = nums.size() - 1;
        
        while (lo <= hi)
        {
            int mi = lo + ((hi - lo) >> 1);

            if (nums[mi + 1] < nums[mi]) return nums[mi + 1];
            if (nums[mi] < nums[mi - 1]) return nums[mi];

            if (nums[0] < nums[mi]) lo = mi + 1;
            else hi = mi;
        }

        return nums[lo];
    }
};