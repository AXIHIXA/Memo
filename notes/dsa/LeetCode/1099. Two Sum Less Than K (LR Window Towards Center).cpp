class Solution 
{
public:
    int twoSumLessThanK(vector<int> & nums, int k) 
    {
        sort(nums.begin(), nums.end());

        int ans = -1;

        int lo = 0, hi = nums.size() - 1, sum;

        while (lo < hi)
        {
            sum = nums[lo] + nums[hi];

            if (sum < k)
            {
                ans = max(ans, sum);
                ++lo;
            }
            else
            {
                --hi;
            }
        }  

        return ans;
    }
};