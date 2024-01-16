class Solution 
{
public:
    int majorityElement(vector<int> & nums) 
    {
        int count = 0;
        int ans = nums[0];

        for (int e : nums)
        {
            if (count == 0) ans = e;
            count += (e == ans) ? 1 : -1;
        }

        return ans;
    }
};