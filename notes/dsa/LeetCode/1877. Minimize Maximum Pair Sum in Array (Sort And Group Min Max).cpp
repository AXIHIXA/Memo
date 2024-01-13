class Solution 
{
public:
    int minPairSum(vector<int> & nums) 
    {
        sort(nums.begin(), nums.end());

        int ans = nums.front() + nums.back();

        for (int i = 1; i < (nums.size() >> 1); ++i)
        {
            ans = max(ans, nums[i] + nums[nums.size() - 1 - i]);
        }

        return ans;
    }
};