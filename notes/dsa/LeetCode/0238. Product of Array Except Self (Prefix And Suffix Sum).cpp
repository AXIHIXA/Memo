class Solution
{
public:
    vector<int> productExceptSelf(vector<int> & nums) 
    {
        int n = nums.size();
        std::vector<int> ans(n, 1);

        for (int i = 1; i < n; ++i)
        {
            ans[i] = ans[i - 1] * nums[i - 1];
        }

        for (int i = n - 1, r = 1; 0 <= i; --i)
        {
            ans[i] *= r;
            r *= nums[i];
        }

        return ans;
    }
};