class Solution
{
public:
    std::vector<int> productExceptSelf(std::vector<int> & nums) 
    {
        auto n = static_cast<const int>(nums.size());
        std::vector<int> ans(n, 1);

        // Note the order is init, binOp, which is opposite to std::inclusive_scan...
        std::exclusive_scan(nums.cbegin(), nums.cend(), ans.begin(), 1, std::multiplies<>());
        
        for (int i = n - 2, prod = 1; 0 <= i; --i)
        {
            prod *= nums[i + 1];
            ans[i] *= prod;
        }
        
        return ans;
    }
};