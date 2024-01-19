int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    vector<vector<int>> threeSum(vector<int> & nums)
    {
        std::vector<std::vector<int>> ans;
        std::sort(nums.begin(), nums.end());

        for (int i = 0, n = nums.size(); i < n - 2 && nums[i] < 1; ++i)
        {
            if (i == 0 || nums[i - 1] != nums[i])
            {
                twoSum(nums, i, ans);
            }
        }

        return ans;
    }

private:
    static void twoSum(
        const std::vector<int> & nums, 
        int i, 
        std::vector<std::vector<int>> & ans
    )
    {
        int lo = i + 1, hi = nums.size() - 1;

        while (lo < hi)
        {
            int sum = nums[i] + nums[lo] + nums[hi];

            if (sum < 0)
            {
                ++lo;
            }
            else if (0 < sum)
            {
                --hi;
            }
            else 
            {
                ans.push_back({nums[i], nums[lo], nums[hi]});
                int ll = lo, rr = hi;
                while (lo < hi && nums[ll] == nums[lo]) ++lo;
                while (lo < hi && nums[hi] == nums[rr]) --hi;
            }
        }
    }
};
