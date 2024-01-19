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
    vector<vector<int>> fourSum(vector<int> & nums, int target) 
    {
        long long t0 = target;
        std::sort(nums.begin(), nums.end());

        std::vector<std::vector<int>> ans;

        for (int i = 0, n = nums.size(); i < n - 3; ++i)
        {
            for (int j = i + 1; j < n - 2; ++j)
            {
                long long t1 = t0 - nums[i] - nums[j];

                int lo = j + 1, hi = n - 1;

                while (lo < hi)
                {
                    long long su = static_cast<long long>(nums[lo]) + nums[hi];
                    
                    if (su < t1)
                    {
                        ++lo;
                    }
                    else if (t1 < su)
                    {
                        --hi;
                    }
                    else
                    {
                        ans.push_back({nums[i], nums[j], nums[lo], nums[hi]});
                        int ll = lo, rr = hi;
                        while (lo < hi && nums[ll] == nums[lo]) ++lo;
                        while (lo < hi && nums[hi] == nums[rr]) --hi;
                    }

                    while (j < n - 2 && nums[j] == nums[j + 1]) ++j;
                }
            }

            while (i < n - 3 && nums[i] == nums[i + 1]) ++i;
        }

        return ans;
    }
};