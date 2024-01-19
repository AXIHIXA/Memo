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
    int threeSumSmaller(vector<int> & nums, int target)
    {
        int n = nums.size();
        if (n < 3) return 0;

        int ans = 0;
        std::sort(nums.begin(), nums.end());

        for (int i = 0; i < n - 2 && nums[i] + nums[i + 1] + nums[i + 2] < target; ++i)
        {
            int lo = i + 1, hi = n - 1;

            while (lo < hi)
            {
                int sum = nums[i] + nums[lo] + nums[hi];

                if (sum < target)
                {
                    // Note: 
                    // NOT ++ans, as there are multiple valid candidates. 
                    ans += hi - lo;
                    ++lo;
                }
                else
                {
                    --hi;
                }
            }
        }

        return ans;
    }
};