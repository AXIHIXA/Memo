class Solution
{
public:
    int subarraysWithKDistinct(std::vector<int> & nums, int k)
    {
        return subarraysWithAtMostKDistinct(nums, k) - subarraysWithAtMostKDistinct(nums, k - 1);
    }

private:
    static int subarraysWithAtMostKDistinct(std::vector<int> & nums, int k)
    {   
        auto n = static_cast<const int>(nums.size());

        std::fill_n(count.begin(), n + 10, 0);
        int ans = 0;

        for (int ll = 0, rr = 0, numDistinct = 0; rr < n; ++rr)
        {
            if (++count[nums[rr]] == 1)
            {
                ++numDistinct;
            }
            
            while (ll <= rr && k < numDistinct)
            {
                if (--count[nums[ll++]] == 0)
                {
                    --numDistinct;
                }
            }

            ans += rr - ll + 1;
        }

        return ans;
    }

private:
    static constexpr int kArraySize = 20'020;
    static std::array<int, kArraySize> count;
};

std::array<int, Solution::kArraySize> Solution::count = {0};
