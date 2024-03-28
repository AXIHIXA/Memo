class Solution
{
public:
    int maxSubarrayLength(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());

        using Element = int;
        using Frequency = int;
        std::unordered_map<Element, Frequency> freq;

        int ans = 0;

        for (int ll = 0, rr = 0; rr < n; ++rr)
        {
            ++freq[nums[rr]];

            while (ll <= rr && k < freq.at(nums[rr]))
            {
                --freq.at(nums[ll++]);
            }

            ans = std::max(ans, rr - ll + 1);
        }

        return ans;
    }
};