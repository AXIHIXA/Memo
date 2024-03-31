class Solution
{
public:
    long long countSubarrays(std::vector<int> & nums, int minK, int maxK)
    {
        auto n = static_cast<const int>(nums.size());

        long long ans = 0LL;

        for (int rr = 0, mostRecentMin = -1, mostRecentMax = -1, mostRecentInvalid = -1; rr < n; ++rr)
        {
            if (nums[rr] < minK || maxK < nums[rr])
            {
                mostRecentInvalid = rr;
            }

            if (nums[rr] == minK)
            {
                mostRecentMin = rr;
            }

            if (nums[rr] == maxK)
            {
                mostRecentMax = rr;
            }

            // Number of valid subarrays ending at rr. 
            ans += std::max(0, std::min(mostRecentMin, mostRecentMax) - mostRecentInvalid);
        }

        return ans;
    }
};