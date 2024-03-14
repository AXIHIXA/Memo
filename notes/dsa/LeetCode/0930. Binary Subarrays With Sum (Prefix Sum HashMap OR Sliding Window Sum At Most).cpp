class PrefixSum
{
public:
    // Subarray, contiguous chunk, prefix sum. 
    int numSubarraysWithSum(std::vector<int> & nums, int goal)
    {
        auto n = static_cast<const int>(nums.size());
        
        // Prefix sum of nums. ps[k] = sum nums[0...k). 
        std::vector<int> ps(n + 1, 0);
        std::partial_sum(nums.cbegin(), nums.cend(), ps.begin() + 1);

        int ans = 0;
        std::unordered_map<int, int> hashMap;
        hashMap.emplace(0, 1);

        for (int i = 1; i <= n; ++i)
        {
            // ps[i] - ps[?] == goal -> Look for ps[i] - goal. 
            ans += hashMap[ps[i] - goal];
            ++hashMap[ps[i]];
        }

        return ans;
    }
};

class SlidingWindow
{
public:
    // Subarray, contiguous chunk, prefix sum. 
    int numSubarraysWithSum(std::vector<int> & nums, int goal)
    {
        return numSubarraysWithSumAtMost(nums, goal) - numSubarraysWithSumAtMost(nums, goal - 1);
    }

private:
    // Number of subarrays with sum <= goal. 
    static int numSubarraysWithSumAtMost(const std::vector<int> & nums, int goal)
    {
        auto n = static_cast<const int>(nums.size());
        int ans = 0;

        for (int ll = 0, rr = 0, sum = 0; rr < n; ++rr)
        {
            sum += nums[rr];

            while (ll <= rr && goal < sum)
            {
                sum -= nums[ll];
                ++ll;
            }

            ans += rr - ll + 1;
        }
        
        return ans;
    }
};

// using Solution = PrefixSum;
using Solution = SlidingWindow;
