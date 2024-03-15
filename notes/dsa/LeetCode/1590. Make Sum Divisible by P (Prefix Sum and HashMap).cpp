class Solution
{
public:
    using PrefixSum = int;
    using RightmostIndex = int;

public:
    int minSubarray(std::vector<int> & nums, int p)
    {
        auto n = static_cast<const int>(nums.size());

        // Find smallest subarray s.t. sum % p == r. 
        int r = std::reduce(nums.cbegin(), nums.cend(), 0, [p](int a, int b) { return (a + b) % p; });

        if (r == 0)
        {
            return 0;
        }

        int ans = n;
        std::unordered_map<PrefixSum, RightmostIndex> hashMap;
        hashMap.emplace(0, -1);

        for (int i = 0, ps = 0; i < n; ++i)
        {
            ps = (ps + nums[i]) % p;
            
            // We want to find ? s.t. (ps - ?) % p == r, where 0 <= ps, ? < p. 
            // ==> ps - ? == k * p + r
            // ==> ? == ps - k * p - r == (ps - r + p) % p. 
            int t = (ps - r + p) % p;
            auto it = hashMap.find(t);

            if (it != hashMap.end())
            {
                ans = std::min(ans, i - it->second);
            }

            hashMap[ps] = i;
        }

        return ans == n ? -1 : ans;
    }
};