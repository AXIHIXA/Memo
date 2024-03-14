class Solution 
{
public:
    int subarraySum(std::vector<int> & nums, int k)
    {
        auto n = static_cast<const int>(nums.size());

        int ans = 0;
        std::unordered_map<int, int> hashMap;
        hashMap.emplace(0, 1);

        for (int i = 0, prefixSum = 0; i < n; ++i)
        {
            prefixSum += nums[i];
            int t = prefixSum - k;
            ans += hashMap[t];
            ++hashMap[prefixSum];
        }

        return ans;
    }

    int subarraySumVanilla(std::vector<int> & nums, int k) 
    {
        auto n = static_cast<const int>(nums.size());
        std::vector<int> ps(n + 1, 0);
        std::inclusive_scan(nums.cbegin(), nums.cend(), ps.begin() + 1, std::plus<>(), 0);
        
        int ans = 0;
        std::unordered_map<int, int> hashMap;
        hashMap.emplace(0, 1);

        for (int i = 0; i < n; ++i)
        {
            int t = ps[i + 1] - k;
            ans += hashMap[t];
            ++hashMap[ps[i + 1]];
        }

        return ans;
    }
};