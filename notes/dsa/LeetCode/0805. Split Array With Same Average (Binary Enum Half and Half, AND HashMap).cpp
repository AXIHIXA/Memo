class Solution
{
public:
    bool splitArraySameAverage(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        const int m = n >> 1;

        std::unordered_map<int, std::unordered_set<int>> hashMap;

        // Binary enumeration for the left half array. 
        for (int s = 0; s < (1 << m); ++s)
        {
            // s: Binary enumeration mask. 
            // The i-th bit (right to left) denotes nums[i] is chosen. 
            int cnt = 0;
            int tot = 0;

            for (int i = 0; i < m; ++i)
            {
                if ((s >> i) & 1) 
                {
                    ++cnt;
                    tot += nums[i];
                }
            }

            hashMap[tot].insert(cnt);
        }

        int sum = std::reduce(nums.cbegin(), nums.cend(), 0);

        for (int s = 0; s < (1 << (n - m)); ++s)
        {
            int cnt = 0;
            int tot = 0;

            for (int i = 0; i < m; ++i)
            {
                if ((s >> i) & 1) 
                {
                    ++cnt;
                    tot += nums[i + m];
                }
            }

            // So we have (tot, cnt) in right half. 
            // We want to find its pair in left half s.t. they form a valid split. 
            // If the pair in left half is (x, y), we must have
            // (tot + x) / (cnt + y) == sum / n. 
            // We enum k == cnt + y, then we want to find x == k * sum / n - tot. 

            for (int k = std::max(1, cnt); k < n; ++k)
            {
                if ((k * sum) % n) continue;
                auto it = hashMap.find(k * sum / n - tot);
                if (it == hashMap.end()) continue;
                std::unordered_set<int> & hashSet = it->second;
                auto jt = hashSet.find(k - cnt);
                if (jt == hashSet.end()) continue;
                return true;
            }
        }

        return false;
    }
};