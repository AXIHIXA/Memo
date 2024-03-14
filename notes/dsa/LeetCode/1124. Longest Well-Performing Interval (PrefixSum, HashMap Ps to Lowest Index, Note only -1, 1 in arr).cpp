class Solution
{
public:
    using PrefixSum = int;
    using LowestIndex = int;

public:
    int longestWPI(std::vector<int> & hours)
    {
        auto n = static_cast<const int>(hours.size());

        int ans = 0;
        std::unordered_map<PrefixSum, LowestIndex> hashMap;
        hashMap.emplace(0, -1);

        for (int i = 0, sum = 0; i < n; ++i)
        {
            sum += (8 < hours[i]) ? 1 : -1;
            
            if (0 < sum)
            {
                ans = i + 1;
            }
            else
            {
                auto it = hashMap.find(sum - 1);

                if (it != hashMap.end())
                {
                    ans = std::max(ans, i - it->second);
                }
            }

            hashMap.try_emplace(sum, i);
        }

        return ans;
    }
};