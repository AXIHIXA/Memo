class Solution
{
public:
    std::vector<int> topKFrequent(std::vector<int> & nums, int k)
    {
        // Frequency of each element. 
        std::unordered_map<int, int> frequency;
        for (int x : nums) ++frequency[x];
        
        // // OPTION 1: QUICK SELECT. 
        // // Array of unique elements. 
        // auto n = static_cast<const int>(frequency.size());
        // std::vector<int> arr;
        // arr.reserve(n);
        // for (auto [x, f] : frequency) arr.emplace_back(x);
        // 
        // // Quick select. 
        // std::nth_element(
        //         arr.begin(), 
        //         arr.begin() + n - k, 
        //         arr.end(), 
        //         [&frequency](int x, int y)
        //         {
        //             return frequency.at(x) < frequency.at(y);
        //         }
        // );
        // 
        // std::vector<int> ans;
        // ans.reserve(k);
        // for (auto it = arr.begin() + n - k; it != arr.end(); ++it) ans.emplace_back(*it);
        // 
        // return ans;

        // OPTION 2: BUCKET SORT ON FREQUENCY. 
        auto n = static_cast<const int>(nums.size());
        std::vector<std::vector<int>> bucket(n + 1, std::vector<int>());
        for (auto [x, f] : frequency) bucket[f].emplace_back(x);

        std::vector<int> ans;
        ans.reserve(k);

        for (int f = n; 0 < f && 0 < k; --f)
        {
            for (int x : bucket.at(f))
            {
                ans.emplace_back(x);
                --k;
            }
        }

        return ans;
    }
};
