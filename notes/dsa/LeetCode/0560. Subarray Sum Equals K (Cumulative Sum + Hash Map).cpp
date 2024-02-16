class Solution 
{
public:
    int subarraySum(std::vector<int> & nums, int k) 
    {
        auto n = static_cast<const int>(nums.size());

        // {Cumulative sum, #Occurance}. 
        std::unordered_map<int, int> hmp {{0, 1}};

        int ans = 0;

        for (int i = 0, cs = 0; i < n; ++i)
        {
            cs += nums[i];
            
            // Whether the current cumulative sum MINUS 
            // a preceeding cumulative sum EQUALS k. 
            auto it = hmp.find(cs - k);
            if (it != hmp.end()) ans += it->second;

            ++hmp[cs];
        }

        return ans;
    }
};