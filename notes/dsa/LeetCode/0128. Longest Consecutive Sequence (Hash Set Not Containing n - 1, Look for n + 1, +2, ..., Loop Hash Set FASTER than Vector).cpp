int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    int longestConsecutive(std::vector<int> & nums)
    {
        if (nums.empty()) return 0;
        
        std::unordered_set<int> hs;
        for (int n : nums) hs.emplace(n);

        int ans = 1;

        for (int n : hs)
        {
            if (hs.find(n - 1) == hs.end())
            {
                int len = 1;

                while (hs.find(n + 1) != hs.end())
                {
                    ++n;
                    ++len;
                }

                ans = std::max(ans, len);
            }
        }

        return ans;
    }
};
