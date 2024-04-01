class Solution
{
public:
    int totalSteps(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        
        using Element = int;
        using Turns = int;
        std::vector<std::pair<Element, Turns>> stk;
        stk.reserve(n);

        int ans = 0;

        for (int i = n - 1; 0 <= i; --i)
        {
            int turns = 0;
            
            while (!stk.empty() && stk.back().first < nums[i])
            {
                auto [cur, turns2] = stk.back();
                stk.pop_back();
                turns = std::max(turns + 1, turns2);
            }
            
            ans = std::max(ans, turns);
            stk.emplace_back(nums[i], turns);
        }

        return ans;
    }
};