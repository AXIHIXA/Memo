class Solution
{
public:
    std::vector<int> dailyTemperatures(std::vector<int> & temp)
    {
        auto n = static_cast<const int>(temp.size());
        std::vector<int> ans(n, 0);

        // Monotonic stack. 
        // Suppose temp[i] >= temp[j], i < j. 
        // Then day j is NOT the answer for any day <= i. 
        std::vector<int> stk;
        stk.reserve(n);
        stk.emplace_back(n - 1);
        
        for (int i = n - 2; 0 <= i; --i)
        {
            while (!stk.empty() && temp[stk.back()] <= temp[i])
            {
                stk.pop_back();
            }

            if (!stk.empty())
            {
                ans[i] = stk.back() - i;
            }

            stk.emplace_back(i);
        }

        return ans;
    }
};