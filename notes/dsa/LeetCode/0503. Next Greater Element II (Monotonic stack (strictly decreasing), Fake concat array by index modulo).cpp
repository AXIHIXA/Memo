class Solution
{
public:
    std::vector<int> nextGreaterElements(std::vector<int> & nums)
    {
        auto n = static_cast<const int>(nums.size());
        const int m = (n << 1);

        // Monotonic stack (strictly decreasing), 
        // Fake concatenated array by modulo. 
        std::vector<int> ans(n, -1);
        std::stack<int> stk;

        for (int i = m - 1; 0 <= i; --i)
        {
            int j = i % n;
            int x = nums[j];

            while (!stk.empty() && stk.top() <= x)
            {
                stk.pop();
            }
            
            if (!stk.empty())
            {
                ans[j] = stk.top();
            }

            stk.emplace(x);
        }

        return ans;
    }
};
