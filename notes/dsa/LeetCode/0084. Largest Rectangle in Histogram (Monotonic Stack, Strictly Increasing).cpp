class Solution
{
public:
    int largestRectangleArea(std::vector<int> & h)
    {
        h.emplace_back(0);
        auto n = static_cast<const int>(h.size());

        // Keep stk strictly monotonically increasing. 
        // When a sequential value == h[stk.top()] arrives, 
        // it updates stack top with new index. 
        std::stack<int> stk;
        stk.emplace(-1);

        int ans = 0;

        for (int i = 0; i < n; ++i)
        {
            while (stk.top() != -1 && h[i] <= h[stk.top()])
            {
                // Note h[stk.top()] == h[i]
                // happens iff. stk.top() + 1 == i. 
                // In this case the update on ans makes no effect;
                // this i simply updates the stack top
                // without touching interior of the stack. 
                int currHeight = h[stk.top()];
                stk.pop();
                int currWidth = i - (stk.top() + 1);
                ans = std::max(ans, currHeight * currWidth);
            }

            // At this point, either stk.top() == -1 or h[stk.top()] < h[i]. 
            stk.emplace(i);
        }

        return ans;
    }
};
