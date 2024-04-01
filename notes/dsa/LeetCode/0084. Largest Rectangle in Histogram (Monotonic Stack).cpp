class Solution
{
public:
    int largestRectangleArea(std::vector<int> & h)
    {
        auto n = static_cast<const int>(h.size());
        std::vector<int> stk;
        stk.reserve(n);
        int ans = 0;

        for (int i = 0; i < n; ++i)
        {
            while (!stk.empty() && h[i] <= h[stk.back()])
            {
                int cur = stk.back();
                stk.pop_back();
                int ll = stk.empty() ? -1 : stk.back();
                int rr = i;
                ans = std::max(ans, (rr - ll - 1) * h[cur]);
            }

            stk.emplace_back(i);
        }

        while (!stk.empty())
        {
            int cur = stk.back();
            stk.pop_back();
            int ll = stk.empty() ? -1 : stk.back();
            int rr = n;
            ans = std::max(ans, (rr - ll - 1) * h[cur]);
        }

        return ans;
    }
};
