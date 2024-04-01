class Solution
{
public:
    int maximalRectangle(std::vector<std::vector<char>> & matrix)
    {
        auto m = static_cast<const int>(matrix.size());
        auto n = static_cast<const int>(matrix.front().size());

        std::vector<int> h(n, 0);
        int ans = 0;

        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                h[col] = matrix[row][col] == '1' ? h[col] + 1 : 0;
            }
            
            ans = std::max(ans, largestRectangleArea(h));
        }

        return ans;
    }

private:
    static int largestRectangleArea(std::vector<int> & h)
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