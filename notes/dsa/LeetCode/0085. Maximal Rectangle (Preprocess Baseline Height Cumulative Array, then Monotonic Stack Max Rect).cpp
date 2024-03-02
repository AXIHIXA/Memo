class Solution
{
public:
    int maximalRectangle(std::vector<std::vector<char>> & matrix)
    {
        auto m = static_cast<const int>(matrix.size());
        auto n = static_cast<const int>(matrix.front().size());

        // Baseline height. 
        // blh[i][j] == Number of consecutive '1's 
        // starting from matrix[i][j], count upwards. 
        // Column size == n + 1 for monotonic stack usage. See below. 
        std::vector blh(m, std::vector<int>(n + 1, 0));

        for (int j = 0; j < n; ++j)
        {
            blh[0][j] = matrix[0][j] - '0';
        }

        for (int i = 1; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (matrix[i][j] == '1')
                {
                    blh[i][j] = 1 + blh[i - 1][j];
                }
            }
        }

        // The maximal rectangle must align with one baseline. 
        // This turns this problem into 
        // 84. Largest Rectangle in Histogram
        // https://leetcode.com/problems/largest-rectangle-in-histogram/
        // to be solved with monotonic stacks (enumerate baseline ID [0..m). 

        int ans = 0;
        
        for (int i = 0; i < m; ++i)
        {
            std::stack<int> stk;
            stk.emplace(-1);

            // "<=" n for trailing zero padding. 
            for (int j = 0; j <= n; ++j)
            {
                while (stk.top() != -1 && blh[i][j] <= blh[i][stk.top()])
                {
                    int currHeight = blh[i][stk.top()];
                    stk.pop();
                    int currWidth = j - (stk.top() + 1);
                    ans = std::max(ans, currWidth * currHeight);
                }

                stk.emplace(j);
            }
        }

        return ans;
    }
};class Solution
{
public:
    int maximalRectangle(std::vector<std::vector<char>> & matrix)
    {
        auto m = static_cast<const int>(matrix.size());
        auto n = static_cast<const int>(matrix.front().size());

        // Baseline height. 
        // blh[i][j] == Number of consecutive '1's 
        // starting from matrix[i][j], count upwards. 
        // Column size == n + 1 for monotonic stack usage. See below. 
        std::vector blh(m, std::vector<int>(n + 1, 0));

        for (int j = 0; j < n; ++j)
        {
            blh[0][j] = matrix[0][j] - '0';
        }

        for (int i = 1; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (matrix[i][j] == '1')
                {
                    blh[i][j] = 1 + blh[i - 1][j];
                }
            }
        }

        // The maximal rectangle must align with one baseline. 
        // This turns this problem into 
        // 84. Largest Rectangle in Histogram
        // https://leetcode.com/problems/largest-rectangle-in-histogram/
        // to be solved with monotonic stacks (enumerate baseline ID [0..m). 

        int ans = 0;
        
        for (int i = 0; i < m; ++i)
        {
            std::stack<int> stk;
            stk.emplace(-1);

            // "<=" n for trailing zero padding. 
            for (int j = 0; j <= n; ++j)
            {
                while (stk.top() != -1 && blh[i][j] <= blh[i][stk.top()])
                {
                    int currHeight = blh[i][stk.top()];
                    stk.pop();
                    int currWidth = j - (stk.top() + 1);
                    ans = std::max(ans, currWidth * currHeight);
                }

                stk.emplace(j);
            }
        }

        return ans;
    }
};