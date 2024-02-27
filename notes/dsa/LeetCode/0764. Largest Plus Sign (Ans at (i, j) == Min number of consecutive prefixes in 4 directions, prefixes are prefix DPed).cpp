class Solution
{
public:
    int orderOfLargestPlusSign(int n, std::vector<std::vector<int>> & mines)
    {
        std::vector grid(n + 10, std::vector<int>(n + 10, 1));
        for (const auto & mine : mines) grid[mine[0] + 1][mine[1] + 1] = 0;

        // Number of consecutive 1-prefixes in each direction. 
        std::vector a(n + 10, std::vector<int>(n + 10, 0));
        std::vector b(n + 10, std::vector<int>(n + 10, 0)); 
        std::vector c(n + 10, std::vector<int>(n + 10, 0)); 
        std::vector d(n + 10, std::vector<int>(n + 10, 0));

        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (grid[i][j] == 1)
                {
                    a[i][j] = a[i - 1][j] + 1;
                    b[i][j] = b[i][j - 1] + 1;
                }
            }
        }

        for (int i = n; 1 <= i; --i)
        {
            for (int j = n; 1 <= j; --j)
            {
                if (grid[i][j] == 1)
                {
                    c[i][j] = c[i + 1][j] + 1;
                    d[i][j] = d[i][j + 1] + 1;
                }
            }
        }

        // Ans at (i, j) == Min number of consecutive prefixes in 4 directions. 
        int ans = 0;

        for (int i = 1; i <= n; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                ans = std::max(ans, std::min({a[i][j], b[i][j], c[i][j], d[i][j]}));
            }
        }

        return ans;
    }
};