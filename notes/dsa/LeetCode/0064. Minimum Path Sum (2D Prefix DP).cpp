class Dp2d
{
public:
    int minPathSum(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        std::vector dp(m, std::vector<int>(n, 0));
        dp[0][0] = grid[0][0];

        for (int j = 1; j < n; ++j)
        {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }

        for (int i = 1; i < m; ++i)
        {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }

        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                dp[i][j] = grid[i][j] + std::min(dp[i - 1][j], dp[i][j - 1]);
            }
        }

        return dp[m - 1][n - 1];
    }
};

class DpSpaceCompression
{
public:
    int minPathSum(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        std::vector<int> dp(n, std::numeric_limits<int>::max());
        dp[0] = 0;

        for (int i = 0; i < m; ++i)
        {
            dp[0] += grid[i][0];
            
            for (int j = 1; j < n; ++j)
            {
                dp[j] = grid[i][j] + std::min(dp[j - 1], dp[j]);
            }
        }

        return dp[n - 1];
    }
};

// using Solution = Dp2d;
using Solution = DpSpaceCompression;
