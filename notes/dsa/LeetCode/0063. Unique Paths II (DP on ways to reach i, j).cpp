static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    int uniquePathsWithObstacles(std::vector<std::vector<int>> & obstacleGrid)
    {
        if (obstacleGrid[0][0]) return 0;
        
        int m = obstacleGrid.size();
        int n = obstacleGrid.front().size();

        // dp[i][j]: #ways to reach (i, j) from (0, 0). 
        // Could move towards right or bottom only. 
        std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));

        dp[0][0] = 1;
        for (int i = 1; i < m; ++i) if (!obstacleGrid[i][0]) dp[i][0] = dp[i - 1][0];
        for (int j = 1; j < n; ++j) if (!obstacleGrid[0][j]) dp[0][j] = dp[0][j - 1];

        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                if (obstacleGrid[i][j]) continue;
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }
};