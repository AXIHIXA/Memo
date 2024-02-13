class Solution
{
public:
    int minPathSum(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<int>(grid.size());
        auto n = static_cast<int>(grid.front().size());

        std::vector<std::vector<int>> dp(m, std::vector<int>(n));
        dp[0][0] = grid[0][0];
        for (int j = 1; j < n; ++j) dp[0][j] = grid[0][j] + dp[0][j - 1];

        for (int i = 1; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int left = j == 0 ? kBigInt : dp[i][j - 1];
                dp[i][j] = std::min(left, dp[i - 1][j]) + grid[i][j];
            }
        }

        return dp.back().back();
    }

private:
    static constexpr int kBigInt = 100'000;
};