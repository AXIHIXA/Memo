class Solution
{
public:
    double knightProbability(int n, int k, int row, int col)
    {
        std::vector dp(k + 10, std::vector(n + 10, std::vector<double>(n + 10, -1.0)));

        return f(n, k, row, col, dp);
    }

private:
    static double f(
            int n, 
            int k, 
            int x, 
            int y, 
            std::vector<std::vector<std::vector<double>>> & dp
    )
    {
        if (x < 0 || n <= x || y < 0 || n <= y)
        {
            return 0.0;
        }

        if (k == 0)
        {
            return dp[k][x][y] = 1.0;
        }

        if (dp[k][x][y] != -1.0)
        {
            return dp[k][x][y];
        }

        double ans = 0.0;

        for (int d = 0; d < 8; ++d)
        {
            ans += f(n, k - 1, x + dx[d], y + dy[d], dp) / 8.0;
        }

        return dp[k][x][y] = ans;
    }

private:
    static constexpr std::array<int, 8> dx = {2, 1, -1, -2, -2, -1, 1, 2};
    static constexpr std::array<int, 8> dy = {1, 2, 2, 1, -1, -2, -2, -1};
};
