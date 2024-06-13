class Solution
{
public:
    int numberOfPaths(std::vector<std::vector<int>> & grid, int k)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        std::vector dp(m + 10, std::vector(n + 10, std::vector<int>(k + 10, -1)));

        return f(m - 1, n - 1, k, 0, m, n, grid, dp);
    }

private:
    static int f(
            int x, 
            int y, 
            int k, 
            int r, 
            int m, 
            int n, 
            const std::vector<std::vector<int>> & grid, 
            std::vector<std::vector<std::vector<int>>> & dp
    )
    {
        if (x < 0 || y < 0)
        {
            return 0;
        }

        if (x == 0 && y == 0)
        {
            return dp[x][y][r] = ((grid[0][0] % k) == r);
        }

        if (dp[x][y][r] != -1)
        {
            return dp[x][y][r];
        }

        int ans = 0;
        int r1 = (r + k - (grid[x][y] % k)) % k;
        ans = (ans + f(x - 1, y, k, r1, m, n, grid, dp)) % p;
        ans = (ans + f(x, y - 1, k, r1, m, n, grid, dp)) % p;
        return dp[x][y][r] = ans;
    }

private:
    static constexpr int p = 1'000'000'007;
};