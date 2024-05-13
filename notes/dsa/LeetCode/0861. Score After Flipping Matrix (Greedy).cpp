class Solution
{
public:
    int matrixScore(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        // Greedy. 
        // Make Column 0 all 1s, 
        // and for all succeeding columns, 
        // as much 1s as possible by column flippings. 
        for (int i = 0; i < m; ++i)
        {
            if (grid[i][0] == 0)
            {
                for (int j = 0; j < n; ++j)
                {
                    grid[i][j] ^= 1;
                }
            }
        }

        int ans = m * (1 << (n - 1));

        for (int j = 1; j < n; ++j)
        {
            int ones = 0;

            for (int i = 0; i < m; ++i)
            {
                ones += grid[i][j];
            }

            ans += std::max(ones, m - ones) * (1 << (n - j - 1));
        }

        return ans;
    }
};