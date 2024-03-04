class Solution
{
public:
    std::vector<std::vector<int>> shiftGrid(std::vector<std::vector<int>> & grid, int k)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        std::vector<std::vector<int>> ans(m, std::vector<int>(n));

        for (int col = 0; col < n; ++col)
        {
            int col2 = (col + k) % n;
            int row2 = ((col + k) / n) % m;

            for (int i = 0; i < m; )
            {
                ans[row2++][col2] = grid[i++][col];
                if (row2 == m) row2 = 0;
            }
        }

        return ans;
    }
};