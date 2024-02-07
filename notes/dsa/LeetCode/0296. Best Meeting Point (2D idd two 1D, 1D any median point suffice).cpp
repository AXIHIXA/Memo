class Solution
{
public:
    int minTotalDistance(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<int>(grid.size());
        auto n = static_cast<int>(grid.front().size());

        // Row, col coordinates of all friends, both in ascending order. 
        std::vector<int> rows;
        std::vector<int> cols;

        for (int r = 0; r < m; ++r)
        {
            for (int c = 0; c < n; ++c)
            {
                if (grid[r][c]) rows.emplace_back(r);
            }
        }

        for (int c = 0; c < n; ++c)
        {
            for (int r = 0; r < m; ++r)
            {
                if (grid[r][c]) cols.emplace_back(c);
            }
        }

        // Min distance of rows is independent to that of cols. 
        // Solve two 1D subproblems independently. 
        // 1D: Any point between the innermost one/two friend(s) is good.

        int minRowDist = std::accumulate(
                rows.cbegin(), rows.cend(), 0, 
                [c = rows[rows.size() >> 1]](int s, int x) { return s + std::abs(x - c); }
        );

        int minColDist = std::accumulate(
                cols.cbegin(), cols.cend(), 0, 
                [c = cols[cols.size() >> 1]](int s, int x) { return s + std::abs(x - c); }
        );
        
        return minRowDist + minColDist;
    }
};