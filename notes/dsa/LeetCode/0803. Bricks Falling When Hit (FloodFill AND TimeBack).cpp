class Solution
{
public:
    std::vector<int> hitBricks(
            std::vector<std::vector<int>> & grid, 
            std::vector<std::vector<int>> & hits)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        auto hs = static_cast<const int>(hits.size());
        std::vector<int> ans(hs, 0);

        if (m == 1)
        {
            return ans;
        }

        for (const auto & h : hits)
        {
            --grid[h[0]][h[1]];
        }

        std::function<int (int, int, int, int)> floodFill = 
        [m, n, &grid, &floodFill](int i, int j, int s, int t) -> int
        {
            grid[i][j] = t;
            int ret = 1;

            for (int d = 0; d < 4; ++d)
            {
                int x = i + dx[d];
                int y = j + dy[d];

                if (x < 0 || m <= x || y < 0 || n <= y || grid[x][y] != s)
                {
                    continue;
                }

                ret += floodFill(x, y, s, t);
            }

            return ret;
        };

        for (int j = 0; j < n; ++j)
        {
            if (grid[0][j] == 1)
            {
                floodFill(0, j, 1, 2);
            }
        }

        // Whether this hit falls bricks. 
        // Falls iff. this hit is on ceil or has stable neighbors. 
        auto worth = [m, n, &grid](int i, int j) -> bool
        {
            if (i == 0)
            {
                return true;
            }
            
            bool hasNeighboringTwos = false;

            for (int d = 0; d < 4; ++d)
            {
                int xx = i + dx[d];
                int yy = j + dy[d];

                if (xx < 0 || m <= xx || yy < 0 || n <= yy)
                {
                    continue;
                }

                if (grid[xx][yy] == 2)
                {
                    hasNeighboringTwos = true;
                    break;
                }
            }

            return hasNeighboringTwos;
        };

        // Time back. 
        for (int hi = hs - 1; 0 <= hi; --hi)
        {
            int x = hits[hi][0];
            int y = hits[hi][1];

            if (++grid[x][y] == 0)
            {
                continue;
            }
            
            if (worth(x, y))
            {
                ans[hi] = floodFill(x, y, 1, 2) - 1;
            }
        }

        return ans;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};