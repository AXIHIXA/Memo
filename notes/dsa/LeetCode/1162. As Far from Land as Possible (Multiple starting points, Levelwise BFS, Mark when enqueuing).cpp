class Solution
{
public:
    struct P
    {
        int x;
        int y;
    };

    int maxDistance(std::vector<std::vector<int>> & grid)
    {
        auto n = static_cast<const int>(grid.size());
        auto m = static_cast<const int>(grid.front().size());

        std::queue<P> que;
        std::vector visited(n, std::vector<unsigned char>(m, false));
        int waterCells = 0;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                if (grid[i][j] == 1)
                {
                    que.emplace(i, j);
                    visited[i][j] = true;
                }
                else
                {
                    ++waterCells;
                }
            }
        }

        if (waterCells == 0 || waterCells == n * m)
        {
            return -1;
        }

        int levels = 0;

        while (!que.empty())
        {
            ++levels;
            auto levelSize = static_cast<const int>(que.size());

            for (int i = 0; i < levelSize; ++i)
            {
                auto [x0, y0] = que.front();
                que.pop();
                
                for (int d = 0; d < 4; ++d)
                {
                    int x1 = x0 + dx[d];
                    int y1 = y0 + dy[d];

                    if (x1 < 0 || n <= x1 || y1 < 0 || m <= y1 || visited[x1][y1])
                    {
                        continue;
                    }

                    visited[x1][y1] = true;
                    que.emplace(x1, y1);
                }
            }
        }

        return levels - 1;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};