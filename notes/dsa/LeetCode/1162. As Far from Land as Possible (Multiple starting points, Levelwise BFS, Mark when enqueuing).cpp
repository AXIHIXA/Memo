class Solution
{
public:
    int maxDistance(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        std::queue<std::pair<int, int>> que;
        std::vector visited(m, std::vector<std::uint8_t>(n, false));

        // BFS with multiple starting points. 
        for (int x = 0; x < m; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                if (grid[x][y] == 1)
                {
                    que.emplace(x, y);
                    visited[x][y] = true;
                }
            }
        }

        // No land or water exists. 
        if (que.size() == 0 || que.size() == m * n)
        {
            return -1;
        }

        int level = 0;

        while (!que.empty())
        {
            auto levelSize = static_cast<const int>(que.size());
            ++level;

            for (int i = 0; i < levelSize; ++i)
            {
                auto [x0, y0] = que.front();
                que.pop();

                for (int d = 0; d < 4; ++d)
                {
                    int x1 = x0 + dx[d];
                    int y1 = y0 + dy[d];

                    if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1 || visited[x1][y1])
                    {
                        continue;
                    }

                    visited[x1][y1] = true;
                    que.emplace(x1, y1);
                }
            }
        }

        return level - 1;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};