class Solution
{
public:
    int minimumObstacles(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        std::deque<std::pair<int, int>> deq;
        deq.emplace_front(0, 0);

        std::vector dist(m, std::vector<int>(n, std::numeric_limits<int>::max()));
        dist[0][0] = 0;

        while (!deq.empty())
        {
            auto [x0, y0] = deq.front();
            deq.pop_front();

            if (x0 == m - 1 && y0 == n - 1)
            {
                return dist[x0][y0];
            }

            for (int d = 0; d < 4; ++d)
            {
                int x1 = x0 + dx[d];
                int y1 = y0 + dy[d];

                if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1 || 
                    dist[x1][y1] <= dist[x0][y0] + grid[x1][y1])
                {
                    continue;
                }

                dist[x1][y1] = dist[x0][y0] + grid[x1][y1];

                if (grid[x1][y1] == 1)
                {
                    deq.emplace_back(x1, y1);
                }
                else
                {
                    deq.emplace_front(x1, y1);
                }
            }
        }

        return -1;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};