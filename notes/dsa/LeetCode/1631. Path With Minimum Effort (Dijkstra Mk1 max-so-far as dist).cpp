class Solution
{
public:
    int minimumEffortPath(std::vector<std::vector<int>> & heights)
    {
        auto m = static_cast<const int>(heights.size());
        auto n = static_cast<const int>(heights.front().size());

        auto cmp = 
        [](const std::tuple<int, int, int> & a, 
           const std::tuple<int, int, int> & b) -> bool
        {
            return std::get<2>(a) > std::get<2>(b);
        };

        std::priority_queue<
                std::tuple<int, int, int>, 
                std::vector<std::tuple<int, int, int>>, 
                decltype(cmp)
        > heap;

        heap.emplace(0, 0, 0);
        std::vector visited(m, std::vector<std::uint8_t>(n, false));
        std::vector dist(m, std::vector<int>(n, std::numeric_limits<int>::max()));
        dist[0][0] = 0;

        while (!heap.empty())
        {
            auto [x0, y0, c0] = heap.top();
            heap.pop();

            if (x0 == m - 1 && y0 == n - 1)
            {
                return c0;
            }

            if (visited[x0][y0])
            {
                continue;
            }

            visited[x0][y0] = true;

            for (int d = 0; d < 4; ++d)
            {
                int x1 = x0 + dx[d];
                int y1 = y0 + dy[d];

                if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1 || visited[x1][y1])
                {
                    continue;
                }

                int c1 = std::max(c0, std::abs(heights[x1][y1] - heights[x0][y0]));

                if (dist[x1][y1] <= c1)
                {
                    continue;
                }

                dist[x1][y1] = c1;
                heap.emplace(x1, y1, c1);
            }
        }

        return -1;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};