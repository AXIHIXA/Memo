class Solution
{
public:
    using Row = int;
    using Col = int;
    using State = int;

public:
    int shortestPathAllKeys(std::vector<std::string> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());
        int k = 0;

        for (int x = 0; x < m; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                if ('a' <= grid[x][y] && grid[x][y] <= 'f')
                {
                    ++k;
                }
            }
        }

        const int targetState = (1 << k) - 1;
        std::vector dist(m, std::vector(n, std::vector<int>(1 << k, kIntMax)));
        std::queue<std::tuple<Row, Col, State>> que;

        for (int x = 0; x < m; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                if (grid[x][y] == '@')
                {
                    que.emplace(x, y, 0);
                    dist[x][y][0] = 0;
                }
            }
        }

        while (!que.empty())
        {
            auto [x, y, state] = que.front();
            que.pop();
            int step = dist[x][y][state];

            for (int d = 0; d < 4; ++d)
            {
                int x1 = x + dx[d];
                int y1 = y + dy[d];

                if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1)
                {
                    continue;
                }

                char c = grid[x1][y1];

                if (c == '#' || ('A' <= c && c <= 'F' && ((state >> (c - 'A')) & 1) == 0))
                {
                    continue;
                }

                int newState = state;

                if ('a' <= c && c <= 'f')
                {
                    newState |= (1 << (c - 'a'));
                }

                if (newState == targetState)
                {
                    return step + 1;
                }

                if (dist[x1][y1][newState] <= step + 1)
                {
                    continue;
                }

                dist[x1][y1][newState] = step + 1;
                que.emplace(x1, y1, newState);
            }
        }

        return -1;
    }

private:
    static constexpr int kIntMax = std::numeric_limits<int>::max();
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};