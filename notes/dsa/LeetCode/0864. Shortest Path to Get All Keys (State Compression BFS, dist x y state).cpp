class Solution
{
public:
    int shortestPathAllKeys(std::vector<std::string> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        int k = 0, si, sj;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (std::islower(grid[i][j]))
                {
                    ++k;
                }

                if (grid[i][j] == '@')
                {
                    si = i;
                    sj = j;
                }
            }
        }

        // Mask: i-th-lowest bit denotes whether key #i is found. 
        const int maxMask = 1 << k;
        const int targetMask = maxMask - 1;

        std::vector dist(m, std::vector(n, std::vector<int>(maxMask, std::numeric_limits<int>::max())));
        dist[si][sj][0] = 0;

        std::queue<std::tuple<int, int, int>> que;
        que.emplace(si, sj, 0);

        while (!que.empty())
        {
            auto [x0, y0, s0] = que.front();
            que.pop();

            if (s0 == targetMask)
            {
                return dist[x0][y0][s0];
            }

            for (int d = 0; d < 4; ++d)
            {
                int x1 = x0 + dx[d];
                int y1 = y0 + dy[d];

                if (x1 < 0 || m <= x1 || y1 < 0 || n <= y1 || grid[x1][y1] == '#')
                {
                    continue;
                }

                if (std::isupper(grid[x1][y1]))
                {
                    int lock = grid[x1][y1] - 'A';

                    if (((s0 >> lock) & 1) == 0)
                    {
                        continue;
                    }
                }

                int s1 = s0;

                if (std::islower(grid[x1][y1]))
                {
                    int key = grid[x1][y1] - 'a';
                    s1 |= 1 << key;
                }
                
                if (dist[x1][y1][s1] <= dist[x0][y0][s0] + 1)
                {
                    continue;
                }

                dist[x1][y1][s1] = dist[x0][y0][s0] + 1;
                que.emplace(x1, y1, s1);
            }
        }

        return -1;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};