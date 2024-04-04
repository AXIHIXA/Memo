class Solution
{
public:
    int largestIsland(std::vector<std::vector<int>> & grid)
    {
        auto n = static_cast<const int>(grid.size());

        int id = 2;

        std::function<void (int, int, int)> dfs = 
        [n, &grid, &dfs](int i, int j, int id)
        {
            grid[i][j] = id;

            for (int d = 0; d < 4; ++d)
            {
                int x = i + dx[d];
                int y = j + dy[d];

                if (x < 0 || n <= x || y < 0 || n <= y || grid[x][y] != 1)
                {
                    continue;
                }

                dfs(x, y, id);
            }
        };

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (grid[i][j] == 1)
                {
                    dfs(i, j, id++);
                }
            }
        }

        std::unordered_map<int, int> islandSize;

        int ans = -1;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (2 <= grid[i][j])
                {
                    ans = std::max(ans, ++islandSize[grid[i][j]]);
                }
            }
        }

        std::unordered_set<int> merged;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (grid[i][j] == 0)
                {
                    int effect = 1;
                    
                    for (int d = 0; d < 4; ++d)
                    {
                        int x = i + dx[d];
                        int y = j + dy[d];

                        if (x < 0 || n <= x || y < 0 || n <= y || grid[x][y] < 2)
                        {
                            continue;
                        }

                        if (!merged.contains(grid[x][y]))
                        {
                            effect += islandSize[grid[x][y]];
                            merged.emplace(grid[x][y]);
                        }
                    }

                    ans = std::max(ans, effect);
                    merged.clear();
                }
            }
        }

        return ans;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};