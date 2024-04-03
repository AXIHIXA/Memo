class UnionFind
{
public:
    int numIslands(std::vector<std::vector<char>> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());
        root.resize(m * n);

        auto idx = [n](int i, int j) -> int
        {
            return i * n + j;
        };

        for (int i = 0, k = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j, ++k)
            {
                if (grid[i][j] == '1')
                {
                    root[k] = k;
                    ++sets;
                }
            }
        }

        for (int i = 0, k = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j, ++k)
            {
                if (grid[i][j] == '1')
                {
                    if (0 < i && grid[i - 1][j] == '1')
                    {
                        unite(idx(i - 1, j), k);
                    }

                    if (0 < j && grid[i][j - 1] == '1')
                    {
                        unite(idx(i, j - 1), k);
                    }
                }
            }
        }

        return sets;
    }

private:
    int find(int x)
    {
        if (x == root[x]) return x;
        return root[x] = find(root[x]);
    }

    void unite(int x, int y)
    {
        int rx = find(x), ry = find(y);
        if (rx == ry) return;

        --sets;
        root[rx] = ry;
    }

private:
    int sets = 0;
    std::vector<int> root;
};

class Search
{
public:
    int numIslands(std::vector<std::vector<char>> & grid) 
    {
        int m = grid.size();
        int n = grid.front().size();
        std::vector<std::vector<bool>> visited(m, std::vector<bool>(n, false));

        int ans = 0;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (grid[i][j] == '1' && !visited[i][j])
                {
                    bfs(grid, m, n, i, j, visited);
                    ++ans;
                }
            }
        }

        return ans;
    }

private:
    static void 
    dfs(
            const std::vector<std::vector<char>> & grid, 
            int m, 
            int n, 
            int i, 
            int j, 
            std::vector<std::vector<bool>> & visited
    )
    {
        if (i < 0 || m <= i || j < 0 || n <= j || visited[i][j] || grid[i][j] == '0')
            return;
        
        visited[i][j] = true;

        dfs(grid, m, n, i, j - 1, visited);
        dfs(grid, m, n, i, j + 1, visited);
        dfs(grid, m, n, i - 1, j, visited);
        dfs(grid, m, n, i + 1, j, visited);
    }

    static void
    bfs(
            const std::vector<std::vector<char>> & grid, 
            int m, 
            int n, 
            int i, 
            int j, 
            std::vector<std::vector<bool>> & visited
    )
    {
        std::queue<std::pair<int, int>> qu;
        qu.emplace(i, j);

        while (!qu.empty())
        {
            auto [x, y] = qu.front();
            qu.pop();

            if (x < 0 || m <= x || y < 0 || n <= y || visited[x][y] || grid[x][y] == '0')
                continue;
        
            visited[x][y] = true;

            qu.emplace(x - 1, y);
            qu.emplace(x + 1, y);
            qu.emplace(x, y - 1);
            qu.emplace(x, y + 1);
        }
    }
};

// using Solution = Search;
using Solution = UnionFind;
