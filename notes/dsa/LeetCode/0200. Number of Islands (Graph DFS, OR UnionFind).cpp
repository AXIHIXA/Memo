class Solution 
{
public:
    int numIslands(std::vector<std::vector<char>> & grid) 
    {
        int m = grid.size();
        int n = grid.front().size();
        std::vector<bool> visited(m * n, false);

        int ans = 0;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (grid[i][j] == '1' && !visited[i * n + j])
                {
                    dfs(grid, m, n, i, j, visited);
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
            std::vector<bool> & visited
    )
    {
        if (i < 0 || m <= i || j < 0 || n <= j || visited[i * n + j] || grid[i][j] == '0')
            return;
        
        visited[i * n + j] = true;

        dfs(grid, m, n, i, j - 1, visited);
        dfs(grid, m, n, i, j + 1, visited);
        dfs(grid, m, n, i - 1, j, visited);
        dfs(grid, m, n, i + 1, j, visited);
    }
};
