class Solution 
{
public:
    int numIslands(vector<vector<char>> & grid) 
    {
        m = grid.size();
        n = grid[0].size();

        visited = vector<bool>(m * n, false);
        pGrid = &grid;

        int numIslands = 0;

        for (int i = 0; i != m; ++i)
        {
            for (int j = 0; j != n; ++j)
            {
                if (!visited[i * n + j] and grid[i][j] == '1')
                {
                    dfs(i, j);
                    ++numIslands;
                }
            }
        }

        return numIslands;
    }

private:
    void dfs(int i, int j)
    {
        if (not (0 <= i and i < m and 0 <= j and j < n and !visited[i * n + j] and (*pGrid)[i][j] == '1'))
        {
            return;
        }

        visited[i * n + j] = true;
        
        dfs(i - 1, j);
        dfs(i + 1, j);
        dfs(i, j - 1);
        dfs(i, j + 1);
    }

    int m;
    int n;

    vector<bool> visited;
    vector<vector<char>> * pGrid = nullptr;
};
