class Solution 
{
public:
    int dfs(vector<vector<int>> & matrix, vector<vector<int>> & cache, int i, int j)
    {
        // No need for visited documentation, 
        // as the path is strictly increasing. 
        if (cache[i][j] != 0)
        {
            return cache[i][j];
        }

        for (int di = 0; di != dx.size(); ++di)
        {
            int x = i + dx[di];
            int y = j + dy[di];

            if (0 <= x && x < m && 0 <= y && y < n && matrix[x][y] > matrix[i][j])
            {
                cache[i][j] = max(cache[i][j], dfs(matrix, cache, x, y));
            }
        }

        return ++cache[i][j];
    }

    int longestIncreasingPath(vector<vector<int>> & matrix) 
    {
        m = matrix.size();

        if (m == 0)
        {
            return 0;
        }

        n = matrix[0].size();

        vector<vector<int>> cache;
        cache.resize(m);

        for (auto & col : cache)
        {
            col.resize(n, 0);
        }

        int ans = 0;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                ans = max(ans, dfs(matrix, cache, i, j));
            }
        }

        return ans;
    }

private:
    static constexpr array<int, 4> dx {0, -1,  0, 1};
    static constexpr array<int, 4> dy {1,  0, -1, 0};

    int m;
    int n;
};