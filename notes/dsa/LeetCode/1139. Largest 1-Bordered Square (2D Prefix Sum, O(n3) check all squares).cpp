class Solution
{
public:
    int largest1BorderedSquare(std::vector<std::vector<int>> & grid)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());
        const int dMax = std::min(m, n);
        std::vector ps(m + 1, std::vector<int>(n + 1, 0));
        
        for (int i = 0; i < m; ++i)
        {
            ps[i + 1][0] = ps[i][0] + grid[i][0];
        }

        for (int j = 0; j < n; ++j)
        {
            ps[0][j + 1] = ps[0][j] + grid[0][j];
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                ps[i + 1][j + 1] = grid[i][j] + ps[i + 1][j] + ps[i][j + 1] - ps[i][j];
            }
        }

        if (ps[m][n] == 0)
        {
            return 0;
        }

        auto sum = [&ps](int a, int b, int c, int d) -> int
        {
            return a <= c ? ps[c + 1][d + 1] - ps[c + 1][b] - ps[a][d + 1] + ps[a][b] : 0;
        };

		// 找到的最大合法正方形的边长
		int ans = 1;

		for (int a = 0; a < m; a++)
        {
			for (int b = 0; b < n; b++)
            {
				// (a, b) 所有左上角点
				// (c, d) 更大边长的右下角点，k 是当前尝试的边长
				for (int c = a + ans, d = b + ans, k = ans + 1; c < m && d < n; c++, d++, k++)
                {
                    if (sum(a, b, c, d) - sum(a + 1, b + 1, c - 1, d - 1) == (k - 1) << 2)
                    {
						ans = k;
					}
				}
			}
		}

		return ans * ans;
    }
};