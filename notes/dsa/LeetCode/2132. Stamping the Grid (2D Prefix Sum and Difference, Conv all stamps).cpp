class Solution
{
public:
    bool possibleToStamp(std::vector<std::vector<int>> & grid, int stampHeight, int stampWidth)
    {
        auto m = static_cast<const int>(grid.size());
        auto n = static_cast<const int>(grid.front().size());

        std::vector ps(m + 1, std::vector<int>(n + 1, 0));
        
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                ps[i + 1][j + 1] = grid[i][j] + ps[i + 1][j] + ps[i][j + 1] - ps[i][j];
            }
        }

        // for (int i = 1; i <= m; ++i)
        // {
        //     for (int j = 1; j <= n; ++j)
        //     {
        //         std::cout << ps[i][j] << ' ';
        //     }

        //     std::cout << '\n';
        // }

        auto sum = [&ps](int a, int b, int c, int d) -> int
        {
            return ps[c + 1][d + 1] - ps[c + 1][b] - ps[a][d + 1] + ps[a][b];
        };

        // for (int i = 0; i + stampHeight - 1 < m; ++i)
        // {
        //     for (int j = 0; j + stampWidth - 1 < n; ++j)
        //     {
        //         std::printf(
        //                 "sum(%d %d %d %d) = %d\n", 
        //                 i, j, i + stampHeight - 1, j + stampWidth - 1, 
        //                 sum(i, j, i + stampHeight - 1, j + stampWidth - 1)
        //         );
        //     }
        // }

        // 2D difference array.
        // Difference arrays are 1-indexed and needs zero padding on both ends.  
        std::vector diff(m + 2, std::vector<int>(n + 2, 0));

        // 1-indexed rectangle top-left (a, b) -> bottom-right (c, d). 
        auto add = [&diff](int a, int b, int c, int d, int k = 1) mutable
        {
            diff[a][b] += k;
            diff[c + 1][b] -= k;
            diff[a][d + 1] -= k;
            diff[c + 1][d + 1] += k;
        };

        for (int i = 0; i + stampHeight - 1 < m; ++i)
        {
            for (int j = 0; j + stampWidth - 1 < n; ++j)
            {
                if (sum(i, j, i + stampHeight - 1, j + stampWidth - 1) == 0)
                {
                    add(i + 1, j + 1, i + stampHeight, j + stampWidth);
                }
            }
        }

        // Prefix-sum differance array into original. 
        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                diff[i][j] += diff[i][j - 1] + diff[i - 1][j] - diff[i - 1][j - 1];
            }
        }

        // for (int i = 1; i <= m; ++i)
        // {
        //     for (int j = 1; j <= n; ++j)
        //     {
        //         std::cout << diff[i][j] << ' ';
        //     }

        //     std::cout << '\n';
        // }

        for (int i = 1; i <= m; ++i)
        {
            for (int j = 1; j <= n; ++j)
            {
                if (grid[i - 1][j - 1] == 0 && diff[i][j] == 0)
                {
                    return false;
                }
            }
        }

        return true;
    }
};