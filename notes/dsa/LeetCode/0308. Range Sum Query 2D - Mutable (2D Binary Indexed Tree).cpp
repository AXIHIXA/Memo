class NumMatrix
{
public:
    NumMatrix(
            std::vector<std::vector<int>> & matrix
    ) :
            m(matrix.size() + 1),
            n(matrix.front().size() + 1),
            orig(matrix)
    {
        tree.assign(m, std::vector<int>(n, 0));

        for (int i = 0; i < m - 1; ++i)
        {
            for (int j = 0; j < n - 1; ++j)
            {
                add(i + 1, j + 1, orig[i][j]);
            }
        }
    }

    void update(int row, int col, int val)
    {
        add(row + 1, col + 1, val - orig[row][col]);
        orig[row][col] = val;
    }

    int sumRegion(int row1, int col1, int row2, int col2)
    {
        return query(row2 + 1, col2 + 1) + query(row1, col1) - query(row2 + 1, col1) - query(row1, col2 + 1);
    }

private:
    int query(int x, int y)
    {
        int ans = 0;

        for (int i = x; i; i -= i & -i)
        {
            for (int j = y; j; j -= j & -j)
            {
                ans += tree[i][j];
            }
        }

        return ans;
    }

    void add(int x, int y, int u)
    {
        for (int i = x; i < m; i += i & -i)
        {
            for (int j = y; j < n; j += j & -j)
            {
                tree[i][j] += u;
            }
        }
    }

    // 2D Binary Indexed Tree
    int m;  // orig.size() + 1
    int n;  // orig.front().size() + 1
    std::vector<std::vector<int>> tree;
    std::vector<std::vector<int>> orig;
};