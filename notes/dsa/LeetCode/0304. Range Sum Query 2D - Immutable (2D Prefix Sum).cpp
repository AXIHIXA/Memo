class NumMatrix
{
public:
    NumMatrix(std::vector<std::vector<int>> & matrix)
    {
        auto m = static_cast<const int>(matrix.size());
        auto n = static_cast<const int>(matrix.front().size());
        ps.resize(m + 1, std::vector<int>(n + 1, 0));

        for (int i = 0; i < m; ++i)
        {
            ps[i + 1][0] = ps[i][0] + matrix[i][0];
        }

        for (int j = 0; j < n; ++j)
        {
            ps[0][j + 1] = ps[0][j] + matrix[0][j];
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                ps[i + 1][j + 1] = matrix[i][j] + ps[i + 1][j] + ps[i][j + 1] - ps[i][j];
            }
        }
    }
    
    int sumRegion(int r1, int c1, int r2, int c2)
    {
        return ps[r2 + 1][c2 + 1] - ps[r2 + 1][c1] - ps[r1][c2 + 1] + ps[r1][c1];
    }

private:
    std::vector<std::vector<int>> ps;
};

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix* obj = new NumMatrix(matrix);
 * int param_1 = obj->sumRegion(row1,col1,row2,col2);
 */