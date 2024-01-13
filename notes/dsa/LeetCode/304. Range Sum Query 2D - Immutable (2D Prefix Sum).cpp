class NumMatrix 
{
public:
    NumMatrix(vector<vector<int>> & matrix) 
    {
        int nRow = matrix.size(), nCol = matrix[0].size();
        
        sum.assign(nRow + 1, std::vector<int>(nCol + 1, 0));

        for (int i = 0; i != nRow; ++i)
        {
            for (int j = 0; j != nCol; ++j)
            {
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] + matrix[i][j] - sum[i][j];
            }
        }
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) 
    {
        return sum[row2 + 1][col2 + 1] + sum[row1][col1] - sum[row1][col2 + 1] - sum[row2 + 1][col1];
    }

private:
    vector<vector<int>> sum;
};

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix* obj = new NumMatrix(matrix);
 * int param_1 = obj->sumRegion(row1,col1,row2,col2);
 */