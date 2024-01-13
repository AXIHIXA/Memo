class Solution 
{
public:
    void rotate(vector<vector<int>> & matrix) 
    {
        const int kN = matrix.size();
        
        for (int i = 0; i != kN; ++i)
        {
            for (int j = i + 1; j != kN; ++j)
            {
                swap(matrix[i][j], matrix[j][i]);
            }
        }

        for (int i = 0; i != kN; ++i)
        {
            reverse(matrix[i].begin(), matrix[i].end());
        }
    }
};