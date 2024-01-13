class Solution 
{
public:
    void setZeroes(vector<vector<int>> & matrix) 
    {
        // Mark matrix[i][0] and matrix[0][j] to 0 to indicate matrix[i][j] == 0. 
        // Problem: We dont know whether we should zero row 0 and col 0, 
        // because these are polluted with our flags. 
        // Thus we use additional flags shouldZeroRow0 and shouldZeroCol0. 
        int m = matrix.size();
        int n = matrix[0].size();

        // bool shouldZeroRow0: simply use matrix[0][0]!
        bool shouldZeroCol0 = false;

        for (int i = 0; i != m; ++i)
        {
            if (matrix[i][0] == 0)
            {
                shouldZeroCol0 = true;
            }
            
            for (int j = 1; j != n; ++j)
            {
                if (matrix[i][j] == 0)
                {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }

        for (int i = 1; i != m; ++i)
        {
            for (int j = 1; j != n; ++j)
            {
                if (matrix[i][0] == 0 or matrix[0][j] == 0)
                {
                    matrix[i][j] = 0;
                }
            }
        }

        if (matrix[0][0] == 0)
        {
            for (int j = 0; j != n; ++j)
            {
                matrix[0][j] = 0;
            }
        }

        if (shouldZeroCol0)
        {
            for (int i = 0; i != m; ++i)
            {
                matrix[i][0] = 0;
            }
        }
    }
};