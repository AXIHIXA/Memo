class Solution 
{
public:
    bool searchMatrix(vector<vector<int>> & matrix, int target) 
    {
        int r = matrix.size() - 1;
        int c = 0;
        const int kN = matrix[0].size();

        while (0 <= r and c < kN)
        {
            if (matrix[r][c] == target) return true;
            else if (matrix[r][c] < target) ++c;
            else --r;
        }

        return false;
    }
};