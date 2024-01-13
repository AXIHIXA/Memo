class Solution 
{
public:
    bool searchMatrix(vector<vector<int>> & matrix, int target) 
    {
        int m = matrix.size(), n = matrix[0].size();

        auto a = [n, &matrix](int idx) -> int &
        {
            return matrix[idx / n][idx % n];
        };

        for (int lo = 0, hi = m * n; lo < hi; )
        {
            int mi = lo + ((hi - lo) >> 1);
            int v = a(mi);

            if (v == target) return true;
            else if (v < target) lo = mi + 1;
            else hi = mi;
        }

        return false;
    }
};