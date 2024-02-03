// static const int init = []
// {
//     std::ios_base::sync_with_stdio(false);
//     std::cin.tie(nullptr);
//     std::cout.tie(nullptr);
//     return 0;
// }();

class Solution
{
public:
    int totalNQueens(int n)
    {
        if (n == 1) return 1;
        if (n == 2) return 0;
        std::vector<std::vector<unsigned char>> curr(n, std::vector<unsigned char>(n, false));
        int ans = 0;
        backtrack(n, 0, curr, ans);
        return ans;
    }

private:
    static void backtrack(int n, int row, std::vector<std::vector<unsigned char>> & curr, int & ans)
    {
        if (row == n)
        {
            ++ans;
            return;
        }

        for (int j = 0; j < n; ++j)
        {
            if (valid(n, curr, row, j))
            {
                curr[row][j] = true;
                backtrack(n, row + 1, curr, ans);
                curr[row][j] = false;
            }
        }
    }

    static bool valid(int n, const std::vector<std::vector<unsigned char>> & curr, int r, int c)
    {
        // column
        for (int i = r - 1; 0 <= i; --i)
        {
            if (curr[i][c])
            {
                return false;
            }
        }

        // left diagonal
        for (int i = r - 1, j = c - 1; 0 <= i && 0 <= j; --i, --j)
        {
            if (curr[i][j]) 
            {
                return false;
            }
        }
 
        // right diagonal
        for (int i = r - 1, j = c + 1; 0 <= i && j < n; --i, ++j)
        {
            if (curr[i][j])
            {
                return false;
            }
        }

        return true;
    }
};