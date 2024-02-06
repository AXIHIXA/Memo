static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    int totalNQueens(int n)
    {
        if (n < 1) return 0;

        // n = 5
		// 1 << 5 = 0...100000 - 1
		// limit  = 0...011111; 
		// n = 7
		// limit  = 0...01111111; 
        int limit = (1 << n) - 1;

        return backtrack(limit, 0, 0, 0);
    }

private:
	// limit : 当前是几皇后问题
	// 之前皇后的列影响：col
	// 之前皇后的右上 -> 左下对角线影响：left
	// 之前皇后的左上 -> 右下对角线影响：right
    static int backtrack(int limit, int col, int left, int right)
    {
        if (col == limit) return 1;

		int ans = 0;

		for (int place = 0, ban = col | left | right, candidate = limit & (~ban); candidate != 0; )
        {
			// ban: 总限制
            // candidate: ~ban : 1可放皇后，0不能放
            // place: 放置皇后的尝试
            
			place = candidate & (-candidate);
			candidate ^= place;
			ans += backtrack(limit, col | place, (left | place) >> 1, (right | place) << 1);
		}

		return ans;
    }

    static int totalNQueensLegacy(int n)
    {
        if (n == 1) return 1;
        if (n == 2) return 0;
        std::vector<std::vector<unsigned char>> curr(n, std::vector<unsigned char>(n, false));
        int ans = 0;
        backtrackLegacy(n, 0, curr, ans);
        return ans;
    }

    static void backtrackLegacy(
            int n, 
            int row, 
            std::vector<std::vector<unsigned char>> & curr, 
            int & ans
    )
    {
        if (row == n)
        {
            ++ans;
            return;
        }

        for (int j = 0; j < n; ++j)
        {
            if (validLegacy(n, curr, row, j))
            {
                curr[row][j] = true;
                backtrackLegacy(n, row + 1, curr, ans);
                curr[row][j] = false;
            }
        }
    }

    static bool validLegacy(
            int n, 
            const std::vector<std::vector<unsigned char>> & curr, 
            int r, 
            int c
    )
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
