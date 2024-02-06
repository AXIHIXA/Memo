class Solution
{
public:
    bool exist(std::vector<std::vector<char>> & board, std::string word)
    {
        int m = board.size();
        int n = board.front().size();

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                bool flag = false;

                if (board[i][j] == word[0])
                {
                    char vanilla = board[i][j];
                    board[i][j] = ' ';
                    backtrack(word, 1, board, m, n, i, j, flag);
                    board[i][j] = vanilla;
                    if (flag) return true;
                }
            }
        }

        return false;
    }

private:
    static constexpr std::array<std::pair<int, int>, 4> drdc
    { std::make_pair(-1, 0), {1, 0}, {0, -1}, {0, 1} };

    static void backtrack(
            const std::string & word, 
            int i, 
            std::vector<std::vector<char>> & board,
            int m, 
            int n, 
            int r, 
            int c, 
            bool & flag
    )
    {
        if (i == word.size())
        {
            flag = true;
            return;
        }

        for (auto [dr, dc] : drdc)
        {
            int r1 = r + dr;
            int c1 = c + dc;

            if (0 <= r1 && r1 < m && 0 <= c1 && c1 < n && board[r1][c1] == word[i])
            {
                char vanilla = board[r1][c1];
                board[r1][c1] = ' ';
                backtrack(word, i + 1, board, m, n, r1, c1, flag);
                board[r1][c1] = vanilla;
            }
        }
    }
};