class Solution
{
public:
    void solve(std::vector<std::vector<char>> & board)
    {
        auto m = static_cast<const int>(board.size());
        auto n = static_cast<const int>(board.front().size());

        std::function<void (int, int, char, char)> floodFill = 
        [m, n, &board, &floodFill](int i, int j, char source, char target)
        {
            board[i][j] = target;
            
            for (int d = 0; d < 4; ++d)
            {
                int x = i + dx[d];
                int y = j + dy[d];

                if (x < 0 || m <= x || y < 0 || n <= y || board[x][y] != source)
                {
                    continue;
                }
                
                floodFill(x, y, source, target);
            }
        };

        for (int x = 0; x < m; ++x)
        {
            if (board[x][0] == 'O')
            {
                floodFill(x, 0, 'O', ' ');
            }

            if (board[x][n - 1] == 'O')
            {
                floodFill(x, n - 1, 'O', ' ');
            }
        }

        for (int y = 0; y < n; ++y)
        {
            if (board[0][y] == 'O')
            {
                floodFill(0, y, 'O', ' ');
            }

            if (board[m - 1][y] == 'O')
            {
                floodFill(m - 1, y, 'O', ' ');
            }
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (board[i][j] == 'O')
                {
                    board[i][j] = 'X';
                }

                if (board[i][j] == ' ')
                {
                    board[i][j] = 'O';
                }
            }
        }
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};