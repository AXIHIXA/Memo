int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution 
{
public:
    void gameOfLife(std::vector<std::vector<int>> & board) 
    {
        int m = board.size(), n = board[0].size();

        for (int i = 0; i != m; ++i)
        {
            for (int j = 0; j != n; ++j)
            {
                int neighborsAlive = 0;
                
                for (int di : dd)
                {
                    for (int dj : dd)
                    {
                        int ii = i + di;
                        int jj = j + dj;

                        if (0 <= ii && ii < m && 0 <= jj && jj < n)
                        {
                            if (ii == i && jj == j) continue;
                            neighborsAlive += board[ii][jj] & k0111;
                        }
                    }
                }
                
                if (board[i][j] & k0111)
                {
                    // Alive
                    if (neighborsAlive < 2 || 3 < neighborsAlive)
                    {
                        board[i][j] |= k1000;
                    }
                }
                else
                {
                    // Dead
                    if (neighborsAlive == 3)
                    {
                        board[i][j] |= k1000;
                    }
                }
            }
        }

        for (int i = 0; i != m; ++i)
        {
            for (int j = 0; j != n; ++j)
            {
                if (board[i][j] & k1000)
                {
                    board[i][j] &= k0111;
                    board[i][j] = !board[i][j];
                }
            }
        }
    }

private:
    static constexpr unsigned k1000 = 0x80000000U;
    static constexpr unsigned k0111 = 0x7FFFFFFFU;
    static constexpr std::array<int, 3> dd = {-1, 0, 1};
};