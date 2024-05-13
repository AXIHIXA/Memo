class Solution
{
public:
    bool exist(std::vector<std::vector<char>> & board, std::string word)
    {
        auto m = static_cast<const int>(board.size());
        auto n = static_cast<const int>(board.front().size());
        auto wl = static_cast<const int>(word.size());

        bool res = false;

        for (int x = 0; x < m; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                if (dfs(board, m, n, x, y, word, wl, 0))
                {
                    return true;
                }
            }
        }

        return false;
    }

private:
    static bool dfs(
            std::vector<std::vector<char>> & board, 
            int m, 
            int n, 
            int x, 
            int y, 
            const std::string & word, 
            int wl, 
            int k)
    {
        if (k == wl)
        {
            return true;
        }
        
        if (x < 0 || m <= x || y < 0 || n <= y || board[x][y] != word[k])
        {
            return false;
        }

        bool ret = false;

        char vanilla = board[x][y];
        board[x][y] = ' ';

        for (int d = 0; d < 4; ++d)
        {
            int x1 = x + dx[d];
            int y1 = y + dy[d];
            ret |= dfs(board, m, n, x1, y1, word, wl, k + 1);
        }

        board[x][y] = vanilla;

        return ret;
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};
};