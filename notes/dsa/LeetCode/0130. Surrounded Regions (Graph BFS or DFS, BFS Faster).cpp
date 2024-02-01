class Solution
{
public:
    void solve(std::vector<std::vector<char>> & board)
    {
        int m = board.size();
        int n = board.front().size();

        std::vector<std::vector<bool>> knownAlive(m, std::vector<bool>(n, false));

        for (int i = 0; i < m; ++i)
        {
            if (board[i][0] == 'O') search(board, m, n, i, 0, knownAlive);
            if (board[i][n - 1] == 'O') search(board, m, n, i, n - 1, knownAlive);
        }

        for (int j = 0; j < n; ++j)
        {
            if (board[0][j] == 'O') search(board, m, n, 0, j, knownAlive);
            if (board[m - 1][j] == 'O') search(board, m, n, m - 1, j, knownAlive);
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (board[i][j] == 'O' && !knownAlive[i][j])
                {
                    board[i][j] = 'X';
                }
            }
        }
    }

private:
    static void bfs(
            const std::vector<std::vector<char>> & board, 
            int m, 
            int n, 
            int i, 
            int j, 
            std::vector<std::vector<bool>> & knownAlive 
    )
    {
        std::queue<std::pair<int, int>> qu;
        qu.emplace(i, j);
        
        while (!qu.empty())
        {
            auto [x, y] = qu.front();
            qu.pop();

            if (x < 0 || m <= x || y < 0 || n <= y || board[x][y] == 'X' || knownAlive[x][y])
            {
                continue;
            }

            knownAlive[x][y] = true;

            qu.emplace(x - 1, y);
            qu.emplace(x + 1, y);
            qu.emplace(x, y - 1);
            qu.emplace(x, y + 1);
        }
    }

    static void dfs(
            const std::vector<std::vector<char>> & board, 
            int m, 
            int n, 
            int i, 
            int j, 
            std::vector<std::vector<bool>> & knownAlive 
    )
    {
        std::stack<std::pair<int, int>> st;
        st.emplace(i, j);

        while (!st.empty())
        {
            auto [x, y] = st.top();
            st.pop();

            if (x < 0 || m <= x || y < 0 || n <= y || board[x][y] == 'X' || knownAlive[x][y])
            {
                continue;
            }

            knownAlive[x][y] = true;

            st.emplace(x, y + 1);
            st.emplace(x, y - 1);
            st.emplace(x + 1, y);
            st.emplace(x - 1, y);
        }
    }

    static inline void search(
            const std::vector<std::vector<char>> & board, 
            int m, 
            int n, 
            int i, 
            int j, 
            std::vector<std::vector<bool>> & knownAlive 
    )
    {
        return bfs(board, m, n, i, j, knownAlive);
    }
};