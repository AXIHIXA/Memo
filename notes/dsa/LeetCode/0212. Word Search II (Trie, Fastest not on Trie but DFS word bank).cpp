static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Trie
{
public:
    Trie() = default;

    void insert(const std::string & word)
    {
        int cur = 1;

        for (int i = 0, path; i < word.size(); ++i)
        {
            path = word[i] - 'a';
            if (tree[cur][path] == 0) tree[cur][path] = ++cnt;
            cur = tree[cur][path];
        }

        ++end[cur];
    }

    int next(char c, int cur = 1) const 
    {
        return tree[cur][c - 'a'];
    }

    void clear()
    {
        for (auto & node : tree)
        {
            std::fill(node.begin(), node.end(), 0);
        }

        std::fill(end.begin(), end.end(), 0);
    }

    static constexpr int kMaxSize = 50003;

    std::array<std::array<int, 26>, kMaxSize> tree;
    std::array<int, kMaxSize> end;

private:
    int cnt = 1;
};


class Solution
{
public:
    std::vector<std::string> findWords(
            std::vector<std::vector<char>> & board, 
            std::vector<std::string> & words
    )
    {
        trie.clear();
        for (const std::string & word : words) trie.insert(word);

        int m = board.size();
        int n = board.front().size();

        std::unordered_set<std::string> ans;
        std::vector visited(m, std::vector<unsigned char>(n, false));

        std::vector<char> word;
        word.reserve(11);

        std::function<void (int, int, int)> dfs = 
        [&](int r, int c, int node)
        {
            if (trie.end[node]) ans.insert(std::string(word.cbegin(), word.cend()));

            for (auto [dr, dc] : dir)
            {
                int rr = r + dr;
                int cc = c + dc;

                if (0 <= rr && rr < m && 0 <= cc && cc < n && !visited[rr][cc])
                {
                    if (int next = trie.next(board[rr][cc], node); next != 0)
                    {
                        visited[rr][cc] = true;
                        word.emplace_back(board[rr][cc]);
                        dfs(rr, cc, next);
                        word.pop_back();
                        visited[rr][cc] = false;
                    }
                }
            }
        };

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int node = trie.next(board[i][j]);
                if (!node) continue;
                word.clear();
                
                visited[i][j] = true;
                word.emplace_back(board[i][j]);
                dfs(i, j, node);
                word.pop_back();
                visited[i][j] = false;
            }
        }

        return std::vector(ans.cbegin(), ans.cend());
    }

private: 
    static constexpr std::array<std::pair<int, int>, 4> dir
    { std::make_pair(-1, 0), {1, 0}, {0, -1}, {0, 1} };

    static Trie trie;
};

Trie Solution::trie {};
