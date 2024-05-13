class Trie
{
public:
    friend class Solution;

    Trie() = default;

    void clear()
    {
        cnt = 1;
        
        for (auto & node : tree)
        {
            std::fill(node.begin(), node.end(), 0);
        }
        
        std::fill(endd.begin(), endd.end(), 0);
    }

    void insert(const std::string & word)
    {
        int cur = 1;

        for (int i = 0, path; i < static_cast<int>(word.size()); ++i)
        {
            path = word[i] - 'a';

            if (tree[cur][path] == 0)
            {
                tree[cur][path] = ++cnt;
            }

            cur = tree[cur][path];
        }

        ++endd[cur];
    }

    int next(char c, int cur = 1)
    {
        return tree[cur][c - 'a'];
    }

    int count(const std::string & word)
    {
        int cur = 1;

        for (int i = 0, path; i < static_cast<int>(word.size()); ++i)
        {
            path = word[i] - 'a';
            cur = tree[cur][path];

            if (cur == 0)
            {
                return 0;
            }
        }

        return endd[cur];
    }

private:
    static constexpr int kMaxSize = 50'003;

    static int cnt;
    static std::array<std::array<int, 27>, kMaxSize> tree;
    static std::array<int, kMaxSize> endd;
};

int Trie::cnt = 0;
std::array<std::array<int, 27>, Trie::kMaxSize> Trie::tree = {};
std::array<int, Trie::kMaxSize> Trie::endd = {};

class Solution
{
public:
    std::vector<std::string> findWords(
            std::vector<std::vector<char>> & board, 
            std::vector<std::string> & words)
    {
        trie.clear();
        
        for (const auto & word : words)
        {
            trie.insert(word);
        }

        auto m = static_cast<const int>(board.size());
        auto n = static_cast<const int>(board.front().size());

        std::unordered_set<std::string> ans;
        std::string tmp;
        tmp.reserve(10);

        for (int x = 0; x < m; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                dfs(board, m, n, x, y, 1, tmp, ans);
            }
        }

        return {ans.cbegin(), ans.cend()};
    }

private:
    static void dfs(
            std::vector<std::vector<char>> & board,
            int m, 
            int n, 
            int x, 
            int y, 
            int cur, 
            std::string & tmp, 
            std::unordered_set<std::string> & ans)
    {
        if (x < 0 || m <= x || y < 0 || n <= y)
        {
            return;
        }

        cur = trie.next(board[x][y], cur);

        if (cur == 0)
        {
            return;
        }

        tmp += board[x][y];
        char vanilla = board[x][y];
        board[x][y] = 'z' + 1;

        if (trie.endd[cur])
        {
            ans.insert(tmp);
        }

        dfs(board, m, n, x + 1, y, cur, tmp, ans);
        dfs(board, m, n, x - 1, y, cur, tmp, ans);
        dfs(board, m, n, x, y + 1, cur, tmp, ans);
        dfs(board, m, n, x, y - 1, cur, tmp, ans);

        tmp.pop_back();
        board[x][y] = vanilla;
    }

private:
    static Trie trie;
};

Trie Solution::trie;
