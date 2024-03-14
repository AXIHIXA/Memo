class Trie
{
public:
    friend class Solution;

public:
    Trie() = default;

    int next(int cur, char c)
    {
        return tree[cur][c - 'a'];
    }

    void clear()
    {
        cnt = 1;

        for (auto & node : tree)
        {
            std::fill(node.begin(), node.end(), 0);
        }

        std::fill(endd.begin(), endd.end(), 0);
    }

    void insert(const std::string & s)
    {
        int curr = 1;

        for (int i = 0, path; i < s.size(); ++i)
        {
            path = s[i] - 'a';
            if (tree[curr][path] == 0) tree[curr][path] = ++cnt;
            curr = tree[curr][path];
        }

        ++endd[curr];
    }

private:
    static constexpr int kMaxSize = 50'003;

    static int cnt;
    static std::vector<std::array<int, 26>> tree;
    static std::vector<int> endd;
};

int Trie::cnt = 1;
std::vector<std::array<int, 26>> Trie::tree(kMaxSize, {0});
std::vector<int> Trie::endd(kMaxSize, 0);

class Solution
{
public:
    std::vector<std::string> findWords(
            std::vector<std::vector<char>> & board,
            std::vector<std::string> & words)
    {
        auto m = static_cast<const int>(board.size());
        auto n = static_cast<const int>(board.front().size());

        trie.clear();

        for (const std::string & s : words)
        {
            trie.insert(s);
        }

        std::unordered_set<std::string> ans;

        std::string word;
        word.reserve(11);  // 1 <= words[i].length() <= 10

        for (int x = 0; x < m; ++x)
        {
            for (int y = 0; y < n; ++y)
            {
                int succ = trie.next(1, board[x][y]);
                if (succ == 0) continue;

                word += board[x][y];
                char old = board[x][y];
                board[x][y] = '\0';
                
                dfs(board, m, n, ans, x, y, word, succ);

                board[x][y] = old;
                word.pop_back();
            }
        }

        return {ans.cbegin(), ans.cend()};
    }

private:
    void dfs(
            std::vector<std::vector<char>> & board,
            int m,
            int n,
            std::unordered_set<std::string> & ans,
            int x,
            int y,
            std::string & word,
            int curr)
    {
        if (trie.endd[curr])
        {
            ans.insert(word);
        }

        for (int d = 0; d < 4; ++d)
        {
            int x1 = x + dx[d];
            int y1 = y + dy[d];

            if (!(0 <= x1 && x1 < m && 0 <= y1 && y1 < n && board[x1][y1] != '\0'))
            {
                continue;
            }

            int succ = trie.next(curr, board[x1][y1]);

            if (succ == 0)
            {
                continue;
            }

            word += board[x1][y1];
            char old = board[x1][y1];
            board[x1][y1] = '\0';

            dfs(board, m, n, ans, x1, y1, word, succ);

            board[x1][y1] = old;
            word.pop_back();
        }
    }

private:
    static constexpr std::array<int, 4> dx = {1, 0, -1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};

    static Trie trie;
};

Trie Solution::trie;
