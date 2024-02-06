static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class WildcardTrie
{
public:
    WildcardTrie() = default;

    void insert(const std::string & word)
    {
        int cur = 1;
        ++pass[cur];

        for (int i = 0, path; i < word.size(); ++i)
        {
            path = word[i] - 'a';
            if (tree[cur][path] == 0) tree[cur][path] = ++cnt;
            cur = tree[cur][path];
            ++pass[cur];
        }

        ++end[cur];
    }

    int find(const std::string & word)
    {
        if (word.empty()) return end[1];
        return count(1, word, word[0], 1);
    }

    int count(int cur, const std::string & word, char c, int i)
    {
        if (c == '.')
        {
            for (c = 'a'; c <= 'z'; ++c)
            {
                int path = c - 'a';
                if (tree[cur][path] == 0) continue;

                if (i < word.size())
                {
                    int ct = count(tree[cur][path], word, word[i], i + 1);
                    if (0 < ct) return ct;
                    continue;
                }
                else
                {
                    if (0 < end[tree[cur][path]]) return end[tree[cur][path]];
                    continue;
                }
            }
        }
        else
        {
            int path = c - 'a';
            cur = tree[cur][path];
            if (cur == 0) return 0;
            return i < word.size() ? count(cur, word, word[i], i + 1) : end[cur];
        }

        return 0;
    }

private:
    // UPDATE with respect to the data constraints!
    static constexpr int kMaxSize = 260003;

    // All chars are stored as edges;
    // each node has its own pass and end variable.
    // pass: How many words pass by this node
    // (i.e., how many times this node serves as a prefix of an inserted word).
    // end: How many words end at this node.
    // tree[0] is reserved for not found results;
    // tree[1] is the root node;
    // tree[2]+ are actual nodes.
    std::array<std::array<int, 26>, kMaxSize> tree {{0}};
    std::array<int, kMaxSize> pass {0};
    std::array<int, kMaxSize> end {0};

    int cnt = 1;
};


class WordDictionary 
{
public:
    WordDictionary() = default;
    
    void addWord(string word) 
    {
        trie.insert(word);
    }
    
    bool search(string word) 
    {
        return trie.find(word);
    }

private:
    WildcardTrie trie;
};

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary* obj = new WordDictionary();
 * obj->addWord(word);
 * bool param_2 = obj->search(word);
 */