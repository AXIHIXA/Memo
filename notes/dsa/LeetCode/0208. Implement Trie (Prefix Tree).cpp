static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class TrieImpl
{
public:
    TrieImpl() = default;

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

    int count(const std::string & word)
    {
        int cur = 1;

        for (int i = 0, path; i < word.size(); ++i)
        {
            path = word[i] - 'a';
            cur = tree[cur][path];
            if (cur == 0) return 0;
        }

        return end[cur];
    }

    int prefixNumber(const std::string & word)
    {
        int cur = 1;

        for (int i = 0, path; i < word.size(); ++i)
        {
            path = word[i] - 'a';
            cur = tree[cur][path];
            if (cur == 0) return 0;
        }

        return pass[cur];
    }

    void erase(const std::string & word)
    {
        if (count(word))
        {
            int cur = 1;
            --pass[cur];

            for (int i = 0, path; i < word.size(); ++i)
            {
                path = word[i] - 'a';
                
                // Needed!
                if (--pass[tree[cur][path]] == 0)
                {
                    tree[cur][path] = 0;
                    return;
                }

                cur = tree[cur][path];
            }
            
            --end[cur];
        }
    }

    void clear()
    {
        for (auto & node : tree)
        {
            std::fill(node.begin(), node.end(), 0);
        }
        
        std::fill(pass.begin(), pass.end(), 0);
        std::fill(end.begin(), end.end(), 0);
    }

private:
    static constexpr int kMaxSize = 50003;

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

class Trie 
{
public:
    Trie() = default;
    
    void insert(std::string word)
    {
        impl.insert(word);
    }
    
    bool search(std::string word)
    {
        return 0 < impl.count(word);
    }
    
    bool startsWith(std::string prefix)
    {
        return 0 < impl.prefixNumber(prefix);
    }

private:
    TrieImpl impl {};
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */