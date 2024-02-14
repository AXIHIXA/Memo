#include <algorithm>
#include <array>
#include <iostream>
#include <string>


class Trie
{
public:
    Trie() = default;

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

    int countWildcard(const std::string & word)
    {
        if (word.empty()) return end[1];
        return countWildcardImpl(1, word, word[0], 1);
    }

private:
    // UPDATE with respect to data size!
    static constexpr int kMaxSize = 50003;

    int countWildcardImpl(int cur, const std::string & word, char c, int i)
    {
        if (c == '.')
        {
            for (c = 'a'; c <= 'z'; ++c)
            {
                int path = c - 'a';
                if (tree[cur][path] == 0) continue;

                if (i < word.size())
                {
                    int ct = countWildcardImpl(tree[cur][path], word, word[i], i + 1);
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
            return i < word.size() ? countWildcardImpl(cur, word, word[i], i + 1) : end[cur];
        }

        return 0;
    }

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


int main(int argc, char * argv[])
{
    Trie trie;
    trie.insert("add");
    trie.insert("eat");
    trie.insert("addition");

    std::printf("%d\n", trie.count("add"));
    std::printf("%d\n", trie.count("eat"));
    std::printf("%d\n", trie.count("addition"));
    std::printf("%d\n", trie.count("adc"));
    std::printf("%d\n", trie.count("add"));
    std::printf("%d\n", trie.count("ad."));
    std::printf("%d\n", trie.count(".d."));
    std::printf("%d\n", trie.count("add."));
    std::printf("%d\n", trie.countWildcard("ad."));
    std::printf("%d\n", trie.countWildcard(".d."));
    std::printf("%d\n", trie.countWildcard("add."));

    return EXIT_SUCCESS;
}
