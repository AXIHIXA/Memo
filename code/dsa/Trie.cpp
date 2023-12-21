#include <algorithm>
#include <memory>
#include <iostream>
#include <unordered_set>


class Trie
{
public:
    static constexpr int kNumAlphabet = 26;
    static constexpr char kFirstChar = 'a';

    struct Node
    {
        explicit Node () = default;

        bool isWordEnd {false};
        std::array<std::unique_ptr<Node>, kNumAlphabet> children {nullptr};
    };

    Trie()
    {
        root = std::make_unique<Node>();
    }

    ~Trie() = default;

    void add(const std::string & word)
    {
        Node * p = root.get();

        for (const char c : word)
        {
            if (not p->children[c - kFirstChar])
            {
                p->children[c - kFirstChar] = std::make_unique<Node>();
            }

            p = p->children[c - kFirstChar].get();
        }

        p->isWordEnd = true;
    }

    bool search(const std::string & word)
    {
        return search(word, 0, root.get());
    }

private:
    bool search(const std::string & word, int index, Node * p)
    {
        if (not (p and index < static_cast<int>(word.size())))
        {
            return false;
        }

        char c = word[index];

        if (index == word.size() - 1)
        {
            return c == '.' ?
                   std::any_of(
                           p->children.cbegin(),
                           p->children.cend(),
                           [](const auto & pp)
                           {
                               return pp and pp->isWordEnd;
                           }
                   ) :
                   p->children[c - kFirstChar] and p->children[c - kFirstChar]->isWordEnd;
        }

        if (c == '.')
        {
            for (char cc = 'a'; cc <= 'z'; ++cc)
            {
                if (search(word, index + 1, p->children[cc - kFirstChar].get()))
                {
                    return true;
                }
            }

            return false;
        }
        else
        {
            return search(word, index + 1, p->children[c - kFirstChar].get());
        }
    }

    std::unique_ptr<Node> root;
};


int main(int argc, char * argv[])
{
    Trie t;
    t.add("add");
    t.add("eat");
    t.add("addition");

    std::printf("%d\n", t.search("add"));
    std::printf("%d\n", t.search("eat"));
    std::printf("%d\n", t.search("addition"));
    std::printf("%d\n", t.search("adc"));
    std::printf("%d\n", t.search("ad."));
    std::printf("%d\n", t.search(".d."));
    std::printf("%d\n", t.search("add."));

    return EXIT_SUCCESS;
}
