static constexpr int kMaxSize = 1'000'000;
std::array<std::array<int, 26>, kMaxSize> tree = {{0}};
std::array<int, kMaxSize> pass = {0};
int cnt = 1;

void insert(const std::string & word)
{
    int cur = 1;
    ++pass[cur];

    for (int i = 0, path; i < word.size(); ++i)
    {
        path = word[i] - 'a';
        if (!tree[cur][path]) tree[cur][path] = ++cnt;
        cur = tree[cur][path];
        ++pass[cur];
    }
}

int solve(const std::string & word)
{
    int cur = 1;
    int ans = 0;

    for (int i = 0, path; i < word.size(); ++i)
    {
        path = word[i] - 'a';
        cur = tree[cur][path];
        ans += pass[cur];
    }

    return ans;
}

void clear()
{
    cnt = 1;
    for (auto & node : tree) std::fill(node.begin(), node.end(), 0);
    std::fill(pass.begin(), pass.end(), 0);
}

class Solution
{
public:
    std::vector<int> sumPrefixScores(std::vector<std::string> & words)
    {
        auto n = static_cast<const int>(words.size());

        clear();
        for (const std::string & word : words) insert(word);

        std::vector<int> ans(n);
        for (int i = 0; i < n; ++i) ans[i] = solve(words[i]);

        return ans;
    }
};