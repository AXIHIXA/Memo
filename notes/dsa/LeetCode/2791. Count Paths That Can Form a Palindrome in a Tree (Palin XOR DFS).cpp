// #define DEBUG

#ifdef DEBUG
#define dprintf(S, ...) do { printf(S, ##__VA_ARGS__); } while (false)
#define dfor(E, S) for (const auto & (E) : (S))
#else
#define dprintf(S, ...)
#define dfor(E, S)
static const int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    // std::setvbuf(stdin, nullptr, _IOFBF, 1 << 20);
    // std::setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
    return 0;
}();
#endif  // #ifdef DEBUG

class Solution
{
public:
    long long countPalindromePaths(std::vector<int> & parent, std::string s)
    {
        auto n = static_cast<int>(parent.size());
        if (n == 1) return 0;

        std::vector<std::vector<int>> child(n);
        for (int i = 1; i < n; ++i) child[parent[i]].emplace_back(i);

        long long ans = 0LL;

        // hmp.at(mask) == k:
        // k nodes have the same parity-of-chars on their paths to root. 
        std::unordered_map<int, int> hmp;
        hmp.emplace(0, 1);

        std::stack<std::pair<int, int>> st;
        st.emplace(0, 0);

        while (!st.empty())
        {
            auto [p, mask] = st.top();
            st.pop();
            
            for (int c : child[p])
            {
                int x = mask ^ (1 << (s[c] - 'a'));
                auto it = hmp.find(x);
                if (it == hmp.end()) it = hmp.emplace(x, 0).first;
                ans += it->second++;
                
                // bincount(x ^ y) == 1 iff.  
                // x and y differs at exactly one bit position. 
                // y is iterated as follows. 
                for (int i = 0; i < 26; ++i)
                {
                    int y = x ^ (1 << i);
                    it = hmp.find(y);
                    if (it != hmp.end()) ans += it->second;
                }

                st.emplace(c, x);
            }
        }

        return ans;
    }
};