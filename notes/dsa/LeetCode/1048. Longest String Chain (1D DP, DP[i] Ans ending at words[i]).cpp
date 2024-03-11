class Solution
{
public:
    int longestStrChain(std::vector<std::string> & words)
    {
        auto n = static_cast<const int>(words.size());

        std::unordered_map<std::string, int> wds;
        for (int i = 0; i < n; ++i) wds.emplace(words[i], i);
        
        // dp[i]: Length of longest chain ending with words[i]. 
        std::vector<int> dp(n, 0);

        std::function<int (int)> dfs = [&words, &wds, &dp, &dfs](int i) -> int
        {
            if (dp[i]) return dp[i];
            
            const std::string & s = words[i];
            auto len = static_cast<const int>(s.size());

            int tmp = 0;

            for (int i = 0; i < len; ++i)
            {
                std::string pred = s.substr(0, i) + s.substr(i + 1);
                auto it = wds.find(pred);
                if (it != wds.end()) tmp = std::max(tmp, dfs(it->second));
            }

            return dp[i] = 1 + tmp;
        };

        for (int i = 0; i < n; ++i)
        {
            dfs(i);
        }

        return *std::max_element(dp.cbegin(), dp.cend());
    }
};