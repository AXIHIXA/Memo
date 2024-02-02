class Solution
{
public:
    std::string alienOrder(std::vector<std::string> & words)
    {
        static constexpr int kNumLowChar = 26;
        std::vector<std::vector<int>> am(kNumLowChar, std::vector<int>(kNumLowChar, 0));
        
        std::vector<int> inDegree(kNumLowChar, -1);
        for (const auto & w : words) for (char c : w) inDegree[c - 'a'] = 0;

        int kinds = 0;
        for (int i = 0; i < kNumLowChar; ++i) if (inDegree[i] != -1) ++kinds;

        // Build graph with consecutive pairs of words. 
        for (int i = 0, ws = words.size(); i < ws - 1; ++i)
        {
            const std::string & a = words[i];
            const std::string & b = words[i + 1];
            int rr = std::min(a.size(), b.size());
            int j = 0;

            for ( ; j < rr; ++j) 
            {
                if (a[j] != b[j])
                {
                    if (!am[a[j] - 'a'][b[j] - 'a']) ++inDegree[b[j] - 'a'];
                    am[a[j] - 'a'][b[j] - 'a'] = 1;
                    break;
                }
            }

            if (j < a.size() && j == b.size()) return "";
        }

        std::string ans;
        ans.reserve(kNumLowChar);

        // Topological sort. 
        std::queue<int> qu;
        for (int i = 0; i < kNumLowChar; ++i) if (inDegree[i] == 0) qu.emplace(i);

        while (!qu.empty())
        {
            int i = qu.front();
            qu.pop();

            ans += i + 'a';

            for (int j = 0; j < kNumLowChar; ++j) 
                if (am[i][j] && --inDegree[j] == 0)
                    qu.emplace(j);
        }

        return ans.size() == kinds ? ans : "";
    }
};