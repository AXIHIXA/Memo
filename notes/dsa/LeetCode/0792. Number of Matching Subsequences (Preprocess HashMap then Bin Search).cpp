class Solution
{
public:
    int numMatchingSubseq(std::string s, std::vector<std::string> & words)
    {
        auto m = static_cast<const int>(s.size());
        std::array<std::vector<int>, 26> mp;
        for (int i = 0; i < m; ++i) mp[s[i] - 'a'].emplace_back(i);

        int ans = 0;

        for (const std::string & w : words)
        {
            if (m < w.size()) continue;

            int i = -1;

            for (char c : w)
            {
                const std::vector<int> & cmap = mp[c - 'a'];
                auto it = std::upper_bound(cmap.cbegin(), cmap.cend(), i);

                if (it == cmap.end())
                {
                    i = -1; 
                    break;
                }

                i = *it;
            }

            if (0 <= i && i < m) ++ans;
        }

        return ans;
    }
};