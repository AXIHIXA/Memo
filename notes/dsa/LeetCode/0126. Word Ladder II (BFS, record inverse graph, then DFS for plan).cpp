class Solution
{
public:
    std::vector<std::vector<std::string>> findLadders(
            std::string beginWord, 
            std::string endWord, 
            std::vector<std::string> & wordList)
    {
        std::unordered_set<std::string> wordDict(wordList.cbegin(), wordList.cend());

        if (!wordDict.contains(endWord))
        {
            return {};
        }

        auto len = static_cast<const int>(beginWord.size());

        std::unordered_set<std::string> currLevel;
        currLevel.emplace(beginWord);

        int level = 0;
        int maxLevel = std::numeric_limits<int>::max();

        // Inverse graph: 
        // For directed edge (s, t), we record (t, s). 
        // TLE if we use regular graph (too many initial choices...)
        std::unordered_map<std::string, std::unordered_set<std::string>> g;

        while (!currLevel.empty() && level <= maxLevel)
        {
            for (const auto & s : currLevel)
            {
                wordDict.erase(s);
            }

            std::unordered_set<std::string> nextLevel;
            
            for (const auto & s : currLevel)
            {
                if (s == endWord)
                {   
                    maxLevel = level;
                }

                for (int i = 0; i < len; ++i)
                {
                    char ori = s[i];
                    std::string t = s;

                    for (char c = 'a'; c <= 'z'; ++c)
                    {
                        if (c == ori)
                        {
                            continue;
                        }

                        t[i] = c;

                        if (wordDict.contains(t))
                        {
                            nextLevel.insert(t);
                            g[t].insert(s);
                        }
                    }
                }
            }

            ++level;
            currLevel = std::move(nextLevel);
        }

        if (maxLevel == std::numeric_limits<int>::max())
        {
            return {};
        }

        // Retrieve path via DFS of the inverse graph. 
        std::vector<std::vector<std::string>> ans;
        std::vector<std::string> path;
        path.reserve(maxLevel);
        path.push_back(endWord);

        std::function<void ()> dfs = 
        [&beginWord, &g, &ans, &path, &dfs]()
        {
            const std::string & s = path.back();
            
            if (s == beginWord)
            {
                ans.emplace_back(path.crbegin(), path.crend());
                return;
            }

            for (const auto & t : g[s])
            {
                path.push_back(t);
                dfs();
                path.pop_back();
            }
        };

        dfs();

        return ans;
    }
};