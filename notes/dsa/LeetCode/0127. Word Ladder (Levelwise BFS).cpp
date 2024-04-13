class Solution
{
public:
    int ladderLength(std::string beginWord, std::string endWord, std::vector<std::string> & wordList)
    {
        std::unordered_set<std::string> unvisited(wordList.cbegin(), wordList.cend());
        
        if (!unvisited.contains(endWord))
        {
            return 0;
        }
        
        auto wordSize = static_cast<const int>(beginWord.size());

        std::queue<std::string> que;
        que.emplace(beginWord);

        unvisited.erase(beginWord);

        int level = 0;

        while (!que.empty())
        {
            auto levelSize = static_cast<const int>(que.size());

            for (int i = 0; i < levelSize; ++i)
            {
                std::string s = que.front();
                que.pop();

                if (s == endWord)
                {
                    return level + 1;
                }

                for (int i = 0; i < wordSize; ++i)
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

                        if (!unvisited.contains(t))
                        {
                            continue;
                        }
                        
                        unvisited.erase(t);
                        que.emplace(t);
                    }
                }
            }
            
            ++level;
        }

        return 0;
    }
};