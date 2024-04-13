class Bfs
{
public:
    int ladderLength(
            std::string beginWord, 
            std::string endWord, 
            std::vector<std::string> & wordList)
    {
        std::unordered_set<std::string> wordDict(wordList.cbegin(), wordList.cend());
        
        if (!wordDict.contains(endWord))
        {
            return 0;
        }
        
        auto wordSize = static_cast<const int>(beginWord.size());

        std::queue<std::string> que;
        que.emplace(beginWord);
        wordDict.erase(beginWord);

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
                        t[i] = c;
                        
                        if (c == ori || !wordDict.contains(t))
                        {
                            continue;
                        }

                        wordDict.erase(t);
                        que.emplace(t);
                    }
                }
            }
            
            ++level;
        }

        return 0;
    }
};

class BidirBfs
{
public:
    int ladderLength(
            std::string beginWord, 
            std::string endWord, 
            std::vector<std::string> & wordList)
    {
        std::unordered_set<std::string> wordDict(wordList.cbegin(), wordList.cend());

        if (!wordDict.contains(endWord))
        {
            return 0;
        }

        std::array<std::unordered_set<std::string>, 2> currLevel;
        std::array<int, 2> level = {0, 0};

        currLevel[0].emplace(beginWord);
        currLevel[1].emplace(endWord);

        wordDict.erase(beginWord);
        wordDict.erase(endWord);

        auto len = static_cast<const int>(beginWord.size());

        while (!currLevel[0].empty() || !currLevel[1].empty())
        {
            int side = currLevel[0].empty() ? 1 : currLevel[1].size() < currLevel[0].size();
            auto levelSize = static_cast<const int>(currLevel[side].size());
            std::unordered_set<std::string> nextLevel;

            for (const auto & s : currLevel[side])
            {
                // std::printf("side %d pop %s\n", side, s.c_str());

                for (int i = 0; i < len; ++i)
                {
                    char ori = s[i];
                    std::string t = s;

                    for (char c = 'a'; c <= 'z'; ++c)
                    {
                        t[i] = c;

                        if (currLevel[side ^ 1].contains(t))
                        {
                            return level[side] + level[side ^ 1] + 2;
                        }

                        if (c == ori || !wordDict.contains(t))
                        {
                            continue;
                        }

                        wordDict.erase(t);
                        nextLevel.emplace(t);

                        // std::printf("side %d enqueue %s\n", side, t.c_str());
                    }
                }
            }

            ++level[side];
            currLevel[side] = std::move(nextLevel);
        }

        return 0;
    }
};

// using Solution = Bfs;
using Solution = BidirBfs;
