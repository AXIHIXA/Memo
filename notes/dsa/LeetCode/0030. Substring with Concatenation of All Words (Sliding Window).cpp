class Solution
{
public:
    vector<int> findSubstring(string s, vector<string> & words)
    {
        std::unordered_map<std::string, int> wordCount;

        for (const std::string & w : words)
        {
            if (auto it = wordCount.find(w); it == wordCount.end())
            {
                wordCount.emplace_hint(it, w, 1);
            }
            else
            {
                ++it->second;
            }
        }

        std::vector<int> ans;
        int numWords = words.size();
        int wordLength = words[0].size();

        for (int i = 0, wl = words[0].size(); i < wl; ++i)
        {
            slidingWindow(s, i, wordCount, numWords, wordLength, ans);
        }

        return ans;
    }

private:
    static void 
    slidingWindow(
        const std::string & s, 
        int left, 
        const std::unordered_map<std::string, int> & wordCount,
        int numWords, 
        int wordLength, 
        std::vector<int> & ans
    )
    {
        int n = s.size();
        int substrSize = wordLength * numWords;

        std::unordered_map<std::string, int> wordsFound;
        int wordsUsed = 0;
        bool excessWord = false;

        for (int right = left; right <= n - wordLength; right += wordLength)
        {
            std::string sub = s.substr(right, wordLength);
            
            if (wordCount.find(sub) == wordCount.end())
            {
                // Mismatched word. Reset the window. 
                wordsFound.clear();
                wordsUsed = 0;
                excessWord = false;
                left = right + wordLength;
            }
            else
            {
                // Reached max window size, or had an excess word. 
                while (right - left == substrSize || excessWord)
                {
                    std::string leftmostWord = s.substr(left, wordLength);
                    left += wordLength;

                    if (wordCount.at(leftmostWord) <= --wordsFound.at(leftmostWord))
                    {
                        excessWord = false;
                    }
                    else
                    {
                        --wordsUsed;
                    }
                }

                // Keep track of how many times this word occurs in the window. 
                if (auto it = wordsFound.find(sub); it == wordsFound.end())
                {
                    wordsFound.emplace_hint(it, sub, 1);
                }
                else
                {
                    ++it->second;
                }
                
                if (wordsFound.at(sub) <= wordCount.at(sub))
                {
                    ++wordsUsed;
                }
                else
                {
                    excessWord = true;
                }
                
                if (wordsUsed == numWords && !excessWord)
                {
                    ans.emplace_back(left);
                }
            }
        }
    }
};