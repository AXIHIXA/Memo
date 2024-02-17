class Solution
{
public:
    std::vector<int> findSubstring(std::string s, std::vector<std::string> & words)
    {
        std::unordered_map<std::string_view, int> wordCount;
        for (const std::string & w : words) ++wordCount[w];

        std::vector<int> ans;

        for (int i = 0; i < words[0].size(); ++i)
            slidingWindow(s, i, wordCount, words.size(), words[0].size(), ans);

        return ans;
    }

private:
    static void slidingWindow(
        const std::string & s, 
        int left, 
        const std::unordered_map<std::string_view, int> & wordCount,
        const int numWords, 
        const int wordLength, 
        std::vector<int> & ans
    )
    {
        auto n = static_cast<const int>(s.size());
        const int substrSize = wordLength * numWords;

        std::unordered_map<std::string_view, int> wordsFound;
        int wordsUsed = 0;
        bool excessWord = false;

        for (int i = left; i <= n - wordLength; i += wordLength)
        {
            std::string_view sub(s.data() + i, wordLength);

            // Mismatched word at s[i:i + wordLength), reset window. 
            if (!wordCount.contains(sub))
            {
                wordsFound.clear();
                wordsUsed = 0;
                excessWord = false;
                left = i + wordLength;
                continue;
            }
            
            // Reached max window size, or had an excess word. 
            // Shrink window size until size is valid and no excess word is present. 
            while (i - left == substrSize || excessWord)
            {
                std::string_view leftmostWord(s.data() + left, wordLength);
                left += wordLength;

                if (wordCount.at(leftmostWord) <= --wordsFound.at(leftmostWord))
                    excessWord = false;
                else 
                    --wordsUsed;
            }

            // Keep track of how many times this word occurs in the window. 
            ++wordsFound[sub];
            
            if (wordsFound.at(sub) <= wordCount.at(sub))
                ++wordsUsed;
            else
                excessWord = true;
            
            if (wordsUsed == numWords && !excessWord)
                ans.emplace_back(left);
        }
    }
};