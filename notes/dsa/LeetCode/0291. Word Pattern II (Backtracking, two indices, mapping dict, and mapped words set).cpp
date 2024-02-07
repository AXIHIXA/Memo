class Solution
{
public:
    bool wordPatternMatch(std::string pattern, std::string s)
    {   
        // p2s: Mapping from a char in pattern into a string in s. 
        // words: All mapped strings, avoid multiple-on-one mappings. 
        std::unordered_map<char, std::string> p2s;
        std::unordered_set<std::string> words;
        return backtrack(pattern, 0, s, 0, p2s, words);
    }

private:
    static bool backtrack(
            const std::string & p, int i, 
            const std::string & s, int j, 
            std::unordered_map<char, std::string> & p2s,
            std::unordered_set<std::string> & words
    )
    {
        if (i == p.size() || j == s.size())
        {
            return i == p.size() && j == s.size();
        }

        auto it = p2s.find(p[i]);

        // p[i] is a mapped char. 
        if (it != p2s.end())
        {
            const std::string & word = it->second;
            return s.substr(j, word.size()) == word ? 
                   backtrack(p, i + 1, s, j + word.size(), p2s, words) : 
                   false;
        }

        // p[i] is a new char. Iterate all possible mappings for p[i]. 
        for (int k = j; k < s.size(); ++k)
        {
            std::string word = s.substr(j, k - j + 1);
            if (words.find(word) != words.end()) continue;

            auto jt = words.emplace(word).first;
            it = p2s.emplace(p[i], word).first;

            if (backtrack(p, i + 1, s, k + 1, p2s, words)) return true;

            words.erase(jt);
            p2s.erase(it);
        }

        return false;
    }
};