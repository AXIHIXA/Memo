class Solution
{
public:
    bool wordPattern(string pattern, string s)
    {
        std::unordered_map<char, int> pMap;
        std::unordered_map<std::string, int> sMap;

        int i = 0;
        std::stringstream sin(s);
        std::string line;

        for ( ; std::getline(sin, line, ' '); ++i)
        {
            char c = pattern[i];
            if (pMap.find(c) == pMap.end()) pMap.emplace(c, i);
            if (sMap.find(line) == sMap.end()) sMap.emplace(line, i);
            if (pMap.at(c) != sMap.at(line)) return false;
        }

        return i == pattern.size();
    }
};