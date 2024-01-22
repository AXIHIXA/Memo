class Solution
{
public:
    bool isAnagram(string s, string t)
    {
        if (s.size() != t.size()) return false;
        
        std::array<int, 26> count {0};
        for (char c : s) ++count[c - 'a'];
        for (char c : t) --count[c - 'a'];
        return std::all_of(count.cbegin(), count.cend(), [](const int i) { return i == 0; });
    }
};