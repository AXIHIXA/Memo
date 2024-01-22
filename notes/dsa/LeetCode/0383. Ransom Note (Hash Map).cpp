int init = []
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

class Solution
{
public:
    bool canConstruct(std::string ransomNote, std::string magazine)
    {
        if (magazine.size() < ransomNote.size()) return false;
        
        std::array<int, std::numeric_limits<char>::max() + 10> typeCount {0};
        for (char c : magazine) ++typeCount[c];

        for (char c : ransomNote)
        {
            if (--typeCount[c] < 0)
            {
                return false;
            }
        }

        return true;
    }
};