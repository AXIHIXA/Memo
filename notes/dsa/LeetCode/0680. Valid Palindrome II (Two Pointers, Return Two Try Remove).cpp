class Solution 
{
public:
    bool validPalindrome(string s) 
    {
        if (s.size() < 3) return true;

        for (int i = 0, j = s.size() - 1; i < j; ++i, --j)
        {
            if (s[i] != s[j])
            {
                // Note: TLE if not returning here. 
                // (Duplicates checks if not returning.)
                return checkPalindrome(s, i + 1, j) || checkPalindrome(s, i, j - 1);
            }
        }

        return true;
    }

private:
    static bool checkPalindrome(const std::string & s, int i, int j)
    {
        while (i < j)
        {
            if (s[i] != s[j]) return false;
            ++i, --j;
        }
        
        return true;
    }
};