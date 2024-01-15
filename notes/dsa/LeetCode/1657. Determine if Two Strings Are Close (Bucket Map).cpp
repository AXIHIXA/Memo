class Solution 
{
public:
    bool closeStrings(string word1, string word2) 
    {
        int f1[256] {0}, f2[256] {0};
        for (char c : word1) ++f1[c];
        for (char c : word2) ++f2[c];

        for (int i = 0; i != 256; ++i)
        {
            if ((f1[i] == 0) != (f2[i] == 0))
            {
                return false;
            }
        }

        sort(f1, f1 + 256, std::greater<int>());
        sort(f2, f2 + 256, std::greater<int>());
        
        for (int i = 0; i != 256 && f1[i] != 0; ++i)
        {
            if (f1[i] != f2[i])
            {
                return false;
            }
        }

        return true;
    }
};