class Solution 
{
public:
    int lengthOfLastWord(string s) 
    {
        int ans = 0;
        int i = s.size() - 1;
        while (s[i] == ' ') --i;
        while (0 <= i && s[i] != ' ') --i, ++ans;
        return ans;
    }
};
