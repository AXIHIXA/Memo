class Solution 
{
public:
    int minSwaps(string s) 
    {
        if (s.empty())
        {
            return 0;
        }

        int extraClose = 0;
        int ans = 0;

        for (char c : s)
        {
            c == ']' ? ++extraClose : --extraClose;
            ans = max(ans, extraClose);
        }

        return (ans + 1) / 2;
    }
};