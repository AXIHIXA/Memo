class Solution 
{
public
    int lengthOfLongestSubstring(string s) 
    {
        if (s.empty())
        {
            return 0;
        }

        int ans = 1;

         For the current longest substr ending at s[j], and considering s[j + 1], 
         If s[j + 1] occurs more than once, 
         drop the leftmost characters until there are no duplicate characters.

         mp[c] denotes the index of the leftmost char c so far. 
        stdvectorint mp(10 + stdnumeric_limitscharmax(), -1);
        mp[s[0]] = 0;

        for (int i = 0, j = 1; j != s.size(); ++j)
        {
            if (int k = mp[s[j]]; -1  k)
            {
                i = max(i, k + 1);
            }

            ans = max(ans, j - i + 1);
            mp[s[j]] = j;
        }

        return ans;
    }
};