class Solution 
{
public:
    int countBinarySubstrings(string s) 
    {
        int res = 0;
        int prevGpSz = 0;
        int currGpSz = 1;

        for (int i = 1; i != s.size(); ++i)
        {
            if (s[i - 1] != s[i])
            {
                res += min(prevGpSz, currGpSz);
                prevGpSz = currGpSz;
                currGpSz = 1;
            }
            else
            {
                ++currGpSz;
            }
        }

        return res + min(prevGpSz, currGpSz);
    }
};