class Solution 
{
public:
    int lengthOfLongestSubstringKDistinct(string s, int k) 
    {
        if (k == 0)
        {
            return 0;
        }

        if (s.size() == 1)
        {
            return 1;
        }

        std::vector<int> frequency(10 + std::numeric_limits<char>::max(), 0);
        int numDistinctChars = 0;

        int ans = 0;

        for (int j = 0; j != s.size(); ++j)
        {
            numDistinctChars += (frequency[s[j]]++ == 0);

            if (numDistinctChars <= k)
            {
                ++ans;
            }
            else
            {
                numDistinctChars -= (--frequency[s[j - ans]] == 0);
            }
        }

        return ans;
    }
};